# Changelog

## v0.11.1

Four defects in `parse_vision_messages`, which was byte-identical to upstream
mem0 2.0.7. Upstream PR #5631 fixes one of them, in the branch that matters
least. Nothing here changes production behaviour on a vision-disabled server
beyond a warning it should always have emitted.

## The wrapper that turned a permanent failure into four retries

`except Exception: raise Exception(f"Error while downloading {image_url}.")`
did three things wrong. It erased the exception chain. It stated a falsehood —
nothing is downloaded for a data URI or a local path. And it raised a bare
`Exception`, whose MRO matches nothing in a consumer's poison list, so a
classifier that distinguishes *bad payload* from *sick infrastructure* by
exception class sees an unknown type and calls it retryable. A payload that can
never succeed was re-added four times, each one a full LLM extraction.

The wrapper is removed, not replaced. A wrapper of any class would still change
the type consumers inspect; only the provider knows whether its own failure is
permanent, so the provider's exception now propagates untouched. This is what
lets `mem0/llms/ollama.py`'s actionable `ValueError`s — "only base64 data URIs
are supported", "http(s) image URLs are not supported" — reach the caller at
all. They were being thrown away.

## The guarded branch was the wrong one

Upstream hardened `content` as a bare dict, which is not a valid OpenAI shape —
it is a mem0-ism. The LIST branch is the canonical multimodal shape *and* the
shape `get_image_description` itself builds, and it had no guard whatsoever. A
single `_image_part_url` now validates both branches, and in the list branch it
validates **every** part before an LLM call is spent, not just the first.

The error string is upstream's, word for word, so a future rebase reconciles
trivially and upstream's own test regex matches.

## Dropping an image is no longer silent

With vision disabled the parser discarded image parts, and a message carrying
only an image vanished whole — the caller was told nothing. It now logs at
WARNING, distinguishing "dropped N images, kept the text" from "dropped N images
and discarded the whole message". This is the only change visible on a
vision-disabled server, which is every server that does not opt in.

The invariant is stated honestly: **not** "never raises", but "does not raise
for a well-formed message". A non-dict message still raises `AttributeError`
and a non-string text part is filtered rather than crashing the join — both
correct, because consumers classify those as poison.

## What the tests prove

Thirteen new cases, twelve of which fail against the previous code; the
thirteenth is declared characterization, not falsification. The six pre-existing
`TestParseVisionMessages` cases pass unedited — 163 additions, 0 deletions.

The gated live smoke now traverses `parse_vision_messages` instead of entering at
`get_image_description`, so the strict list branch has real coverage against a
real `qwen3-vl:4b-instruct`. It asserts on what went **on the wire** — the
base64 submitted to Ollama decodes to the exact PNG the test generated, and the
call routed to `vision_model` — because a VLM can name a colour it was never
shown. A malformed part is proven refused *before* any provider call by spying
on `client.chat` and asserting it was never invoked; the error message alone
would not prove ordering, since the Ollama adapter rejects the same part locally.

Full suite: failure identity sets compared before and after are byte-identical,
100 signatures each side, +14 passed.

## v0.11.0

Same change set as v0.10.1, released under a minor version because it adds
functionality — a per-language spaCy pipeline, `doc.ents` in the extractor,
technical-identifier recognition and the `delete_observer` hook — not only fixes.
v0.10.1 is superseded and kept for anyone who already fetched it.


The entity store was losing links in silence. This release fixes the cause at
every writer, gives the extractor a Portuguese pipeline, and adds the instruments
that made each claim checkable.

## Link integrity

**Read-after-write was not guaranteed.** `insert` called `client.upsert(...)` with
no `wait`, and Qdrant's default is not to wait: the write is acknowledged before
it is visible. A writer that reads back what it just wrote does not find it,
creates the row again, and the last `insert` REPLACES the point — deleting every
other writer's links. Entity writers now pass `wait=True`; the memory path keeps
the previous default.

Measured with a barrier-synchronised thread test and a negative control that must
show zero loss in series: **0 of 32 links lost with 16 concurrent writers**,
against **4 of 8 lost** with the fix disabled.

**A serialized list must not explode into characters.** A `str` payload fed to
`set()` iterates character by character; real rows lost their links that way.
`normalize_linked_memory_ids` is applied on every write path, and read paths fail
closed with a warning instead of silently skipping.

**Cleanup now reports whether it finished.** `unlink_memory_from_entity_rows`
returns a completeness verdict, and the delete intent is committed only after a
successful cleanup — committing over a failed cleanup is what turns a transient
error into a permanent dangling link. A truncated scan returns `False`.

**`delete_all` paginates** and terminates on "nothing new" rather than "page
empty", which previously could loop.

## Entity identity

Identity was vector similarity (>= 0.95) — probabilistic. Case variants of one
entity became separate rows, each holding a slice of the links, so which row
received the ranking boost depended on the spelling the user typed.

- `data_normalized` (NFKC + casefold + collapsed whitespace) is the identity, with
  a payload index. Separators are deliberately not collapsed: in a technical
  corpus `num_ctx` and `num ctx` are different things.
- The point id is derived from `(scope, normalized key)`, so two writers that find
  nothing write to the SAME point instead of creating two rows.
- Links are also stored one key per link (`lnk_<id>`), because `set_payload`
  merges KEYS but replaces a LIST value. Unlinking deletes the key — leaving it
  behind resurrected the link through the union.
- All four writers share the same rules: sync, async, and both Phase 7 batch
  paths. The batch path is the one that runs on every `add(infer=True)`.

## Entity extraction

The extractor never consulted `doc.ents` and ran an English pipeline over
non-English text. Four phases, each measured on its own against a 61-case span
golden (46 failing cases at the start, 0 at the end):

- **span hygiene** — `in`/`at`/`for`/`is` leave the proper-noun whitelist (they
  glued distinct entities into one span, and the substring cleanup then deleted
  the short one); 5-token / 60-char caps; the fallback branch no longer appends a
  verb; a PROPER is never suppressed by a longer span containing it.
- **per-language lexicon** — word lists were 100% English. Uppercase emphasis is
  told from an acronym by MORPHOLOGY, not length: a length rule deleted
  `PYTHONPATH`.
- **Portuguese model and `doc.ents`** — `spacy_models` is a cache per language.
  Over Portuguese, the English pipeline tags every token of "Ontem eu viajei para
  Recife" as PROPN, pronoun and verb included, which makes every POS-keyed rule
  inert. A missing model for a configured non-default language now RAISES instead
  of falling back in silence (`MEM0_SPACY_STRICT=0` opts out explicitly).
- **technical identifiers** — `num_ctx`, `bge-m3`, `linked_memory_ids` and the
  like were all lost. POS does not help (the same tokens come back VERB, ADJ,
  NOUN, NUM); shape does. A digit or an underscore is required, or hyphenated
  common words would become entities.

Language models are declared as extras (`nlp-en`, `nlp-pt`) with PEP 508 direct
references — spaCy models are not on PyPI, and leaving them undeclared is what let
a Portuguese deployment run on the English pipeline.

## Observability

`delete_observer` exposes what only the core can see: `rows_scanned`,
`rows_touched`, `truncated`, `complete` and elapsed time. `truncated` is the only
signal that a cleanup may have left a link behind. Instrumenting this from outside
was impossible without reimplementing the function.

## Tests

`tests/known_fork_failures.json` records the pre-existing failures by SIGNATURE
(nodeid + exception class + normalized message), so a NEW defect that replaces an
old cause inside the same test is detected instead of counted as reproduced. 128
signatures over 1421 collected tests; the whole change set introduces zero new
failures.
