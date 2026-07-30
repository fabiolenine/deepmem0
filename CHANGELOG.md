# Changelog

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
