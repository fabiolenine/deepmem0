# Changelog

## Unreleased

### `AsyncMemory.delete_all` crashed on an empty scope

`hit_page_cap` was assigned only at the end of the pagination loop body, so the
`break` taken on an empty first page skipped both the assignment and the `else`
clause, and the read further down raised `UnboundLocalError`. An empty scope is
the ordinary case — a wrong id, or a scope already drained.

It survived unnoticed because nothing calls `delete_all` in the deployment where
it was found; the bulk path is used instead. Code with no caller is not correct
code, it is unmeasured code.

### Scope identifiers are normalized at one place, and `delete_all` uses it

A scope filter is an EXACT value match. `" alice"` and `"alice"` are different
scopes to the vector store, and the difference never surfaces as an error — it
surfaces as an empty result. On a delete that is worse than a crash: nothing
matches, nothing is removed, and the call returns success.

`normalize_scope_id(value, name)` is now public in `mem0.memory.utils`, next to
the other identity primitives. `_validate_and_trim_entity_id` is a thin alias
over it. Making it public is the point: every boundary that builds a scope filter
needs the same rule, and depending on a leading-underscore name to get it means
depending on an implementation detail.

**Non-string ids no longer crash on `.strip()`**, but they are not blanket-coerced
either:

* `int` becomes `str` — a database primary key is a legitimate id, and `42` and
  `"42"` become the SAME scope rather than two;
* `bool`, `float`, and everything else are REJECTED with a message naming the
  parameter and the type received.

The asymmetry is deliberate. `str(42.0)` is `"42.0"`, which never matches the
`"42"` that the integer writes, and `str(True)` is `"True"`, a scope nobody wrote
to. Both would be silent scope splits. `bool` needs its own guard because
`isinstance(True, int)` is `True` in Python.

**`delete_all` validates again, in both twins.** The paginating rewrite of this
method dropped the validation calls, so the sync and async `delete_all` were the
only scope entry points accepting a raw argument. Validation runs before the
filter is built and before the truthiness test — otherwise `user_id=0` is
discarded and the call dies with "At least one filter is required", the wrong
error for the right defect.

Tests are parameterized over `user_id`, `agent_id` and `run_id` — the async twin
normalizes with three separate lines, so a test that only ever passes `user_id`
cannot see one of them being dropped. Success cases assert the filter actually
handed to the store, rejection cases assert the exception type, the message and
the parameter named in it, and each of the eleven guards was falsified by
reverting its own hunk alone and requiring the expected failure.

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
