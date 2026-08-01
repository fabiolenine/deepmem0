# Changelog

## v0.15.0

Speaker attribution: **who said this**, per extracted fact. A minor version
because it changes what reads return and what writes store, not because the
shape of the API moved.

### `attributed_to` was a write-only dead value

The field was written on every `add` with `infer=True` and returned by no
reader. It sat in `core_and_promoted_keys` — which EXCLUDES it from the
`metadata` bucket — and was absent from `promoted_payload_keys`, which is what
actually copies to the result. It fell through the gap between the two.

Measured in a production corpus before the fix: **1079 of 1218 memories (88.6%)**
carried it and none of them showed it. It is now promoted at all six read sites
(get/get_all/search × sync/async), guarded by `if key in payload`.

The documented vocabulary said `user | assistant`. Production also holds
`document`, emitted under the document-ingestion prompt. The real vocabulary has
three values and is now written down as such; nothing is validated
retroactively, because rejecting `document` would invalidate 171 legitimate
memories.

### The extractor could not see the speaker

`parse_messages` rendered `role: content` and dropped the OpenAI-style `name`
field, so the prompt's own instruction to attribute facts "to the speaker by
name" could not be followed — the input never carried the name. Turns with a
speaker now render as `role (Speaker): content`; turns without one render
byte-identically to before.

`parse_vision_messages` rebuilds a message from scratch in its three multimodal
branches, and the speaker evaporated there. Attribution would have worked on
plain text and silently vanished on a message with an image.

### The speaker label is sanitized before it reaches the prompt

The label is caller text entering a prompt whose grammar is one turn per line. A
`name` containing a newline forges entire turns in the conversation the
extractor reads — prompt injection through a structured field, not through free
content. Labels are NFKC-normalized, whitespace-collapsed, and rejected outright
on control characters or excess length. Rejection is total: the turn becomes
anonymous rather than half-sanitized, because a half-sanitized label is one the
caller never wrote and no one can query later.

Deliberately NOT casefolded, for the same reason `user_id` is not: it would
merge distinct people. The declared cost is the inverse — `Maria` and `maria`
are different speakers.

### The model proposes, the code decides

A conversation where every extractable turn carries the SAME speaker is resolved
in code, with no model involvement. That is not "one distinct name": a
conversation of `[user name=Maria, assistant unnamed]` has exactly one distinct
name, and attributing everything to Maria would hand her the facts the assistant
produced. Uniformity across all extractable turns is the condition.

Otherwise a conditional prompt suffix enumerates the CLOSED SET of speakers that
actually reached the rendered prompt, and the emitted value is stored only if it
is a `str` that canonicalizes into that set. Wrong type, invented name, or a
label from a turn the model never saw: the field is omitted. Absence is exactly
the previous behaviour and is safe; a wrong label is corruption nothing detects
by looking at the result.

The suffix exists ONLY on the path that needs it. Conversations with no speaker
— all of today's traffic — and uniform conversations pay zero tokens. That is
the `num_ctx` budget, not an optimization: the extraction prompt floor is
already ~42% of the window, and this system has measured a silent total loss of
facts when the prompt approached the ceiling.

### Ownership scope was immutable in one update path and forgeable in the other

`_build_version_metadata` (versioned update) already imposed the head's exact
ownership scope *including its absence*. The legacy in-place update only knew how
to PRESERVE an existing value, so `update(id, data, metadata={"actor_id": "X"})`
stamped authorship onto a memory that had none — and "none" is the state of every
legacy memory. The guard protected a written value and accepted writing one from
scratch, which stops authorship from being rewritten but not from being forged.
The same asymmetry applied to `user_id`/`agent_id`/`run_id`, i.e. scope
escalation.

Both paths now call one function, because it was the divergence between them
that opened the hole. The rule the module's own comment already declared —
*"a caller can neither change nor ADD a scope the head does not have"* — now
holds on both.

### Reading by speaker

`actor_id` was already promoted and indexed; only the write was missing. Read
filters now canonicalize `actor_id` with the SAME function as the write, because
Qdrant matches exactly and a filter that diverges by one space returns nothing
without erroring. An unusable label raises instead of being dropped from the
filter: dropping it would widen the query to the whole scope and return every
speaker's memories as if they were one speaker's.

`attributed_to` gains a payload index, created online on existing collections at
the next startup.

### Config

`MEM0_SPEAKER_ATTRIBUTION` (default on) disables the suffix, the write, and the
scope rule together. Read per call, so a service restart is enough. An
unrecognized value falls back to ON: a typo must not silently disable a feature
the operator believes is running.

### Measured

`eval/eval_speaker_attribution.py` — hard-gates the MECHANISM (uniform path is
deterministic and costs zero tokens; mixed named/anonymous does not take the fast
path; nothing outside the closed set is ever stored; the kill switch really
kills; prompt budget). Attribution QUALITY is reported but does not gate, because
its oracle is judgement over a 9B model's output and a gate that oscillates is
not a gate.

On a reference deployment with a local 9B extractor: suffix costs 271 tokens;
worst-case prompt 50.9% of `num_ctx`; uniform path delta ZERO tokens. Attribution
coverage and accuracy 6/6 and 6/6 over 3 repetitions of a two-speaker
conversation with existing memories in the prompt — one conversation shape, not a
survey.

The first measurement of that eval found the opposite: with the suffix phrased as
an "optional" field the model emitted it on an empty collection and OMITTED it
once the prompt also carried existing memories. What fixed it was an explicit
mapping example plus making the field required-when-a-speaker-is-shown. Worth
recording because the mechanism was already correct at that point — only the
model's compliance was not, and a suffix the extractor ignores delivers nothing.

⚠️ Existing memories are NOT back-filled. Retroactive attribution would need the
original messages, which are not kept.

## v0.14.0

A minor version because entity identity changes behaviour, not because anything
new is exposed. The theme: a lookup that disagreed with the id it would write to.

### Entity scope matched by subset, and it corrupted a real corpus

A Qdrant filter on `{user_id: U}` also matches rows that carry a `run_id`, and
both entity writers validated only the keys they had SEARCHED for — never the
absence of the others. Measured in production: an entity row scoped to a test
run accumulated **12 links from memories in the broad scope**, written by the
production worker, while the correct broad-scope row sat untouched. The
deterministic point id was already exact — `f({user_id})` differs from
`f({user_id, run_id})` — so finding and writing disagreed by construction.

`escopo_exato()` now requires equality on all three scope keys, in both
directions, which realigns the two.

### Two identity rules in one corpus, the weaker one on the hottest path

Phase 7 — the writer that runs on every `add` with `infer=True` — was still on
the vector-probe-only rule that the exact-key lookup had replaced in
`_upsert_entity`. Both writers now share the same primitives, and the
single-entity decision lives in one module function that the sync and async
twins call, because those twins have already diverged on exactly this decision
once.

### Every uncertain outcome fails closed

Insert uses the deterministic id and REPLACES the payload, so "I don't know"
that turns into a write is data loss. A store error on the exact lookup skips
the entity instead of probe-then-insert; more than one row for a key is
reported, never silently chosen; a saturated lookup limit raises, because a
bounded `top_k` is consumed before the Python-side scope validation and the
exact row may have fallen outside it; and `search_batch` must return one LIST
per query — an entry that is merely iterable yields zero matches and would
therefore insert.

Batch insert now passes `wait=True` on both twins.

### Batched entity linking on the update path

The update path did one embed per entity, and the embedder's cost is dominated
by the CALL (~450-590 ms) rather than the item (~12 ms short, ~62 ms at ~1.8k
chars). `vincular_entidades_em_lote` does one embed, one lookup, one insert.

Measured on the UPDATE branch with the same harness on both sides, 11
repetitions, warm-up discarded, dispersion max/min 1.1-1.9x, and links re-read
from the store to confirm the write: **N=2 2.1x, N=4 3.5x, N=8 5.5x, N=16 7.0x**
(8844 -> 1268 ms). No gain at N=1, as expected. The insert branch is a different
animal — it pays `wait=True` plus reconciliation — and measuring that one to
draw conclusions about this one was a real error along the way.

### `embed_batch` is chunked for Ollama

`MEM0_EMBED_MAX_BATCH`, default 256. It was the only provider sending an
unbounded list. The justification is not what one would assume: there is no
reachable failure boundary — measured OK at **32768 items in a single request**,
`ms/item` flat at 20-23, model VRAM constant. The run stopped on a chosen
wall-time threshold (744 s per call), not on an error.

What the cap limits is call latency and the blast radius of one dead request.
`ms/item` flattens at ~256 and does not improve above it, so a larger batch buys
risk for nothing, and chunking costs about 1%.

Counts are validated PER CHUNK: a short chunk compensated by a long one would
pass a total-only check, and callers match vector to text by POSITION.

### `infer=False` embeds once, not once per message

Discard rules (system role, malformed dict) and per-item failure semantics are
preserved — one bad embed drops one message, never the batch.

## v0.13.0

A minor version because ranking constants change value, not because anything new
is exposed. The theme is the one from v0.12.0 seen from the other side: there,
values that were wrong in a way that produced no error. Here, a transform that
was wrong in a way that produced no *reordering* — and so survived nine months
of measurement.

### `rerank_score` was sigmoided twice

`_apply_post_rerank_adjustments` computed `1 / (1 + e^-rerank_score)`, treating
the reranker's output as a raw cross-encoder logit. It is not one. Every
provider already emits an absolute [0, 1] relevance: sentence-transformers
applies `nn.Sigmoid` for a `num_labels=1` cross-encoder (which
`BAAI/bge-reranker-v2-m3` is), Cohere and ZeroEntropy return `relevance_score`,
the LLM reranker scores 0-1, and HuggingFace normalizes.

Measured over 370 recorded production scores: min 4.7e-05, median 0.064, max
0.999884, **zero negative, zero above 1**. A logit spans roughly [-10, +10] and
goes negative for an irrelevant pair.

The proof needs no measurement, though. In the same `if/else`, `base` was a
cosine in [0, 1] on one branch and a doubly-sigmoided value in [0.5, 0.731] on
the other, and both had the same `superseded_penalty` subtracted and the same
tie band applied.

**The UNADJUSTED order never moved**, which is why it lasted: a sigmoid is
monotonic, so with no penalty and no tie-break the primary sort is exactly the
reranker's own order in either space. That invariance does not extend to the
ADJUSTED order — subtracting a constant and grouping by a fixed band are not
scale-free, and that is precisely what broke. What was distorted:

* **the superseded penalty.** 0.2, documented as a "[0, 1] scale" constant,
  against an axis whose real width was 0.231 — 86% of the reachable range
  instead of 20%. It stopped being a demotion and became an override; no
  in-contract pair could ever clear it.
* **the tie bands.** Calibrated *in* the compressed space, so they were
  self-consistent and no measurement caught them — but the constant no longer
  meant what its name said.

A provider that breaks the contract is now CLAMPED and warned once, not
sigmoided: guessing that an out-of-range value "must be a logit" would re-enter
the bug for anything merely miscalibrated. Clamping is monotonic and the sort is
stable, so ranking order survives; the penalty and bands do not, and the warning
says so.

### `RERANK_TIE_BAND` 0.002 → 0.008 is a unit conversion

Not a re-fit. The band compared `d(sigma(r)) = sigma'(r) * dr` and now compares
`dr`; `sigma'` is 0.2500 at r=0 and 0.2497 at the measured production median
r=0.064, so the factor is 4.00 across effectively the whole pool. The 190x
separation the 2026-07-21 calibration found between near-ties and decisive gaps
is preserved.

Replayed over 407 recorded production candidate pools (3663 adjacent pairs),
reproducing the real leader-anchored grouping: **11 pools differ, from 2 distinct
pairs**, both at the top of the range where `sigma'` is smallest (0.1966), both
`tie -> decisive`. **Zero** went the other way. The window only ever narrows,
which is the conservative direction.

### HuggingFace reranker: min-max → sigmoid (upstream PR #5715)

Ported, not rebased. Min-max produced *set-relative* scores: the lowest-ranked
document was pinned to 0.0, and a single document or an all-tied set collapsed to
0.0 — reporting a relevant result as completely irrelevant.

Computed in the numerically stable form. Upstream's `1 / (1 + np.exp(-arr))`
overflows for large negative logits; numpy returns the right number with a
RuntimeWarning, so it is right for the wrong reason, once per such document.

`normalize=False` still emits raw logits and now warns at construction that it
breaks the contract, rather than letting the downstream clamp look like it
worked.

### Tests

Eleven of thirteen synthetic `rerank_score` values in the suite were ≥ 2.0 —
including the supersedence fixture (2.10 vs 2.00), where the clamp would have
collapsed both to 1.0 and dissolved the "slightly more similar" premise the test
exists to check. All converted; out-of-range values now appear only in the
dedicated clamp test. Fixtures were converted to preserve each test's gap-to-band
RATIO, not its number — converting by `sigma()` would have reproduced the old
`base` exactly while the band moved 4x underneath it.

Seven targeted mutations, one per property (double sigmoid at the call site; at
the helper; clamp removed; warning dedup removed; band reverted; min-max
restored; naive sigmoid), each failing in the test that covers it. Full fork
suite: 128 known signatures, 0 new, 0 changed.

The premise itself is pinned, because nothing in this repo enforces it: a
sentence-transformers upgrade that changed the `num_labels=1` default would
silently push every score out of contract and quietly flatten the penalty and
the bands. `tests/rerankers/test_sentence_transformer_score_space.py` asserts the
default-selection rule against a stub (with the `num_labels>1 -> Identity`
counterfactual, so it is not vacuous) and that `predict` actually APPLIES the
activation rather than merely holding it. The real `bge-reranker-v2-m3` probe
lives behind a new `live_model` marker, deselected by default — it costs ~19s and
needs the HF cache, which would make the normal suite environment-dependent. It
asserts both that scores land in [0, 1] and that they still DISCRIMINATE, since
bounded-but-constant would pass a range check while proving nothing.

`eval/eval_event_date.py` gained criterion **[F]**, a HARD gate on the rerank
path — the path the band change actually touches, and the one where [A] is
declared informational because strengthening it would need the held-out band
calibration this project has deferred. [F] sidesteps that: it measures each twin
pair's real relevance gap at runtime and asserts the mechanism's own
specification, `gap < band -> the dated twin must be promoted` and
`gap >= band -> the reranker's order stands`. Band-value-agnostic by
construction — re-calibrating moves which branch a pair lands in, never whether
the assertion holds — so it gates the mechanism without hand-picking a pair to
sit just inside the band.

Two things make it a gate rather than a formality. The `margin` half compares
against an INDEPENDENTLY captured reference (the same query with
`event_ranking=OFF`), not against the pipeline's own scores: `base` is monotonic
in `rerank_score`, so "the order agrees with the scores" is true by construction
whenever no tie-break fires — it would pass vacuously exactly when there is
nothing to check. And **both branches are required**: with `event_tie_band=0`, a
legal config, every pair lands in `margin` and the tie half is never exercised,
so that case now reports INCONCLUSIVE and fails instead of passing 3/3.

Measured 3/3 with both branches populated (tie=1 at a real gap of 0.002066,
margin=2), and each half falsified separately: inverting the event tie-break key
fails the tie half, an unbounded band fails the margin half, and
`--event-tie-band 0` fails on coverage.

`TestProductionConfigIsInert` covers the DEPLOYED configuration
(`MEM0_DYNAMICS_WEIGHT=0`, `MEM0_DYNAMICS_TIE_BAND=0`), which `eval_temporal.py`
does not: that eval builds `{"dynamics": {"enabled": ...}}` and inherits the fork
defaults, so it never exercised the zeros production actually runs.

## v0.12.0

A minor version, not a patch: `normalize_scope_id` is a new PUBLIC function and
`entity_pipeline_status` gained a field. Everything else here is a fix.

The theme is one failure mode, found three times: a value that is wrong in a way
that produces **no error**. A padded scope matches nothing and the delete reports
success; a pipeline that cannot load reports healthy; a readiness field that was
documented but never existed. None of these crash. All of them are read as
working.

### `entity_pipeline_status`: degraded now means unusable

Two states of an unusable entity pipeline were reported as healthy, so a
readiness probe built on this field answered 200 while extraction ran inert.

**Explicitly configured English with a missing model.** The exemption was
written as `code != DEFAULT_LANGUAGE`. It protects a real case — a deployment
that configured NO language never asked for a pipeline, and failing its
readiness would turn an optional dependency into a mandatory one — but that case
is *fell back to the default*, not *is the default language*. Choosing English on
purpose and lacking `en_core_web_sm` leaves extraction exactly as inert as it
would be in Portuguese. The condition is now `explicito or code !=
DEFAULT_LANGUAGE`, and `explicitly_configured` is published beside it so a reader
can tell which branch applied.

**`load_failed` did not contribute at all.** A model that is installed but will
not load is as inert as one that is absent, and it was the only one of the three
unavailability states that passed readiness.

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
