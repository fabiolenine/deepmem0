"""DeepMem0 v0.8 — T1S, the semantic re-encounter. Pure units, no live infrastructure.

Why this trigger exists: T1 proper requires the LLM to reproduce a fact
byte-identical (MD5 match) — measured on the live corpus (2026-07-27, 1065
memories) it had fired ZERO times ever. T1S reinforces the nearest corpus
near-paraphrase of a NEW fact while still inserting the fact: nothing is
suppressed, nothing is lost.

The integration tests here exercise ``_add_to_vector_store`` for real (fake
self, unbound call) — incidentally the first end-to-end coverage of the T1
hash path too, which until now was only tested at the helper level.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import mem0.memory.main as main_mod
from mem0.configs.base import MemoryConfig, MemoryDynamicsConfig
from mem0.memory.main import (
    _apply_similar_reinforcements,
    _digits_compatible,
    _similar_reinforcement_target,
)
from mem0.utils.dynamics import FIELD_REINFORCED_BY, TRIGGER_SIMILAR

NOW = datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def hours_ago(h):
    return (NOW - timedelta(hours=h)).isoformat()


def _dyn(**kw):
    kw.setdefault("enabled", True)
    return MemoryDynamicsConfig(**kw)


def _hit(hit_id, score, data="", **payload):
    payload.setdefault("data", data)
    return SimpleNamespace(id=hit_id, score=score, payload=payload)


class SearchStore:
    """Vector store fake for the helper: canned search hits, recorded updates."""

    PAYLOAD_UPDATE_MERGES = True

    def __init__(self, hits=None, points=None):
        self.hits = hits or []
        self.points = points or {}
        self.updates = []
        self.search_calls = []

    def search(self, query, vectors, top_k=5, filters=None):
        self.search_calls.append({"top_k": top_k, "filters": filters})
        return self.hits[:top_k]

    def get(self, vector_id):
        if vector_id not in self.points:
            return None
        return SimpleNamespace(id=vector_id, payload=dict(self.points[vector_id]))

    def update(self, vector_id, vector=None, payload=None):
        self.updates.append((vector_id, payload))


class TestDigitGuard:
    """The measured FP class: same template, one identifier swapped.

    All 6 labeled false positives of the threshold probe (41 real boundary
    pairs) differed in digits; zero true positives in the >=0.95 band did.
    """

    def test_swapped_identifier_is_incompatible(self):
        assert not _digits_compatible("Pedido PO-8188 do fornecedor", "Pedido PO-8189 do fornecedor")

    def test_version_strings_are_tokens_not_digit_soup(self):
        # "0.7" vs "0.7.1" must differ as WHOLE tokens — split into bare digits
        # both reduce to {0,7,...} and the swap becomes invisible.
        assert not _digits_compatible("v0.7 deployado em 24/07", "v0.7.1 deployado em 25/07")

    def test_superset_restatement_is_compatible(self):
        # A restatement that ADDS a clause with a new number still counts —
        # subset, not equality (probe pair 34: regra base + gatilho extra).
        assert _digits_compatible("janela de manutenção de 1,5h encerra o deploy",
                                  "janela de manutenção de 1,5h encerra o deploy; se 50% dos health checks falharem, abortar")

    def test_decimal_separator_is_normalized(self):
        assert _digits_compatible("tolerância de 1,5% na latência", "tolerância de 1.5% na latência")

    def test_no_digits_is_compatible(self):
        assert _digits_compatible("hotel não aceita animais", "animais não são aceitos no hotel")

    def test_repeated_token_multiset(self):
        # multiset: "10 e 10" contém dois tokens "10" — um só não basta ao inverso
        assert _digits_compatible("timeout de 10 segundos", "timeout de 10 segundos e retry de 10")
        assert not _digits_compatible("timeout de 10 e 20 segundos", "timeout de 20 e 30 segundos")

    def test_one_sided_digits_is_compatible_by_policy(self):
        # POLÍTICA (pinada pelo /critic-results): o multiset vazio é o extremo
        # do subset — um texto SEM dígitos é compatível com qualquer um. A
        # re-declaração comprimida ("há um limite diário definido") pode reforçar
        # o fato numérico ("limite diário de 1,5 mil"); a ≥0.95 de cosseno isso é
        # quase sempre o mesmo fato. Mudar isto é decisão, não bug.
        assert _digits_compatible("há um limite diário de requisições definido no serviço",
                                  "no serviço o limite diário de requisições é de 1,5 mil")


class TestSimilarTarget:
    def test_fires_at_threshold(self):
        store = SearchStore(hits=[_hit("m1", 0.96, "fato re-declarado")])
        got = _similar_reinforcement_target(store, "fato re-declarado", [0.1], {}, _dyn())
        assert got == ("m1", 0.96)

    def test_below_threshold_stops(self):
        store = SearchStore(hits=[_hit("m1", 0.93, "vizinho apenas relacionado")])
        assert _similar_reinforcement_target(store, "fato", [0.1], {}, _dyn()) is None

    def test_skips_superseded_and_takes_next_eligible(self):
        store = SearchStore(hits=[
            _hit("old", 0.98, "fato", superseded_by="newer"),
            _hit("m2", 0.97, "fato"),
        ])
        got = _similar_reinforcement_target(store, "fato", [0.1], {}, _dyn())
        assert got == ("m2", 0.97)

    def test_skips_digit_mismatch(self):
        store = SearchStore(hits=[_hit("m1", 0.9754, "Pedido PO-8188 do fornecedor")])
        assert _similar_reinforcement_target(
            store, "Pedido PO-8189 do fornecedor", [0.1], {}, _dyn()) is None

    def test_flag_off_and_threshold_zero_disable(self):
        store = SearchStore(hits=[_hit("m1", 0.99, "fato")])
        assert _similar_reinforcement_target(
            store, "fato", [0.1], {}, _dyn(reinforce_on_similar=False)) is None
        assert _similar_reinforcement_target(
            store, "fato", [0.1], {}, _dyn(reinforce_similarity_threshold=0)) is None
        assert store.search_calls == [], "disabled must not even search"

    def test_search_failure_is_fail_open(self):
        class Boom(SearchStore):
            def search(self, *a, **kw):
                raise RuntimeError("qdrant down")

        seen = []
        main_mod.reinforcement_observer = lambda *a: seen.append((a[0], a[1], a[2]))
        try:
            got = _similar_reinforcement_target(Boom(), "fato", [0.1], {}, _dyn())
        finally:
            main_mod.reinforcement_observer = None
        assert got is None
        assert seen and seen[0][1] == TRIGGER_SIMILAR and seen[0][2] == "failed"

    def test_scope_filters_are_forwarded(self):
        store = SearchStore(hits=[])
        _similar_reinforcement_target(store, "fato", [0.1], {"user_id": "u"}, _dyn())
        assert store.search_calls == [{"top_k": 3, "filters": {"user_id": "u"}}]


class TestApplyDeferred:
    def test_fresh_read_applies_and_notifies(self):
        store = SearchStore(points={"m1": {"data": "fato", "created_at": hours_ago(72)}})
        seen = []
        main_mod.reinforcement_observer = lambda *a: seen.append((a[0], a[1], a[2], a[4]))
        try:
            _apply_similar_reinforcements(store, _dyn(), [("m1", 0.97, "new-id")])
        finally:
            main_mod.reinforcement_observer = None
        assert len(store.updates) == 1
        assert store.updates[0][1][FIELD_REINFORCED_BY][-1] == TRIGGER_SIMILAR
        (mem_id, trigger, outcome, ctx), = seen
        assert (mem_id, trigger, outcome) == ("m1", TRIGGER_SIMILAR, "applied")
        assert ctx == {"similarity": 0.97, "from_add": "new-id"}, "ids e score, nunca texto"

    def test_fresh_read_sees_superseded_and_skips(self):
        # o alvo foi supersedido DEPOIS da decisão (por outro fato do batch, ou
        # por qualquer escrita concorrente): a leitura fresca é o guard
        store = SearchStore(points={"m1": {"data": "fato", "superseded_by": "other"}})
        seen = []
        main_mod.reinforcement_observer = lambda *a: seen.append(a[2])
        try:
            _apply_similar_reinforcements(store, _dyn(), [("m1", 0.97, "new-id")])
        finally:
            main_mod.reinforcement_observer = None
        assert store.updates == []
        assert seen == ["suppressed"]

    def test_missing_target_notifies_missing(self):
        seen = []
        main_mod.reinforcement_observer = lambda *a: seen.append(a[2])
        try:
            _apply_similar_reinforcements(SearchStore(), _dyn(), [("gone", 0.97, "n")])
        finally:
            main_mod.reinforcement_observer = None
        assert seen == ["missing"]

    def test_global_window_suppresses_cross_trigger(self):
        # t3 aplicado há minutos → a janela GLOBAL de 1h silencia o t1s também
        store = SearchStore(points={"m1": {
            "data": "fato", "reinforced_at": [hours_ago(0.1)], "access_count": 2,
        }})
        seen = []
        main_mod.reinforcement_observer = lambda *a: seen.append(a[2])
        try:
            _apply_similar_reinforcements(store, _dyn(), [("m1", 0.97, "n")])
        finally:
            main_mod.reinforcement_observer = None
        assert store.updates == []
        assert seen == ["suppressed"]


# ---------------------------------------------------------------------------
# Integration: the REAL _add_to_vector_store, fake self, unbound call
# ---------------------------------------------------------------------------

class PipelineStore:
    """Serves phase-1 (top_k=10) and the per-fact T1S search (top_k=3)."""

    PAYLOAD_UPDATE_MERGES = True

    def __init__(self, corpus=None, fail_insert=False):
        self.corpus = corpus or []
        self.fail_insert = fail_insert
        self.inserted = []
        self.updates = []

    def search(self, query, vectors, top_k=5, filters=None):
        return self.corpus[:top_k]

    def get(self, vector_id):
        for pt in self.corpus:
            if str(pt.id) == str(vector_id):
                return SimpleNamespace(id=pt.id, payload=dict(pt.payload))
        return None

    def insert(self, vectors, ids, payloads):
        if self.fail_insert:
            raise RuntimeError("insert down")
        self.inserted.extend(zip(ids, payloads))

    def update(self, vector_id, vector=None, payload=None):
        self.updates.append((vector_id, payload))


def _fake_self(store, facts, dynamics=None):
    """Minimal Memory-shaped object for the V3 pipeline (infer=True)."""
    cfg = MemoryConfig()
    if dynamics is not None:
        cfg.dynamics = dynamics
    llm = SimpleNamespace(generate_response=lambda **kw: json.dumps({"memory": facts}))
    embedder = SimpleNamespace(
        embed=lambda text, action: [0.1, 0.2],
        embed_batch=lambda texts, action: [[0.1, 0.2] for _ in texts],
    )
    db = SimpleNamespace(
        get_last_messages=lambda scope, limit=10: [],
        save_messages=lambda *a, **kw: None,
        batch_add_history=lambda *a, **kw: None,
        add_history=lambda *a, **kw: None,
    )
    return SimpleNamespace(
        config=cfg, vector_store=store, llm=llm, embedding_model=embedder,
        db=db, custom_instructions=None, api_version="v1.1",
    )


def _run_add(store, facts, dynamics=None, use_async=False):
    fake = _fake_self(store, facts, dynamics)
    orig_entities = main_mod.extract_entities_batch
    main_mod.extract_entities_batch = lambda texts: [[] for _ in texts]
    try:
        if use_async:
            return asyncio.run(main_mod.AsyncMemory._add_to_vector_store(
                fake, [{"role": "user", "content": "msg"}], {}, {"user_id": "u"}, True))
        return main_mod.Memory._add_to_vector_store(
            fake, [{"role": "user", "content": "msg"}], {}, {"user_id": "u"}, True)
    finally:
        main_mod.extract_entities_batch = orig_entities


def _events(store, facts, dynamics=None, use_async=False):
    seen = []
    main_mod.reinforcement_observer = lambda *a: seen.append((a[0], a[1], a[2]))
    try:
        _run_add(store, facts, dynamics, use_async=use_async)
    finally:
        main_mod.reinforcement_observer = None
    return seen


class TestAddPathIntegration:
    def test_near_dup_reinforces_neighbor_and_still_inserts(self):
        store = PipelineStore(corpus=[
            _hit("m1", 0.97, "o coletor usa compressão gzip com rotação diária",
                 created_at=hours_ago(72), hash="x"),
        ])
        seen = _events(store, [{"text": "coletor com compressão gzip e rotação diária"}])
        assert [s for s in seen if s[1] == TRIGGER_SIMILAR and s[2] == "applied"], seen
        assert len(store.inserted) == 1, "o fato novo TEM que ser inserido (nada é suprimido)"
        assert store.updates and store.updates[0][0] == "m1"

    def test_exact_hash_takes_precedence_no_insert_no_t1s(self):
        text = "fato byte a byte idêntico"
        store = PipelineStore(corpus=[
            _hit("m1", 0.99, text, created_at=hours_ago(72),
                 hash=hashlib.md5(text.encode()).hexdigest()),
        ])
        seen = _events(store, [{"text": text}])
        assert [s[1] for s in seen] == ["t1"], "hash exato = t1, nunca t1s"
        assert store.inserted == [], "dedup exato mantém a supressão HERDADA de inserção"

    def test_exact_hash_on_superseded_dedupes_but_never_reinforces(self):
        """v0.9: a timeline do supersedido vive no sucessor — reforçá-lo
        recriaria o double-dip que a máscara removeu. Dedup fica; t1 não."""
        text = "fato byte a byte idêntico"
        store = PipelineStore(corpus=[
            _hit("m1", 0.99, text, created_at=hours_ago(72),
                 hash=hashlib.md5(text.encode()).hexdigest(),
                 superseded_by="m2"),
        ])
        seen = _events(store, [{"text": text}])
        assert seen == [], f"supersedido não é reforçado — veio {seen}"
        assert store.inserted == [], "a supressão de inserção do dedup exato fica"
        assert store.updates == []

    def test_supersedes_intent_skips_semantic_reinforcement(self):
        # correção é quase-paráfrase com um valor trocado — reforçar o corrigido
        # é o pior caso; QUALQUER marca supersedes desliga o t1s para o fato
        store = PipelineStore(corpus=[
            _hit("m1", 0.98, "a porta do serviço é 8080", created_at=hours_ago(72), hash="x"),
        ])
        seen = _events(store, [{"text": "a porta do serviço é 8080 agora", "supersedes": ["0"]}])
        assert not [s for s in seen if s[1] == TRIGGER_SIMILAR], seen
        assert len(store.inserted) == 1

    def test_two_facts_one_target_reinforces_once(self):
        store = PipelineStore(corpus=[
            _hit("m1", 0.97, "o limite da fila é de duzentos itens",
                 created_at=hours_ago(72), hash="x"),
        ])
        seen = _events(store, [
            {"text": "limite da fila: duzentos itens"},
            {"text": "o tamanho máximo da fila é de duzentos itens"},
        ])
        applied = [s for s in seen if s[1] == TRIGGER_SIMILAR]
        assert len(applied) == 1, f"um reforço por alvo por add — veio {applied}"
        assert len(store.inserted) == 2

    def test_failed_persist_reinforces_nobody(self):
        store = PipelineStore(corpus=[
            _hit("m1", 0.97, "fato re-declarado", created_at=hours_ago(72), hash="x"),
        ], fail_insert=True)
        seen = _events(store, [{"text": "fato re-declarado de novo"}])
        assert not [s for s in seen if s[1] == TRIGGER_SIMILAR], seen
        assert store.updates == []

    def test_async_twin_mirrors(self):
        store = PipelineStore(corpus=[
            _hit("m1", 0.97, "o coletor usa compressão gzip com rotação diária",
                 created_at=hours_ago(72), hash="x"),
        ])
        seen = _events(store, [{"text": "coletor com compressão gzip e rotação diária"}],
                       use_async=True)
        assert [s for s in seen if s[1] == TRIGGER_SIMILAR and s[2] == "applied"], seen
        assert len(store.inserted) == 1

    def test_dynamics_disabled_never_searches_per_fact(self):
        calls = []

        class CountingStore(PipelineStore):
            def search(self, query, vectors, top_k=5, filters=None):
                calls.append(top_k)
                return super().search(query, vectors, top_k, filters)

        store = CountingStore(corpus=[])
        cfg_dyn = MemoryDynamicsConfig(enabled=False)
        _run_add(store, [{"text": "um fato qualquer"}], dynamics=cfg_dyn)
        assert calls == [10], f"só a busca de fase 1 — veio {calls}"
        assert len(store.inserted) == 1
