"""DeepMem0 v0.2 human-memory dynamics tests — pure units, no live infrastructure."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from mem0.configs.base import MemoryConfig, MemoryDynamicsConfig
from mem0.memory.main import (
    _apply_activation_post_rerank,
    _dynamics_config,
    _reinforce_memory,
    plan_reinforcement,
)
from mem0.utils.dynamics import (
    FIELD_REINFORCE_COUNTS,
    DYNAMICS_FIELDS,
    FIELD_LAST_SEARCH_REINFORCED_AT,
    FIELD_REINFORCED_BY,
    TRIGGER_SEARCH,
    activation_boost,
    base_level_activation,
    boost_from_payload,
    reinforcement_fields,
    should_reinforce,
)
from mem0.utils.scoring import score_and_rank

NOW = datetime(2030, 6, 15, 12, 0, 0, tzinfo=timezone.utc)


def hours_ago(h):
    return (NOW - timedelta(hours=h)).isoformat()


class TestBaseLevelActivation:
    def test_no_history_is_neutral(self):
        assert base_level_activation(None, now=NOW) is None
        assert base_level_activation([], now=NOW) is None
        assert activation_boost(None) == 0.0

    def test_recency_raises_activation(self):
        recent = base_level_activation([hours_ago(1)], now=NOW)
        stale = base_level_activation([hours_ago(720)], now=NOW)
        assert recent > stale

    def test_frequency_raises_activation(self):
        once = base_level_activation([hours_ago(24)], now=NOW)
        thrice = base_level_activation([hours_ago(72), hours_ago(48), hours_ago(24)], now=NOW)
        assert thrice > once

    def test_decay_is_implicit_with_passing_time(self):
        history = [hours_ago(2)]
        earlier = base_level_activation(history, now=NOW)
        later = base_level_activation(history, now=NOW + timedelta(days=30))
        assert later < earlier

    def test_petrov_tail_counts_trimmed_reinforcements(self):
        history = [hours_ago(48), hours_ago(24)]
        exact = base_level_activation(history, access_count=2, now=NOW)
        with_tail = base_level_activation(
            history, access_count=50, now=NOW, first_seen=hours_ago(24 * 365)
        )
        assert with_tail > exact

    def test_future_or_immediate_timestamps_are_clamped(self):
        activation = base_level_activation([NOW.isoformat(), (NOW + timedelta(hours=1)).isoformat()], now=NOW)
        assert activation is not None
        assert activation < 10  # finite, clamped — not an unbounded spike

    def test_malformed_timestamps_are_ignored(self):
        assert base_level_activation(["not-a-date", 42], now=NOW) is None
        ok = base_level_activation(["not-a-date", hours_ago(5)], now=NOW)
        assert ok is not None

    def test_boost_is_bounded(self):
        for h in (0.01, 1, 24, 24 * 365):
            b = activation_boost(base_level_activation([hours_ago(h)], now=NOW))
            assert 0.0 < b < 1.0

    def test_boost_from_payload_reads_dynamics_fields(self):
        payload = {
            "created_at": hours_ago(1000),
            "reinforced_at": [hours_ago(48), hours_ago(2)],
            "access_count": 7,
        }
        assert boost_from_payload(payload, now=NOW) > 0.0
        assert boost_from_payload({"created_at": hours_ago(2)}, now=NOW) == 0.0


class TestReinforcementWindow:
    def test_no_history_reinforces(self):
        assert should_reinforce({}, now=NOW) is True

    def test_inside_window_is_suppressed(self):
        payload = {"reinforced_at": [hours_ago(0.5)]}
        assert should_reinforce(payload, now=NOW, window_seconds=3600) is False

    def test_outside_window_reinforces(self):
        payload = {"reinforced_at": [hours_ago(1.5)]}
        assert should_reinforce(payload, now=NOW, window_seconds=3600) is True

    def test_zero_window_disables_suppression(self):
        payload = {"reinforced_at": [NOW.isoformat()]}
        assert should_reinforce(payload, now=NOW, window_seconds=0) is True


class TestReinforcementFields:
    def test_legacy_memory_adopts_created_at(self):
        payload = {"created_at": hours_ago(240), "data": "hermes_fx uses walk-forward validation"}
        fields = reinforcement_fields(payload, now=NOW)
        assert fields["reinforced_at"] == [hours_ago(240), NOW.isoformat()]
        assert fields["access_count"] == 2
        assert fields["last_accessed"] == NOW.isoformat()

    def test_history_is_bounded_but_count_is_not(self):
        payload = {
            "reinforced_at": [hours_ago(h) for h in range(20, 10, -1)],
            "access_count": 40,
        }
        fields = reinforcement_fields(payload, now=NOW, max_timestamps=10)
        assert len(fields["reinforced_at"]) == 10
        assert fields["reinforced_at"][-1] == NOW.isoformat()
        assert fields["access_count"] == 41

    def test_creation_is_neutral_until_first_reinforcement(self):
        # Option B: a freshly created memory (no dynamics fields) is neutral,
        # exactly like the legacy corpus — no new-vs-old bias.
        fresh = {"data": "boreal_app ships weekly", "created_at": hours_ago(0)}
        assert boost_from_payload(fresh, now=NOW) == 0.0
        # First reinforcement adopts created_at, yielding a two-event history.
        fields = reinforcement_fields(fresh, now=NOW)
        assert fields["reinforced_at"] == [hours_ago(0), NOW.isoformat()]
        assert fields["access_count"] == 2
        assert boost_from_payload({**fresh, **fields}, now=NOW) > 0.0


class TestActivationInFusion:
    CANDIDATES = [
        {"id": "aaa", "score": 0.80, "payload": {"data": "fact A"}},
        {"id": "bbb", "score": 0.80, "payload": {"data": "fact B"}},
    ]

    def test_activation_breaks_ties(self):
        ranked = score_and_rank(
            semantic_results=self.CANDIDATES,
            bm25_scores={},
            entity_boosts={},
            threshold=0.1,
            top_k=2,
            activation_boosts={"bbb": 0.9},
            activation_weight=0.15,
        )
        assert ranked[0]["id"] == "bbb"

    def test_no_boosts_is_backward_compatible(self):
        legacy = score_and_rank(self.CANDIDATES, {}, {}, 0.1, 2)
        explicit = score_and_rank(
            self.CANDIDATES, {}, {}, 0.1, 2, activation_boosts={}, activation_weight=0.15
        )
        assert [r["score"] for r in legacy] == [r["score"] for r in explicit]

    def test_explain_exposes_activation(self):
        ranked = score_and_rank(
            self.CANDIDATES, {}, {}, 0.1, 2, explain=True,
            activation_boosts={"bbb": 1.0}, activation_weight=0.2,
        )
        by_id = {r["id"]: r["score_details"] for r in ranked}
        assert by_id["bbb"]["activation_boost"] == 0.2
        assert by_id["aaa"]["activation_boost"] == 0.0


class TestActivationPostRerank:
    def make_docs(self):
        # _apply_activation_post_rerank reads the real clock, so these
        # timestamps must be relative to the actual now (not the fixed NOW).
        def real_hours_ago(h):
            return (datetime.now(timezone.utc) - timedelta(hours=h)).isoformat()

        # ⚠️ REALISTIC rerank scores. The bge-reranker-v2-m3 scores most of this
        # corpus near ZERO (measured 2026-07-21: golden 0.01–0.25; the full
        # production distribution is min 4.7e-05 / median 0.064 / max 0.9999).
        # An earlier fixture used 2.0–8.0, a region the reranker can NEVER reach
        # — rerank_score is an absolute relevance in [0, 1], not a logit — so its
        # "decisive gap" test validated a fantasy and the real overturning bug
        # went uncaught. cold≈hot here is a TRUE near-tie.
        #
        # HISTORY: the 2026-07-21 note called these values "logits" and blamed
        # the compression on the reranker's operating point. They were already
        # sigmoid outputs; the compression came from main.py sigmoiding them a
        # SECOND time. Fixed 2026-07-31 — the instrument, not the system.
        return [
            {
                "id": "cold",
                "memory": "atlas_ingest retries three times",
                "created_at": real_hours_ago(500),
                "rerank_score": 0.020,
                "metadata": {},
            },
            {
                "id": "hot",
                "memory": "atlas_ingest retries thrice",
                "created_at": real_hours_ago(500),
                "rerank_score": 0.019,  # gap 0.001 < tie_band 0.008
                "metadata": {
                    "reinforced_at": [real_hours_ago(30), real_hours_ago(4)],
                    "access_count": 6,
                },
            },
        ]

    def test_reinforced_memory_wins_near_tie(self):
        # genuine tie (gap 0.001 < tie_band 0.008): activation decides → hot wins
        dyn = MemoryDynamicsConfig()
        ordered = _apply_activation_post_rerank(self.make_docs(), dyn)
        assert ordered[0]["id"] == "hot"
        assert ordered[0]["activation"] > 0

    def test_decisive_rerank_gap_is_not_overturned(self):
        # REAL operating point (regression for the 2026-07-21 overturn bug): the
        # reranker prefers cold by a decisive 0.25 relevance margin (>> tie_band
        # 0.008). Reinforcement must NOT flip it. The additive form
        # (base + 0.15*activation) DID flip exactly this, on the live golden.
        docs = self.make_docs()
        docs[0]["rerank_score"] = 0.27  # vs hot 0.019: gap 0.251 >> 0.008
        dyn = MemoryDynamicsConfig()
        ordered = _apply_activation_post_rerank(docs, dyn)
        assert ordered[0]["id"] == "cold"

    def test_zero_weight_still_breaks_ties(self):
        """weight gates the FUSION term only — the post-rerank tie-break is its
        own knob (tie_band), mirroring how v0.6 gates the event tie-break.

        This reverses an earlier expectation, deliberately: the two were coupled,
        which made "tie-break only" unreachable. Zeroing the fusion weight is the
        way to keep exposure bias out of pool composition, and it must not also
        silence the bounded tie-break — the one form of activation the 2026-07-21
        ablation vindicated.
        """
        dyn = MemoryDynamicsConfig(weight=0.0)
        ordered = _apply_activation_post_rerank(self.make_docs(), dyn)
        assert ordered[0]["id"] == "hot"

    def test_fully_inert_when_both_knobs_are_zero(self):
        """The shadow-collection mode: reinforcement is recorded, ranking untouched."""
        dyn = MemoryDynamicsConfig(weight=0.0, tie_band=0.0)
        ordered = _apply_activation_post_rerank(self.make_docs(), dyn)
        assert [d["id"] for d in ordered] == ["cold", "hot"]

    def test_tie_band_zero_disables_post_rerank_reorder(self):
        # tie_band=0 → activation cannot reorder post-rerank; reranker wins the
        # near-tie by its (tiny) margin even against a reinforced candidate.
        dyn = MemoryDynamicsConfig(tie_band=0.0)
        ordered = _apply_activation_post_rerank(self.make_docs(), dyn)
        assert ordered[0]["id"] == "cold"


class FakeVectorStore:
    def __init__(self):
        self.updates = []

    def update(self, vector_id, vector=None, payload=None):
        self.updates.append((vector_id, payload))


class TestReinforceMemory:
    """The outcome is a structured string, not a bool: the old boolean collapsed
    "the window suppressed it" with "the store blew up", which made the two
    indistinguishable in telemetry — the exact blindness that let reinforcement
    sit inert in production for three weeks unnoticed."""

    def test_writes_full_merged_payload_when_store_replaces(self):
        store = FakeVectorStore()  # no PAYLOAD_UPDATE_MERGES → must get everything
        dyn = MemoryDynamicsConfig()
        payload = {"data": "orion pipeline", "created_at": hours_ago(72), "domain": "infra"}
        assert _reinforce_memory(store, dyn, "mem-1", payload) == "applied"
        vector_id, written = store.updates[0]
        assert vector_id == "mem-1"
        assert written["data"] == "orion pipeline"  # non-dynamics keys preserved
        assert written["domain"] == "infra"
        assert written["access_count"] == 2

    def test_writes_only_dynamics_fields_when_store_merges(self):
        """Read-modify-write reverts whatever a concurrent writer changed between
        the read and the write. On a merging store, send only what changed."""
        class MergingStore(FakeVectorStore):
            PAYLOAD_UPDATE_MERGES = True

        store = MergingStore()
        dyn = MemoryDynamicsConfig()
        payload = {"data": "orion pipeline", "created_at": hours_ago(72), "domain": "infra"}
        assert _reinforce_memory(store, dyn, "mem-1", payload) == "applied"
        _vector_id, written = store.updates[0]
        assert set(written) <= set(DYNAMICS_FIELDS)
        assert "data" not in written and "domain" not in written
        assert written["access_count"] == 2

    def test_window_suppresses_write(self):
        store = FakeVectorStore()
        dyn = MemoryDynamicsConfig()
        payload = {"reinforced_at": [datetime.now(timezone.utc).isoformat()]}
        assert _reinforce_memory(store, dyn, "mem-1", payload) == "suppressed"
        assert store.updates == []

    def test_store_failure_never_raises(self):
        class ExplodingStore:
            def update(self, **kwargs):
                raise RuntimeError("boom")

        dyn = MemoryDynamicsConfig()
        assert _reinforce_memory(ExplodingStore(), dyn, "mem-1", {"data": "x"}) == "failed"

    def test_every_trigger_reaches_the_observer(self):
        """T2 used to write the timeline inline, so instrumenting the shared
        helper missed it entirely. All three triggers must be observable."""
        import mem0.memory.main as main_mod

        seen = []
        store = FakeVectorStore()
        dyn = MemoryDynamicsConfig()
        main_mod.reinforcement_observer = lambda mid, trig, out, ms, ctx=None: seen.append(
            (mid, trig, out))
        try:
            _reinforce_memory(store, dyn, "mem-1", {"data": "x"}, trigger="t1")
            _reinforce_memory(store, dyn, "mem-2", {"data": "y"}, trigger="t3")
            fields, outcome = main_mod.plan_reinforcement({"data": "z"}, dyn, "t2")
            main_mod._notify_reinforcement("mem-3", "t2", outcome)
        finally:
            main_mod.reinforcement_observer = None
        assert [t for _, t, _ in seen] == ["t1", "t3", "t2"]
        assert {o for _, _, o in seen} == {"applied"}
        assert fields is not None

    def test_t2_end_to_end_notifies_after_the_write(self):
        """COMPORTAMENTAL: exercita `_update_memory` de verdade.

        O teste anterior chamava `plan_reinforcement` e `_notify_reinforcement`
        à mão — ficaria VERDE mesmo se o update parasse de notificar T2.
        """
        import mem0.memory.main as main_mod

        order = []

        class Store:
            def __init__(self):
                self.payload = {"data": "antigo", "created_at": hours_ago(72),
                                "user_id": "u"}

            def get(self, vector_id):
                order.append("read")
                return SimpleNamespace(id=vector_id, payload=dict(self.payload))

            def update(self, vector_id, vector=None, payload=None):
                order.append("write")
                self.payload = payload

        class Embedder:
            def embed(self, data, action):
                order.append("embed")
                return [0.1, 0.2]

        seen = []
        store = Store()
        # version_on_update DESLIGADO: é o modo em que o T2 existe (ver o teste
        # seguinte, que prova o contrário no modo default).
        cfg = MemoryConfig()
        cfg.temporality.version_on_update = False
        fake = SimpleNamespace(
            config=cfg, vector_store=store, embedding_model=Embedder(),
            db=SimpleNamespace(add_history=lambda *a, **kw: order.append("history")),
            _remove_memory_from_entity_store=lambda *a, **kw: None,
            _add_memory_to_entity_store=lambda *a, **kw: None,
            _link_entities_for_memory=lambda *a, **kw: None,
        )
        main_mod.reinforcement_observer = lambda *a: seen.append((a[1], a[2], list(order)))
        try:
            main_mod.Memory._update_memory(fake, "mem-1", "novo texto", {})
        finally:
            main_mod.reinforcement_observer = None

        assert [t for t, _o, _s in seen] == ["t2"], "o T2 tem que notificar de verdade"
        _trigger, outcome, order_at_notify = seen[0]
        assert outcome == "applied"
        # embed ANTES da leitura (encurta a janela em que um T3 é apagado) e
        # notify DEPOIS da escrita (não afirmar reforço que não persistiu)
        assert order.index("embed") < order.index("read")
        assert "write" in order_at_notify
        assert store.payload["reinforced_by"][-1] == "t2"
        assert store.payload["reinforce_counts"] == {"t2": 1}

    def test_versioned_update_routes_and_delegates_t2(self):
        """COMPORTAMENTAL: com version_on_update ligado (DEFAULT do fork) o
        update roteia para o caminho versionado. HISTÓRICO: até a v0.9 este
        teste afirmava que o T2 NÃO existia nesse modo (a versão nascia neutra
        e o gatilho sumia em silêncio); a v0.9 inverteu a decisão — o T2 vive
        DENTRO de _version_update via _plan_version_dynamics, coberto pelos
        testes de TestVersionDynamicsPlanner. Aqui fica só o contrato de
        roteamento: nenhum T2 é emitido FORA do caminho versionado (o roteador
        não pode duplicar o evento)."""
        import mem0.memory.main as main_mod

        routed, seen = [], []
        fake = SimpleNamespace(
            config=MemoryConfig(),  # default = version_on_update ligado
            _version_update=lambda *a, **k: routed.append(a[0]) or ("novo", "velho"),
        )
        main_mod.reinforcement_observer = lambda *a: seen.append(a[1])
        try:
            main_mod.Memory._update_memory(fake, "mem-1", "novo texto", {})
        finally:
            main_mod.reinforcement_observer = None

        assert routed == ["mem-1"], "deveria rotear para o caminho versionado"
        assert seen == [], "o ROTEADOR não emite T2 — o evento pertence a _version_update"

    def test_observer_failure_never_breaks_bookkeeping(self):
        import mem0.memory.main as main_mod

        store = FakeVectorStore()
        main_mod.reinforcement_observer = lambda *a: 1 / 0
        try:
            assert _reinforce_memory(
                store, MemoryDynamicsConfig(), "mem-1", {"data": "x"}
            ) == "applied"
        finally:
            main_mod.reinforcement_observer = None
        assert len(store.updates) == 1


class TestTriggerProvenance:
    """Without provenance a timeline is unattributable: a search exposure cannot
    be told from an explicit write, so neither a trigger-specific window nor a
    selective rollback of exposure events is possible."""

    def test_origins_stay_aligned_with_timestamps(self):
        payload = {"created_at": hours_ago(72)}
        fields = reinforcement_fields(payload, trigger="t3")
        assert fields[FIELD_REINFORCED_BY] == ["created", "t3"]
        assert len(fields[FIELD_REINFORCED_BY]) == len(fields["reinforced_at"])

    def test_legacy_timeline_is_padded_not_guessed(self):
        """A history written before provenance existed must not be back-labelled
        with an invented trigger."""
        payload = {"reinforced_at": [hours_ago(72), hours_ago(48)], "access_count": 2}
        fields = reinforcement_fields(payload, trigger="t1")
        assert fields[FIELD_REINFORCED_BY] == [None, None, "t1"]

    def test_trim_keeps_both_lists_aligned(self):
        payload = {
            "reinforced_at": [hours_ago(100 - i) for i in range(10)],
            FIELD_REINFORCED_BY: ["t1"] * 10,
            "access_count": 10,
        }
        fields = reinforcement_fields(payload, max_timestamps=10, trigger="t3")
        assert len(fields["reinforced_at"]) == 10
        assert len(fields[FIELD_REINFORCED_BY]) == 10
        assert fields[FIELD_REINFORCED_BY][-1] == "t3"

    def test_search_trigger_stamps_its_own_clock(self):
        fields = reinforcement_fields({"created_at": hours_ago(72)}, trigger=TRIGGER_SEARCH)
        assert FIELD_LAST_SEARCH_REINFORCED_AT in fields
        assert FIELD_LAST_SEARCH_REINFORCED_AT not in reinforcement_fields(
            {"created_at": hours_ago(72)}, trigger="t1"
        )


class TestExposureWindow:
    """T3 is exposure, not confirmed use: it gets its own budget so a write and a
    mere retrieval never spend each other's."""

    def test_search_window_blocks_only_search(self):
        payload = {
            "reinforced_at": [hours_ago(5)],
            FIELD_LAST_SEARCH_REINFORCED_AT: hours_ago(5),
        }
        # 1h global window already elapsed; the 24h exposure window has not
        assert should_reinforce(payload, now=NOW, window_seconds=3600, trigger=TRIGGER_SEARCH,
                                search_window_seconds=86400) is False
        assert should_reinforce(payload, now=NOW, window_seconds=3600, trigger="t1",
                                search_window_seconds=86400) is True

    def test_explicit_write_does_not_consume_the_exposure_budget(self):
        payload = {"reinforced_at": [hours_ago(5)]}  # a T1/T2 event, no T3 clock
        assert should_reinforce(payload, now=NOW, window_seconds=3600, trigger=TRIGGER_SEARCH,
                                search_window_seconds=86400) is True

    def test_exposure_window_expires(self):
        payload = {
            "reinforced_at": [hours_ago(30)],
            FIELD_LAST_SEARCH_REINFORCED_AT: hours_ago(30),
        }
        assert should_reinforce(payload, now=NOW, window_seconds=3600, trigger=TRIGGER_SEARCH,
                                search_window_seconds=86400) is True

    def test_global_window_still_applies_to_search(self):
        payload = {"reinforced_at": [hours_ago(0)]}
        assert should_reinforce(payload, now=NOW, window_seconds=3600, trigger=TRIGGER_SEARCH,
                                search_window_seconds=86400) is False


class TestSearchReinforcementGate:
    """The per-call opt-out exists to protect measurement: a golden set running
    its own queries against the live corpus would otherwise reinforce its own
    expected targets on every run."""

    def _dyn(self, **kw):
        return MemoryDynamicsConfig(reinforce_on_search=True, **kw)

    def test_per_call_false_opts_out_even_with_t3_on(self):
        from mem0.memory.main import _t3_enabled

        assert _t3_enabled(self._dyn(), None) is True
        assert _t3_enabled(self._dyn(), True) is True
        assert _t3_enabled(self._dyn(), False) is False

    def test_per_call_true_cannot_force_t3_when_globally_off(self):
        from mem0.memory.main import _t3_enabled

        assert _t3_enabled(MemoryDynamicsConfig(reinforce_on_search=False), True) is False

    def test_none_config_never_reinforces(self):
        from mem0.memory.main import _t3_enabled

        assert _t3_enabled(None, True) is False

    def _targets(self, dyn, docs):
        from mem0.memory.main import _t3_targets

        return _t3_targets(dyn, docs, search_id="sid", exposed_at=NOW)

    def test_top_n_limits_what_a_search_reinforces(self):
        docs = [{"id": f"m{i}"} for i in range(10)]
        got = self._targets(self._dyn(reinforce_top_n=3), docs)
        assert [t.memory_id for t in got] == ["m0", "m1", "m2"]
        assert len(self._targets(self._dyn(reinforce_top_n=0), docs)) == 10

    def test_targets_skip_docs_without_id(self):
        docs = [{"id": "m0"}, {"score": 1}, {"id": "m1"}]
        assert [t.memory_id for t in self._targets(self._dyn(), docs)] == ["m0", "m1"]

    def test_target_carries_rank_search_id_and_exposure_time(self):
        """Correlação e instante de exposição viajam POR VALOR: o reforço roda
        noutra thread e, no async, tasks concorrentes dividem a thread do loop —
        contexto implícito vazaria ou sumiria."""
        docs = [{"id": "a", "metadata": {"domain": "ai"}}, {"id": "b"}]
        got = self._targets(self._dyn(), docs)
        assert [t.rank for t in got] == [1, 2]  # 1-based, posição na página final
        assert {t.search_id for t in got} == {"sid"}
        assert all(t.exposed_at == NOW for t in got)
        assert got[0].snapshot == {"domain": "ai"}
        assert got[1].snapshot == {}


class TestReinforceCounts:
    """A tally por gatilho existe porque `reinforced_by` trunca em K=10: sem ela,
    passando de dez eventos o breakdown some em silêncio."""

    def test_counts_only_real_events_not_the_created_seed(self):
        fields = reinforcement_fields({"created_at": hours_ago(72)}, trigger="t3")
        assert fields[FIELD_REINFORCE_COUNTS] == {"t3": 1}
        assert fields["access_count"] == 2  # a semente conta no access_count...
        # ...mas NÃO na tally: o invariante é sum(counts) == access_count - semente
        assert sum(fields[FIELD_REINFORCE_COUNTS].values()) == fields["access_count"] - 1

    def test_legacy_timeline_migrates_to_unknown_never_guessed(self):
        payload = {
            "created_at": hours_ago(100),
            "reinforced_at": [hours_ago(100), hours_ago(50), hours_ago(20)],
            "access_count": 3,
        }
        fields = reinforcement_fields(payload, trigger="t1")
        assert fields[FIELD_REINFORCE_COUNTS] == {"unknown": 2, "t1": 1}
        assert sum(fields[FIELD_REINFORCE_COUNTS].values()) == fields["access_count"] - 1

    def test_tally_survives_the_trim(self):
        payload = {
            "created_at": hours_ago(200),
            "reinforced_at": [hours_ago(100 - i) for i in range(10)],
            FIELD_REINFORCED_BY: ["t3"] * 10,
            FIELD_REINFORCE_COUNTS: {"t3": 40},
            "access_count": 41,
        }
        fields = reinforcement_fields(payload, max_timestamps=10, trigger="t3")
        assert len(fields["reinforced_at"]) == 10          # timestamps truncados
        assert fields[FIELD_REINFORCE_COUNTS] == {"t3": 41}  # contagem inteira

    def test_malformed_tally_is_dropped_not_propagated(self):
        payload = {"created_at": hours_ago(72), FIELD_REINFORCE_COUNTS: {"t3": "muitos"}}
        fields = reinforcement_fields(payload, trigger="t2")
        assert fields[FIELD_REINFORCE_COUNTS] == {"t2": 1}

    def test_tally_inheritance_is_a_decision_not_a_blacklist_accident(self):
        """v0.9: o tally sai da blacklist cega (herdar virou decisão explícita),
        mas o CALLER continua proibido de escrevê-lo — a proveniência só pode
        vir do head."""
        from mem0.memory.main import _VERSION_CALLER_ONLY_BLOCKED, _VERSION_NON_INHERITED

        assert FIELD_REINFORCE_COUNTS not in _VERSION_NON_INHERITED
        assert FIELD_REINFORCE_COUNTS in _VERSION_CALLER_ONLY_BLOCKED


class TestExposureTime:
    """O timestamp do T3 é o instante em que o caller VIU a memória, não o
    instante em que o worker rodou. Sob backlog os dois divergem, e a diferença
    cai direto na linha do tempo ACT-R e nas duas janelas."""

    def _store(self):
        class S(FakeVectorStore):
            PAYLOAD_UPDATE_MERGES = True

        return S()

    def test_written_timestamp_is_the_exposure_instant(self):
        store = self._store()
        exposed = NOW - timedelta(hours=6)  # job atrasado 6h
        _reinforce_memory(store, MemoryDynamicsConfig(), "m", {"created_at": hours_ago(72)},
                          trigger=TRIGGER_SEARCH, now=exposed)
        _vid, written = store.updates[0]
        assert written["reinforced_at"][-1] == exposed.isoformat()

    def test_a_write_after_the_exposure_suppresses_the_late_job(self):
        """T2 aconteceu DEPOIS da exposição e ANTES do job T3: o evento antigo
        não pode ser anexado fora de ordem nem reabrir a janela."""
        store = self._store()
        exposed = NOW - timedelta(hours=6)
        payload = {"reinforced_at": [(NOW - timedelta(minutes=5)).isoformat()]}
        outcome = _reinforce_memory(store, MemoryDynamicsConfig(), "m", payload,
                                    trigger=TRIGGER_SEARCH, now=exposed)
        assert outcome == "suppressed"
        assert store.updates == []


class TestBackgroundDispatch:
    """Pré-filtro NEGATIVO, contadores na mesma unidade e drop visível."""

    def _dyn(self):
        return MemoryDynamicsConfig(reinforce_on_search=True, reinforce_top_n=0)

    def _targets(self, docs, exposed_at=None):
        from mem0.memory.main import _t3_targets

        return _t3_targets(self._dyn(), docs, search_id="sid",
                           exposed_at=exposed_at or datetime.now(timezone.utc))

    def test_snapshot_inside_the_window_never_costs_a_fetch(self):
        import mem0.memory.main as main_mod

        class ExplodingStore:
            def get(self, vector_id):
                raise AssertionError("não deveria buscar payload de alvo suprimido")

        recent = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        docs = [{"id": "x", "metadata": {"reinforced_at": [recent],
                                         "last_search_reinforced_at": recent}}]
        seen = []
        main_mod.reinforcement_observer = lambda *a: seen.append((a[2], a[4]))
        try:
            main_mod._reinforce_hits_in_background(ExplodingStore(), self._dyn(),
                                                   self._targets(docs))
        finally:
            main_mod.reinforcement_observer = None
        assert seen and seen[0][0] == "suppressed"
        assert seen[0][1]["prefiltered"] is True

    def test_backlog_full_emits_a_dropped_event_per_memory(self):
        import mem0.memory.main as main_mod

        seen = []
        main_mod.reinforcement_observer = lambda *a: seen.append((a[1], a[2]))
        original = main_mod._reinforce_pending
        main_mod._reinforce_pending = main_mod._REINFORCE_MAX_PENDING
        try:
            docs = [{"id": "a", "metadata": {}}, {"id": "b", "metadata": {}}]
            main_mod._reinforce_hits_in_background(FakeVectorStore(), self._dyn(),
                                                   self._targets(docs))
        finally:
            main_mod._reinforce_pending = original
            main_mod.reinforcement_observer = None
        # drop é EVENTO, não só contador: antes a exposição descartada não
        # aparecia em lugar nenhum do stream.
        assert [o for _t, o in seen] == ["dropped", "dropped"]

    def test_counters_use_the_same_unit(self):
        """`pending` contava JOBS e `dropped` contava MEMÓRIAS no mesmo gauge."""
        import mem0.memory.main as main_mod

        original = main_mod._reinforce_dropped
        main_mod._reinforce_pending = main_mod._REINFORCE_MAX_PENDING
        try:
            docs = [{"id": f"m{i}", "metadata": {}} for i in range(3)]
            main_mod._reinforce_hits_in_background(FakeVectorStore(), self._dyn(),
                                                   self._targets(docs))
            assert main_mod.reinforcement_backlog()["dropped"] == original + 3
        finally:
            main_mod._reinforce_pending = 0
            main_mod._reinforce_dropped = original

    def test_submit_failure_does_not_leak_pending(self):
        import mem0.memory.main as main_mod

        class BrokenExecutor:
            def submit(self, *a, **k):
                raise RuntimeError("interpreter shutting down")

        original_get = main_mod._get_reinforce_executor
        main_mod._get_reinforce_executor = lambda: BrokenExecutor()
        before = main_mod.reinforcement_backlog()["pending"]
        try:
            docs = [{"id": "a", "metadata": {}}]
            main_mod._reinforce_hits_in_background(FakeVectorStore(), self._dyn(),
                                                   self._targets(docs))
        finally:
            main_mod._get_reinforce_executor = original_get
        assert main_mod.reinforcement_backlog()["pending"] == before


class TestWindowInteraction:
    """A janela global (1h, todos os gatilhos) e a de exposição (24h, só T3) se
    COMPÕEM — não são orçamentos separados. A documentação dizia "não dividem
    orçamento", o que é falso numa direção: um T3 recente cala T1/T2 por 1h."""

    def test_recent_t3_mutes_an_explicit_write_for_the_global_window(self):
        payload = {
            "reinforced_at": [hours_ago(0.5)],
            FIELD_LAST_SEARCH_REINFORCED_AT: hours_ago(0.5),
        }
        assert should_reinforce(payload, now=NOW, window_seconds=3600, trigger="t1",
                                search_window_seconds=86400) is False

    def test_after_the_global_window_an_explicit_write_passes(self):
        payload = {
            "reinforced_at": [hours_ago(5)],
            FIELD_LAST_SEARCH_REINFORCED_AT: hours_ago(5),
        }
        assert should_reinforce(payload, now=NOW, window_seconds=3600, trigger="t2",
                                search_window_seconds=86400) is True

    def test_search_stays_muted_long_after_the_global_window(self):
        payload = {
            "reinforced_at": [hours_ago(5)],
            FIELD_LAST_SEARCH_REINFORCED_AT: hours_ago(5),
        }
        assert should_reinforce(payload, now=NOW, window_seconds=3600,
                                trigger=TRIGGER_SEARCH, search_window_seconds=86400) is False


class TestVersionInheritance:
    """v0.9 INVERTEU o contrato: um fato atualizado é o MESMO fato, evoluído —
    a versão nova COPIA a timeline do head (decisão version_inherits_dynamics,
    default on) e ganha um T2. O que a versão pré-v0.9 protegia (blacklist
    cega para o HEAD) virou proteção só contra o CALLER: um cliente nunca
    forja timeline via update(metadata=...). A completude continua derivada da
    tupla única — um campo novo de dynamics nasce bloqueado para forgery por
    default, sem depender de lista mantida à mão."""

    def test_every_dynamics_field_is_caller_forgery_blocked(self):
        from mem0.memory.main import _VERSION_CALLER_ONLY_BLOCKED, _VERSION_NON_INHERITED

        missing = [f for f in DYNAMICS_FIELDS if f not in _VERSION_CALLER_ONLY_BLOCKED]
        assert missing == [], f"campos de dynamics forjáveis pelo caller: {missing}"
        leaked = [f for f in DYNAMICS_FIELDS if f in _VERSION_NON_INHERITED]
        assert leaked == [], (
            f"campos de dynamics ainda na blacklist cega (a herança viraria no-op): {leaked}"
        )


class TestVersionDynamicsPlanner:
    """_plan_version_dynamics: o lado ACT-R de um update versionado, PURO e
    compartilhado pelos twins sync/async (a semântica não pode derivar).

    O pin central é o ANTI-DEGRAU: o T2 planeja sobre o payload do HEAD, nunca
    o da versão nova. Com head neutro, semear do created_at da v2 (= instante
    da operação) + evento no MESMO instante cunharia boost 0.667 do nada —
    PIOR que a opção A medida e revertida (0.5)."""

    def _plan(self, head, dyn=None, op_ts=None, inherit=True):
        from mem0.memory.main import _plan_version_dynamics

        return _plan_version_dynamics(
            head, dyn if dyn is not None else MemoryDynamicsConfig(),
            op_ts or NOW.isoformat(), inherit=inherit,
        )

    def test_not_inheriting_returns_nothing(self):
        extra, outcome = self._plan({"created_at": hours_ago(72)}, inherit=False)
        assert extra == {} and outcome is None

    def test_neutral_head_seeds_from_the_heads_created_at(self):
        """Paridade com o T2 legado: mesmo seed, mesmo evento — nunca o degrau."""
        head = {"data": "fato", "created_at": hours_ago(72)}
        extra, outcome = self._plan(head)
        assert outcome == "applied"
        assert extra["first_seen_at"] == hours_ago(72)
        assert extra["reinforced_at"][0] == hours_ago(72), \
            "seed TEM que ser o created_at do HEAD, não o instante da operação"
        assert extra["reinforced_by"] == ["created", "t2"]
        legacy_fields, _ = plan_reinforcement(head, MemoryDynamicsConfig(), "t2", now=NOW)
        assert extra["reinforced_at"] == legacy_fields["reinforced_at"]

    def test_reinforced_head_appends_t2_and_carries_tally(self):
        head = {"data": "fato", "created_at": hours_ago(96),
                "reinforced_at": [hours_ago(96), hours_ago(48)],
                "reinforced_by": ["created", "t3"],
                "access_count": 2, "reinforce_counts": {"t3": 1}}
        extra, outcome = self._plan(head)
        assert outcome == "applied"
        assert extra["reinforced_at"][0] == hours_ago(96)  # timeline preservada
        assert extra["reinforced_by"] == ["created", "t3", "t2"]
        assert extra["reinforce_counts"] == {"t3": 1, "t2": 1}, \
            "tally herdado junto = sem migração fabricando bucket 'unknown'"
        assert extra["access_count"] == 3

    def test_window_suppresses_the_event_but_not_the_anchor(self):
        head = {"data": "fato", "created_at": hours_ago(72),
                "reinforced_at": [hours_ago(0.2)], "access_count": 2}
        extra, outcome = self._plan(head)
        assert outcome == "suppressed"
        assert extra == {"first_seen_at": hours_ago(72)}, \
            "a cópia (feita pelo _build) fica; só o EVENTO é suprimido"

    def test_late_queued_job_is_suppressed_not_backdated(self):
        # operation_ts (submitted_at da fila) ANTERIOR ao último evento do head:
        # disciplina do exposed_at — job atrasado não retro-data nem reabre.
        head = {"data": "fato", "created_at": hours_ago(72),
                "reinforced_at": [hours_ago(1)], "access_count": 2}
        extra, outcome = self._plan(head, op_ts=hours_ago(2))
        assert outcome == "suppressed"

    def test_dynamics_disabled_still_stamps_the_anchor(self):
        from mem0.memory.main import _plan_version_dynamics

        extra, outcome = _plan_version_dynamics(
            {"created_at": hours_ago(72)}, None, NOW.isoformat(), inherit=True)
        assert extra == {"first_seen_at": hours_ago(72)} and outcome is None

    def test_anchor_propagates_across_versions(self):
        # head já é uma v2: created_at = op anterior, first_seen_at = origem
        head = {"created_at": hours_ago(24), "first_seen_at": hours_ago(720)}
        extra, _ = self._plan(head)
        assert extra["first_seen_at"] == hours_ago(720)

    def test_malformed_anchor_falls_back_to_created_at(self):
        head = {"created_at": hours_ago(24), "first_seen_at": "não é data"}
        extra, _ = self._plan(head)
        assert extra["first_seen_at"] == hours_ago(24), \
            "first_seen_at truthy-mas-inválido não pode vencer um created_at válido"


class TestSupersededEligibility:
    """v0.9: com a timeline COPIADA ao sucessor, o registro supersedido perde
    elegibilidade de ativação e de reforço — senão a família faria double-dip
    (fusão: penalidade − boost se cancelam em parte) e a exposição na busca
    re-semearia o registro velho. O t1s já pulava; T1/T3 eram a inconsistência."""

    def test_t3_targets_skip_superseded(self):
        from mem0.memory.main import _t3_targets

        dyn = MemoryDynamicsConfig(reinforce_top_n=3)
        docs = [
            {"id": "old", "metadata": {"superseded_by": "new"}},
            {"id": "new", "metadata": {}},
        ]
        got = _t3_targets(dyn, docs, search_id="s", exposed_at=NOW.isoformat())
        assert [t.memory_id for t in got] == ["new"]

    def test_post_rerank_mask_zeroes_activation_for_superseded(self):
        from mem0.memory.main import _apply_post_rerank_adjustments
        from mem0.configs.base import MemoryTemporalityConfig

        dyn = MemoryDynamicsConfig()
        temp = MemoryTemporalityConfig()
        timeline = {"reinforced_at": [hours_ago(48), hours_ago(24)], "access_count": 2}
        docs = [
            {"id": "old", "rerank_score": 0.9, "created_at": hours_ago(72),
             "metadata": {**timeline, "superseded_by": "new",
                          "superseded_at": hours_ago(1)}},
            {"id": "new", "rerank_score": 0.9, "created_at": hours_ago(1),
             "metadata": dict(timeline)},
        ]
        out = _apply_post_rerank_adjustments(docs, dyn=dyn, temp=temp)
        by_id = {d["id"]: d for d in out}
        assert "activation" not in by_id["old"], "supersedido MASCARADO"
        assert by_id["old"].get("superseded_penalty") == temp.superseded_penalty
        assert by_id["new"].get("activation", 0) > 0, "o atual mantém a ativação"

    def test_post_rerank_mask_respects_as_of_time_travel(self):
        """as_of ANTES da supersedência: o registro era o atual — sem penalidade
        E com ativação (a vista histórica fica íntegra; bônus da cópia sobre a
        transferência, que a deixaria neutra)."""
        from datetime import timedelta

        from mem0.memory.main import _apply_post_rerank_adjustments
        from mem0.configs.base import MemoryTemporalityConfig

        dyn = MemoryDynamicsConfig()
        temp = MemoryTemporalityConfig()
        docs = [{"id": "old", "rerank_score": 0.9, "created_at": hours_ago(72),
                 "metadata": {"reinforced_at": [hours_ago(48)], "access_count": 1,
                              "superseded_by": "new", "superseded_at": hours_ago(1)}}]
        anchor = NOW - timedelta(hours=24)  # antes da supersedência (1h atrás)
        out = _apply_post_rerank_adjustments(docs, dyn=dyn, temp=temp, as_of=anchor)
        assert "superseded_penalty" not in out[0]
        assert out[0].get("activation", 0) > 0

    def test_post_rerank_anchor_prefers_first_seen_at(self):
        """A v2 (created_at = agora) com cauda dobrada usa first_seen_at como
        âncora de Petrov — sem ele a cauda seria super-pesada em silêncio.

        ⚠️ timestamps relativos ao RELÓGIO REAL: o adjuster usa utcnow() interno
        (sem injeção) — o cenário fixo em 2030 fica no futuro e TODO Δt clampa
        a 1 dia, apagando a diferença que o teste mede (foi o 1º modo de falha
        deste teste)."""
        from datetime import timedelta

        from mem0.memory.main import _apply_post_rerank_adjustments
        from mem0.utils.dynamics import utcnow

        real_now = utcnow()

        def ago(days):
            return (real_now - timedelta(days=days)).isoformat()

        dyn = MemoryDynamicsConfig()
        base_meta = {"reinforced_at": [ago(3), ago(1)], "access_count": 30}
        with_anchor = [{"id": "a", "rerank_score": 0.9, "created_at": ago(0.001),
                        "metadata": {**base_meta, "first_seen_at": ago(365)}}]
        without = [{"id": "a", "rerank_score": 0.9, "created_at": ago(0.001),
                    "metadata": dict(base_meta)}]
        got_anchor = _apply_post_rerank_adjustments(with_anchor, dyn=dyn)[0]["activation"]
        got_plain = _apply_post_rerank_adjustments(without, dyn=dyn)[0]["activation"]
        assert got_anchor < got_plain, \
            "âncora antiga espalha a cauda dobrada => ativação MENOR que o fallback"


class TestConfigSurface:
    def test_defaults(self):
        dyn = MemoryConfig().dynamics
        assert dyn.enabled is True
        assert dyn.decay == 0.5
        assert dyn.weight == 0.15
        assert dyn.reinforcement_window == 3600
        assert dyn.max_timestamps == 10
        assert dyn.reinforce_on_search is False
        assert dyn.reinforce_top_n == 3
        assert dyn.reinforce_on_search_window == 86400

    def test_version_inherits_dynamics_defaults_on(self):
        assert MemoryConfig().temporality.version_inherits_dynamics is True

    def test_disabled_dynamics_resolves_to_none(self):
        config = MemoryConfig(dynamics=MemoryDynamicsConfig(enabled=False))
        assert _dynamics_config(config) is None
        assert _dynamics_config(MemoryConfig()) is not None

    def test_from_dict_config(self):
        config = MemoryConfig(**{"dynamics": {"weight": 0.3, "reinforcement_window": 7200}})
        assert config.dynamics.weight == 0.3
        assert config.dynamics.reinforcement_window == 7200
        assert config.dynamics.enabled is True
