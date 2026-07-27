"""DeepMem0 v0.2 human-memory dynamics tests — pure units, no live infrastructure."""

from datetime import datetime, timedelta, timezone

from mem0.configs.base import MemoryConfig, MemoryDynamicsConfig
from mem0.memory.main import (
    _apply_activation_post_rerank,
    _dynamics_config,
    _reinforce_memory,
)
from mem0.utils.dynamics import (
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

        # ⚠️ REALISTIC rerank logits. The bge-reranker-v2-m3 emits logits near
        # ZERO on this corpus (measured 2026-07-21: golden logits 0.01–0.25),
        # where sigmoid slope is steepest and gaps compress hardest. The old
        # fixture used logits 2.0–8.0 (sigmoid ~0.88–1.0), a region the reranker
        # NEVER reaches — so its "decisive gap" test validated a fantasy and the
        # real overturning bug went uncaught. cold≈hot here is a TRUE near-tie.
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
                "rerank_score": 0.019,  # sigmoid gap ~0.00025 < tie_band 0.002
                "metadata": {
                    "reinforced_at": [real_hours_ago(30), real_hours_ago(4)],
                    "access_count": 6,
                },
            },
        ]

    def test_reinforced_memory_wins_near_tie(self):
        # genuine tie (gap 0.00025 < tie_band): activation decides → hot wins
        dyn = MemoryDynamicsConfig()
        ordered = _apply_activation_post_rerank(self.make_docs(), dyn)
        assert ordered[0]["id"] == "hot"
        assert ordered[0]["activation"] > 0

    def test_decisive_rerank_gap_is_not_overturned(self):
        # REAL operating point (regression for the 2026-07-21 overturn bug): the
        # reranker prefers cold by a decisive 0.25-logit margin (sigmoid gap
        # ~0.06 >> tie_band). Reinforcement must NOT flip it. The additive form
        # (base + 0.15*activation) DID flip exactly this, on the live golden.
        docs = self.make_docs()
        docs[0]["rerank_score"] = 0.27  # vs hot 0.019: sigmoid gap ~0.062
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
        main_mod.reinforcement_observer = lambda mid, trig, out, ms: seen.append((mid, trig, out))
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

    def test_top_n_limits_what_a_search_reinforces(self):
        from mem0.memory.main import _t3_targets

        docs = [{"id": f"m{i}"} for i in range(10)]
        assert _t3_targets(self._dyn(reinforce_top_n=3), docs) == ["m0", "m1", "m2"]
        assert len(_t3_targets(self._dyn(reinforce_top_n=0), docs)) == 10

    def test_targets_skip_docs_without_id(self):
        from mem0.memory.main import _t3_targets

        docs = [{"id": "m0"}, {"score": 1}, {"id": "m1"}]
        assert _t3_targets(self._dyn(reinforce_top_n=0), docs) == ["m0", "m1"]


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

    def test_disabled_dynamics_resolves_to_none(self):
        config = MemoryConfig(dynamics=MemoryDynamicsConfig(enabled=False))
        assert _dynamics_config(config) is None
        assert _dynamics_config(MemoryConfig()) is not None

    def test_from_dict_config(self):
        config = MemoryConfig(**{"dynamics": {"weight": 0.3, "reinforcement_window": 7200}})
        assert config.dynamics.weight == 0.3
        assert config.dynamics.reinforcement_window == 7200
        assert config.dynamics.enabled is True
