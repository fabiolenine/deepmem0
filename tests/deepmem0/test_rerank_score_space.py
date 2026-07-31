"""The rerank_score space: [0, 1] relevance, not a logit, not set-relative.

Covers the 2026-07-31 fix. `_apply_post_rerank_adjustments` used to apply a
sigmoid to `rerank_score`, which every provider already emits in [0, 1] — a
SECOND sigmoid, squeezing the axis into [0.5, 0.731]. Ordering was unaffected
(sigmoid is monotonic), which is why it survived; the superseded penalty and the
tie bands were not.

These tests are written to FAIL against the old code, not merely to pass against
the new one — see `TestBandIsInRelevanceSpace`, which pins the band to a gap the
two spaces classify differently.
"""

import logging

import pytest

from mem0.configs.base import (
    RERANK_TIE_BAND,
    MemoryDynamicsConfig,
    MemoryTemporalityConfig,
)
from mem0.memory import main as main_mod
from mem0.memory.main import (
    _apply_post_rerank_adjustments,
    _relevance_from_rerank_score,
)


@pytest.fixture(autouse=True)
def _reset_contract_warning():
    """The warning is deduplicated per process; each test starts clean."""
    main_mod._rerank_contract_warned = False
    yield
    main_mod._rerank_contract_warned = False


class TestRelevanceIsReadVerbatim:
    @pytest.mark.parametrize("value", [0.0, 4.7e-05, 0.064, 0.5, 0.9, 0.999884, 1.0])
    def test_in_contract_score_passes_through_unchanged(self, value):
        """The heart of the fix: no transform at all on a contract-abiding score.

        Against the old code every one of these returns sigmoid(value) instead —
        e.g. 0.0 -> 0.5, which is the bug in one line.
        """
        assert _relevance_from_rerank_score(value) == value

    def test_zero_is_zero_not_one_half(self):
        # Called out separately because 0.5 is exactly what the old code returned
        # for a document the reranker judged maximally irrelevant.
        assert _relevance_from_rerank_score(0.0) == 0.0

    def test_production_range_spans_the_full_axis(self):
        """Measured production extremes must span ~1.0, not the 0.231 of [0.5, 0.731]."""
        lo = _relevance_from_rerank_score(4.7e-05)
        hi = _relevance_from_rerank_score(0.999884)
        assert hi - lo > 0.99


class TestOutOfContractIsClampedAndWarned:
    def test_raw_logit_above_one_is_clamped(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert _relevance_from_rerank_score(7.5) == 1.0
        assert "outside the [0, 1] contract" in caplog.text

    def test_negative_logit_is_clamped(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert _relevance_from_rerank_score(-3.0) == 0.0
        assert "outside the [0, 1] contract" in caplog.text

    def test_nan_fails_low_rather_than_through(self, caplog):
        with caplog.at_level(logging.WARNING):
            assert _relevance_from_rerank_score(float("nan")) == 0.0

    def test_warning_is_emitted_once_per_process(self, caplog):
        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                _relevance_from_rerank_score(9.0)
        hits = [r for r in caplog.records if "outside the [0, 1] contract" in r.message]
        assert len(hits) == 1, f"esperava 1 aviso, veio {len(hits)}"

    def test_clamping_preserves_order_via_stable_sort(self):
        """Documented consequence: ORDER survives out-of-contract input.

        Three raw logits all clamp to 1.0, so the sort key is identical and the
        stable sort must keep the reranker's own ordering.
        """
        docs = [
            {"id": "a", "rerank_score": 9.0, "metadata": {}},
            {"id": "b", "rerank_score": 5.0, "metadata": {}},
            {"id": "c", "rerank_score": 2.0, "metadata": {}},
        ]
        temp = MemoryTemporalityConfig()
        out = _apply_post_rerank_adjustments(docs, temp=temp)
        assert [d["id"] for d in out] == ["a", "b", "c"]


class TestSupersededPenaltyIsProportionate:
    """The penalty is documented as a [0, 1]-scale constant. It now is one."""

    def _docs(self, sup_score, cur_score):
        return [
            {"id": "sup", "rerank_score": sup_score,
             "metadata": {"superseded_by": "cur", "superseded_at": "2026-01-01T00:00:00+00:00"}},
            {"id": "cur", "rerank_score": cur_score, "metadata": {}},
        ]

    def test_penalty_is_twenty_percent_of_the_axis(self):
        """0.2 must cost 0.2 of relevance — under the double sigmoid it cost 86%
        of the reachable range, so a superseded fact could never win anything."""
        temp = MemoryTemporalityConfig()
        out = _apply_post_rerank_adjustments(self._docs(0.90, 0.85), temp=temp)
        # 0.90 - 0.20 = 0.70 < 0.85 -> demoted
        assert [d["id"] for d in out] == ["cur", "sup"]

    def test_a_decisively_better_superseded_fact_still_outranks(self):
        """The penalty DEMOTES, it does not exclude. 0.95 - 0.2 = 0.75 > 0.60.

        This is the test the old space could not express: with an effective range
        of 0.231, no in-contract pair could ever clear a 0.2 penalty.
        """
        temp = MemoryTemporalityConfig()
        out = _apply_post_rerank_adjustments(self._docs(0.95, 0.60), temp=temp)
        assert [d["id"] for d in out] == ["sup", "cur"]
        assert out[0]["superseded_penalty"] == temp.superseded_penalty


class TestBandIsInRelevanceSpace:
    """Pins the band to the space it is applied in.

    The first two tests are expressed RELATIVE to the constant (half a band,
    double a band), so they verify the tie machinery without re-encoding the
    number — they pass in either space, by construction. That is deliberate:
    the conversion was chosen to keep tie/decisive classification stable, so a
    behavioural test CANNOT distinguish the two spaces at the operating point.

    What does distinguish them is the SIZE of the constant, which is why
    ``test_band_is_bracketed_in_relevance_units`` asserts it directly: 0.002 is
    a plausible band only on the compressed axis.
    """

    def _pair(self, gap):
        """Cold leads by `gap`; hot is heavily reinforced so activation wants it."""
        from datetime import datetime, timedelta, timezone

        def ago(h):
            return (datetime.now(timezone.utc) - timedelta(hours=h)).isoformat()

        return [
            {"id": "cold", "rerank_score": 0.50 + gap, "created_at": ago(500), "metadata": {}},
            {"id": "hot", "rerank_score": 0.50, "created_at": ago(500),
             "metadata": {"reinforced_at": [ago(30), ago(4)], "access_count": 6}},
        ]

    def test_gap_just_inside_the_band_is_a_tie(self):
        dyn = MemoryDynamicsConfig()
        out = _apply_post_rerank_adjustments(self._pair(RERANK_TIE_BAND * 0.5), dyn=dyn)
        assert out[0]["id"] == "hot", "gap < banda: ativação decide"

    def test_gap_just_outside_the_band_is_decisive(self):
        dyn = MemoryDynamicsConfig()
        out = _apply_post_rerank_adjustments(self._pair(RERANK_TIE_BAND * 2.0), dyn=dyn)
        assert out[0]["id"] == "cold", "gap > banda: o reranker decide"

    def test_band_is_bracketed_in_relevance_units(self):
        """The constant must be sized for a [0, 1] axis.

        Against the pre-fix value (0.002 read on a doubly-sigmoided axis) this
        fails: 0.002 is below the lower bracket.
        """
        assert 0.004 < RERANK_TIE_BAND < 0.05

    def test_dynamics_and_event_share_the_same_default_band(self):
        assert MemoryDynamicsConfig().tie_band == RERANK_TIE_BAND
        assert MemoryTemporalityConfig().event_tie_band == RERANK_TIE_BAND


class TestProductionConfigIsInert:
    """Control for the DEPLOYED configuration, not the fork defaults.

    Production runs MEM0_DYNAMICS_WEIGHT=0 and MEM0_DYNAMICS_TIE_BAND=0, so the
    band conversion cannot reach the ACT-R path there at all. `eval_temporal.py`
    does NOT cover this: it builds `{"dynamics": {"enabled": ...}}` and inherits
    the fork defaults (weight 0.15, band 0.008). Asserted here deterministically
    rather than through an LLM-seeded eval.
    """

    def _pool(self):
        from datetime import datetime, timedelta, timezone

        def ago(h):
            return (datetime.now(timezone.utc) - timedelta(hours=h)).isoformat()

        return [
            {"id": "cold", "rerank_score": 0.5005, "created_at": ago(500), "metadata": {}},
            {"id": "hot", "rerank_score": 0.5000, "created_at": ago(500),
             "metadata": {"reinforced_at": [ago(30), ago(4)], "access_count": 6}},
        ]

    def test_zero_tie_band_makes_activation_inert_on_a_true_near_tie(self):
        """The pair is a near-tie (gap 0.0005 << 0.008): at the default band
        activation reorders it, at the production band it must not."""
        default_on = _apply_post_rerank_adjustments(self._pool(), dyn=MemoryDynamicsConfig())
        assert default_on[0]["id"] == "hot", "controle: no default a ativação decide"

        prod = MemoryDynamicsConfig(weight=0.0, tie_band=0.0)
        on = [d["id"] for d in _apply_post_rerank_adjustments(self._pool(), dyn=prod)]
        off = [d["id"] for d in _apply_post_rerank_adjustments(self._pool(), dyn=None)]
        assert on == off == ["cold", "hot"], "ON == OFF sob a config de produção"

    def test_zero_band_annotates_no_activation(self):
        prod = MemoryDynamicsConfig(weight=0.0, tie_band=0.0)
        out = _apply_post_rerank_adjustments(self._pool(), dyn=prod)
        assert not any("activation" in d for d in out)

    def test_zero_weight_is_inert_in_fusion(self):
        """The other half of the production config: the fusion term is gated by
        weight, so weight=0 must reproduce the no-boost ranking exactly."""
        from mem0.utils.scoring import score_and_rank

        cands = [
            {"id": "aaa", "score": 0.80, "payload": {"data": "fact A"}},
            {"id": "bbb", "score": 0.80, "payload": {"data": "fact B"}},
        ]
        plain = score_and_rank(cands, {}, {}, 0.1, 2)
        zeroed = score_and_rank(cands, {}, {}, 0.1, 2,
                                activation_boosts={"bbb": 0.9}, activation_weight=0.0)
        assert [r["id"] for r in plain] == [r["id"] for r in zeroed]
        assert [r["score"] for r in plain] == [r["score"] for r in zeroed]
