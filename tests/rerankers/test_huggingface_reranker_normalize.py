"""Unit tests for HuggingFaceReranker score normalization.

These exercise the pure ``_normalize_scores`` helper directly, so they do not
require ``transformers`` / ``torch`` to be installed.

Ported from upstream mem0ai/mem0 PR #5715, plus the numerical-stability cases
the upstream version does not cover: the upstream helper computes
``1 / (1 + np.exp(-arr))`` directly, which overflows for large negative logits.
"""

import math

import pytest

from mem0.reranker.huggingface_reranker import HuggingFaceReranker


def _sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


class TestHuggingFaceNormalizeScores:
    def test_logits_mapped_via_sigmoid(self):
        scores = HuggingFaceReranker._normalize_scores([2.0, 8.0, 5.0])
        assert scores == pytest.approx([_sigmoid(2.0), _sigmoid(8.0), _sigmoid(5.0)])

    def test_output_bounded_between_zero_and_one(self):
        for s in HuggingFaceReranker._normalize_scores([-12.0, -1.0, 0.0, 3.0, 15.0]):
            assert 0.0 <= s <= 1.0

    def test_sigmoid_preserves_ranking_order(self):
        raw = [1.0, -4.0, 9.0, 2.5]
        normalized = HuggingFaceReranker._normalize_scores(raw)
        # argsort of raw and normalized must match — sigmoid is monotonic.
        assert sorted(range(len(raw)), key=lambda i: raw[i]) == sorted(
            range(len(normalized)), key=lambda i: normalized[i]
        )

    def test_empty_input(self):
        assert HuggingFaceReranker._normalize_scores([]) == []

    def test_single_document_is_not_forced_to_zero(self):
        """The whole point of #5715: min-max pinned a lone document to 0.0."""
        (only,) = HuggingFaceReranker._normalize_scores([3.0])
        assert only == pytest.approx(_sigmoid(3.0))
        assert only > 0.9

    def test_tied_scores_are_not_forced_to_zero(self):
        """min-max mapped an all-tied set to 0/0 -> every document 'irrelevant'."""
        scores = HuggingFaceReranker._normalize_scores([4.0, 4.0, 4.0])
        assert scores == pytest.approx([_sigmoid(4.0)] * 3)

    def test_zero_logit_is_one_half(self):
        assert HuggingFaceReranker._normalize_scores([0.0]) == pytest.approx([0.5])


class TestNormalizeScoresNumericalStability:
    """DeepMem0 addition: the naive form overflows, this one must not.

    ``np.exp(-x)`` for x = -1000 is ``exp(1000)`` = inf. numpy returns inf with a
    RuntimeWarning rather than raising, so the naive version yields the right
    number for the WRONG reason and emits a warning on every such document. The
    stable form never evaluates ``exp`` of a large positive argument.
    """

    def test_large_negative_logit_does_not_warn_or_overflow(self, recwarn):
        scores = HuggingFaceReranker._normalize_scores([-1000.0, -750.0])
        assert scores == pytest.approx([0.0, 0.0])
        overflow = [w for w in recwarn if "overflow" in str(w.message).lower()]
        assert not overflow, f"overflow warning emitted: {[str(w.message) for w in overflow]}"

    def test_large_positive_logit_saturates_to_one(self):
        assert HuggingFaceReranker._normalize_scores([1000.0]) == pytest.approx([1.0])

    def test_extremes_stay_in_contract_and_ordered(self):
        scores = HuggingFaceReranker._normalize_scores([-800.0, -1.0, 0.0, 1.0, 800.0])
        assert all(0.0 <= s <= 1.0 for s in scores)
        assert scores == sorted(scores)
