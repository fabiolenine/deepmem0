"""The sentence-transformers side of the [0, 1] rerank_score contract.

`SentenceTransformerReranker` stores `CrossEncoder.predict()` output verbatim as
`rerank_score`. That is in-contract ONLY because sentence-transformers applies
`nn.Sigmoid` by default for a `num_labels=1` cross-encoder. Nothing in our code
enforces it, so this pins the premise: if a sentence-transformers upgrade changed
that default, the consumer would silently start clamping and the superseded
penalty plus the tie bands would go inert.

Two layers, deliberately separated:

* the DEFAULT-SELECTION rule, tested against a stub — cheap, deterministic, and
  the actual thing that could change under us;
* the real `BAAI/bge-reranker-v2-m3`, marked `live_model` and deselected by
  default (loading it costs ~19s and needs the HF cache, which would make the
  normal suite environment-dependent).
"""

import pytest

torch = pytest.importorskip("torch")
st_model = pytest.importorskip("sentence_transformers.cross_encoder.model")

CrossEncoder = st_model.CrossEncoder


class _StubConfig:
    """Mimics a HF config with no declared activation override."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _StubCrossEncoder:
    """Just enough surface for `get_default_activation_fn` to run unbound."""

    def __init__(self, num_labels, config=None):
        self.num_labels = num_labels
        self.config = config if config is not None else _StubConfig()
        self.trust_remote_code = False

    get_default_activation_fn = CrossEncoder.get_default_activation_fn
    _resolve_activation_fn = CrossEncoder._resolve_activation_fn


class TestDefaultActivationSelection:
    def test_single_label_cross_encoder_defaults_to_sigmoid(self):
        """The load-bearing rule: num_labels=1 -> Sigmoid -> scores in (0, 1).

        `bge-reranker-v2-m3` has id2label {"0": "LABEL_0"}, i.e. num_labels=1,
        and declares no activation override.
        """
        fn = _StubCrossEncoder(num_labels=1).get_default_activation_fn()
        assert isinstance(fn, torch.nn.Sigmoid)

    def test_multi_label_defaults_to_identity_which_breaks_the_contract(self):
        """The counterfactual, so the test above is not vacuous.

        A num_labels>1 model emits RAW logits through predict(), which is exactly
        the out-of-contract case the consumer clamps and warns about.
        """
        fn = _StubCrossEncoder(num_labels=3).get_default_activation_fn()
        assert isinstance(fn, torch.nn.Identity)

    def test_sigmoid_maps_logits_into_the_contract(self):
        fn = _StubCrossEncoder(num_labels=1).get_default_activation_fn()
        out = fn(torch.tensor([-9.97, 0.0, 9.06]))
        assert all(0.0 < float(v) < 1.0 for v in out)

    def test_predict_applies_the_activation(self):
        """Pins that predict() APPLIES activation_fn rather than merely holding it.

        The reranker calls predict() with no activation_fn argument, so the
        `activation_fn or self.activation_fn` fallback is what puts the score in
        contract. A refactor that dropped the call would be invisible otherwise.
        """
        import inspect

        src = inspect.getsource(CrossEncoder.predict)
        assert "activation_fn = activation_fn or self.activation_fn" in src
        assert "scores = activation_fn(scores)" in src


@pytest.mark.live_model
class TestRealRerankerEmitsContractScores:
    """Deselected by default (`addopts = -m 'not live_model'`). Run explicitly:

        pytest tests/rerankers/test_sentence_transformer_score_space.py -m live_model
    """

    MODEL = "BAAI/bge-reranker-v2-m3"

    def test_production_model_scores_are_in_zero_one(self):
        model = CrossEncoder(self.MODEL, device="cpu")
        scores = model.predict([
            ["qual é o limite de memória dos workers?", "os workers usam 2 GB de RAM cada"],
            ["qual é o limite de memória dos workers?", "a receita do bolo leva três ovos"],
        ])
        values = [float(s) for s in scores]
        assert all(0.0 <= v <= 1.0 for v in values), values
        # and it must still DISCRIMINATE — bounded-but-constant would pass the
        # range check while telling us nothing.
        assert values[0] > values[1], values
