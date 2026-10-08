"""Vertex AI Ranking reranker: score mapping, failure semantics, batching.

The failure half matters most. A reranker that "degrades" by writing 0.0 to
every candidate is indistinguishable downstream from one that judged them all
irrelevant, so this provider must RAISE whenever a candidate goes unscored.
"""

from unittest.mock import MagicMock

import pytest

import mem0.reranker.vertex_ranking_reranker as vr
from mem0.configs.rerankers.vertex_ranking import VertexRankingRerankerConfig
from mem0.reranker.vertex_ranking_reranker import VertexRankingError, VertexRankingReranker
from mem0.utils.factory import RerankerFactory


class _Resp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = str(self._payload)

    def json(self):
        return self._payload


@pytest.fixture
def session(monkeypatch):
    sess = MagicMock()
    monkeypatch.setattr(vr, "GOOGLE_AUTH_AVAILABLE", True)
    monkeypatch.setattr(vr.google.auth, "default", lambda scopes: (object(), "adc-project"))
    monkeypatch.setattr(vr, "AuthorizedSession", lambda creds: sess)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    return sess


def _reranker(**cfg):
    return VertexRankingReranker(VertexRankingRerankerConfig(**cfg))


def _docs(n):
    return [{"id": f"m{i}", "memory": f"texto {i}"} for i in range(n)]


def _scores(*pairs):
    """Response body in the API's shape: sorted by score, ids as strings."""
    recs = [{"id": str(i), "score": s} for i, s in pairs]
    recs.sort(key=lambda r: r["score"], reverse=True)
    return _Resp(200, {"records": recs})


def test_scores_land_on_the_right_document_and_sort_descending(session):
    session.post.return_value = _scores((0, 0.10), (1, 0.90), (2, 0.40))
    out = _reranker().rerank("q", _docs(3))
    assert [d["id"] for d in out] == ["m1", "m2", "m0"]
    assert [d["rerank_score"] for d in out] == [0.90, 0.40, 0.10]


def test_equal_scores_keep_the_fused_order(session):
    session.post.return_value = _scores((0, 0.5), (1, 0.5), (2, 0.5))
    out = _reranker().rerank("q", _docs(3))
    assert [d["id"] for d in out] == ["m0", "m1", "m2"]


def test_input_documents_are_not_mutated(session):
    docs = _docs(2)
    session.post.return_value = _scores((0, 0.2), (1, 0.3))
    _reranker().rerank("q", docs)
    assert all("rerank_score" not in d for d in docs)


def test_record_without_score_field_scored_zero(session):
    # proto3 JSON omits a float equal to 0: id present, score absent.
    session.post.return_value = _Resp(200, {"records": [{"id": "1", "score": 0.7}, {"id": "0"}]})
    out = _reranker().rerank("q", _docs(2))
    assert [(d["id"], d["rerank_score"]) for d in out] == [("m1", 0.7), ("m0", 0.0)]


def test_missing_record_raises_instead_of_zeroing(session):
    session.post.return_value = _Resp(200, {"records": [{"id": "0", "score": 0.7}]})
    with pytest.raises(VertexRankingError, match="1 of 2"):
        _reranker().rerank("q", _docs(2))


def test_http_error_keeps_status_and_diagnostic_message(session):
    session.post.return_value = _Resp(403, {"error": {
        "code": 403, "status": "PERMISSION_DENIED",
        "message": "Permission 'discoveryengine.rankingConfigs.rank' denied"}})
    with pytest.raises(VertexRankingError) as exc:
        _reranker().rerank("q", _docs(1))
    msg = str(exc.value)
    assert "403" in msg and "PERMISSION_DENIED" in msg and "rankingConfigs.rank" in msg


def test_invalid_argument_message_is_dropped_because_it_can_quote_content(session):
    session.post.return_value = _Resp(400, {"error": {
        "code": 400, "status": "INVALID_ARGUMENT",
        "message": "Invalid value at 'records[0].content': 'memória sigilosa'"}})
    with pytest.raises(VertexRankingError) as exc:
        _reranker().rerank("consulta sigilosa", [{"memory": "memória sigilosa"}])
    msg = str(exc.value)
    assert "400" in msg and "INVALID_ARGUMENT" in msg
    assert "sigilosa" not in msg


def test_transport_error_propagates(session):
    session.post.side_effect = TimeoutError("read timed out")
    with pytest.raises(TimeoutError):
        _reranker().rerank("q", _docs(2))


def test_request_body_and_timeout(session):
    session.post.return_value = _scores((0, 0.1), (1, 0.2))
    _reranker(model="semantic-ranker-fast-004", timeout_s=2.5).rerank("a pergunta", _docs(2))
    (url,), kwargs = session.post.call_args
    assert url == ("https://discoveryengine.googleapis.com/v1/projects/adc-project"
                   "/locations/global/rankingConfigs/default_ranking_config:rank")
    body = kwargs["json"]
    assert body["model"] == "semantic-ranker-fast-004"
    assert body["query"] == "a pergunta"
    assert body["records"] == [{"id": "0", "content": "texto 0"}, {"id": "1", "content": "texto 1"}]
    assert body["topN"] == 2  # every candidate must come back scored
    assert body["ignoreRecordDetailsInResponse"] is True
    assert kwargs["timeout"] == 2.5


def test_pool_larger_than_api_limit_is_split_and_ids_remapped(session):
    calls = []

    def fake_post(url, json, timeout):
        calls.append(len(json["records"]))
        # score from the CONTENT ("texto N" -> N/1000), so any id/position
        # remapping bug moves a score onto the wrong document
        return _scores(*[(int(r["id"]), int(r["content"].split()[1]) / 1000) for r in json["records"]])

    session.post.side_effect = fake_post
    out = _reranker(max_records_per_request=200).rerank("q", _docs(450))
    assert calls == [200, 200, 50]
    assert [d["id"] for d in out[:3]] == ["m449", "m448", "m447"]
    assert out[-1]["id"] == "m0"
    assert all(d["rerank_score"] == int(d["id"][1:]) / 1000 for d in out)


def test_top_k_argument_then_config_fallback(session):
    session.post.return_value = _scores((0, 0.1), (1, 0.3), (2, 0.2))
    assert [d["id"] for d in _reranker().rerank("q", _docs(3), top_k=2)] == ["m1", "m2"]
    session.post.return_value = _scores((0, 0.1), (1, 0.3), (2, 0.2))
    assert [d["id"] for d in _reranker(top_k=1).rerank("q", _docs(3))] == ["m1"]


def test_empty_documents_make_no_request(session):
    assert _reranker().rerank("q", []) == []
    session.post.assert_not_called()


def test_project_precedence_config_then_env_then_adc(session, monkeypatch):
    assert _reranker(project="cfg-project").project == "cfg-project"
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    assert _reranker().project == "env-project"
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT")
    assert _reranker().project == "adc-project"


def test_no_project_anywhere_fails_at_construction(monkeypatch):
    monkeypatch.setattr(vr, "GOOGLE_AUTH_AVAILABLE", True)
    monkeypatch.setattr(vr.google.auth, "default", lambda scopes: (object(), None))
    monkeypatch.setattr(vr, "AuthorizedSession", lambda creds: MagicMock())
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    with pytest.raises(ValueError, match="project"):
        _reranker()


def test_batch_limit_cannot_exceed_api_maximum():
    with pytest.raises(ValueError):
        VertexRankingRerankerConfig(max_records_per_request=201)


def test_factory_builds_it_from_a_dict(session):
    r = RerankerFactory.create("vertex_ranking", {"model": "semantic-ranker-default-004", "project": "p"})
    assert isinstance(r, VertexRankingReranker)
    assert r.project == "p"
