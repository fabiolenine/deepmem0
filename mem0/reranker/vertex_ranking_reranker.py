import os
from typing import Any, Dict, List

from mem0.reranker.base import BaseReranker

try:
    import google.auth
    from google.auth.transport.requests import AuthorizedSession

    GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    GOOGLE_AUTH_AVAILABLE = False

_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class VertexRankingError(RuntimeError):
    """The Ranking API did not give every candidate a score.

    Raised instead of degrading: the Cohere reranker's fallback writes 0.0 to
    every document, which downstream is indistinguishable from "every candidate
    is irrelevant" -- and the superseded penalty and tie bands then operate on a
    flat line. The core already catches reranker exceptions and keeps the fused
    order, with a warning, so raising costs a search nothing but its rerank.
    """


class VertexRankingReranker(BaseReranker):
    """Google Vertex AI Ranking API (``rankingConfigs.rank``) over REST.

    SCORE CONTRACT. The API returns a relevance score in [0, 1] per record, so
    the range half of ``BaseReranker``'s contract holds by construction. The
    AXIS does not match bge-reranker-v2-m3: on the F0 smoke a clearly relevant
    PT record scored 0.248 on default-004 against ~0.99 on the cross-encoder.
    Constants calibrated on the old axis (superseded penalty 0.2, tie bands)
    therefore mean something else here -- that is the caller's calibration
    problem, and this class does not rescale to hide it.

    The ABSOLUTE half was measured live (08/10/2026, default-004 and fast-004):
    a record's score is identical alone, among its peers, among 25 distractors,
    and in reversed order (max |delta| 0.0). That is what makes splitting a pool
    larger than the API limit into several requests safe.

    ⚠️ Scores come back QUANTIZED to 4 decimals (1e-4). Eight near-duplicates
    of one fact spanned only 0.0037, so exact ties are reachable and a tie band
    of 0.008 swallows the whole near-duplicate group.

    ⚠️ Proto3 JSON omits a float field equal to 0, so a record that comes back
    with an ``id`` and no ``score`` scored exactly 0.0. A record that does not
    come back at all is an error.
    """

    def __init__(self, config):
        if not GOOGLE_AUTH_AVAILABLE:
            raise ImportError(
                "google-auth is required for VertexRankingReranker. Install with: pip install google-auth requests"
            )
        self.config = config
        credentials, adc_project = google.auth.default(scopes=[_SCOPE])
        self.project = config.project or os.getenv("GOOGLE_CLOUD_PROJECT") or adc_project
        if not self.project:
            raise ValueError(
                "Vertex Ranking needs a GCP project: set it in the reranker config, "
                "GOOGLE_CLOUD_PROJECT, or the ADC quota project."
            )
        self.model = config.model
        self.session = AuthorizedSession(credentials)
        self.url = (
            f"https://discoveryengine.googleapis.com/v1/projects/{self.project}"
            f"/locations/{config.location}/rankingConfigs/{config.ranking_config}:rank"
        )

    @staticmethod
    def _text(doc: Dict[str, Any]) -> str:
        for key in ("memory", "text", "content"):
            if key in doc:
                return doc[key]
        return str(doc)

    def _score_batch(self, query: str, texts: List[str]) -> List[float]:
        body = {
            "model": self.model,
            "query": query,
            "records": [{"id": str(i), "content": t} for i, t in enumerate(texts)],
            "topN": len(texts),
            "ignoreRecordDetailsInResponse": True,
        }
        resp = self.session.post(self.url, json=body, timeout=self.config.timeout_s)
        if resp.status_code != 200:
            # The core logs this exception, and the request carried the query and
            # the memories. A 400 INVALID_ARGUMENT message can quote the offending
            # field, so it is dropped; permission/quota/server messages are about
            # the caller, not the content, and are what diagnoses the failure.
            try:
                err = resp.json().get("error", {})
            except ValueError:
                err = {}
            status = err.get("status", "")
            detail = "" if resp.status_code == 400 or status == "INVALID_ARGUMENT" else str(err.get("message", ""))[:160]
            raise VertexRankingError(f"rank HTTP {resp.status_code} {status}: {detail}".rstrip(": "))
        records = resp.json().get("records", [])
        scores: Dict[int, float] = {}
        for rec in records:
            scores[int(rec["id"])] = float(rec.get("score", 0.0))
        missing = [i for i in range(len(texts)) if i not in scores]
        if missing:
            raise VertexRankingError(
                f"rank returned {len(scores)} of {len(texts)} records; missing ids {missing[:10]}"
            )
        return [scores[i] for i in range(len(texts))]

    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        if not documents:
            return documents

        texts = [self._text(doc) for doc in documents]
        step = self.config.max_records_per_request
        scores: List[float] = []
        for start in range(0, len(texts), step):
            scores.extend(self._score_batch(query, texts[start:start + step]))

        reranked = []
        for doc, score in zip(documents, scores):
            out = doc.copy()
            out["rerank_score"] = score
            reranked.append(out)
        # sorted() is stable: equal scores keep the fused order they arrived in.
        reranked.sort(key=lambda d: d["rerank_score"], reverse=True)

        final_top_k = top_k or self.config.top_k
        return reranked[:final_top_k] if final_top_k else reranked
