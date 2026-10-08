from typing import Optional

from pydantic import Field

from mem0.configs.rerankers.base import BaseRerankerConfig


class VertexRankingRerankerConfig(BaseRerankerConfig):
    """
    Configuration for the Google Vertex AI Ranking API reranker
    (Discovery Engine ``rankingConfigs.rank``).

    Authentication is Application Default Credentials (a service account on the
    host, or ``gcloud auth application-default login`` with impersonation) -- the
    API takes no key, so ``api_key`` is ignored.
    """

    model: Optional[str] = Field(
        default="semantic-ranker-default-004",
        description="Ranking model id (semantic-ranker-default-004, semantic-ranker-fast-004, ...)",
    )
    project: Optional[str] = Field(
        default=None,
        description="GCP project id. None = GOOGLE_CLOUD_PROJECT, then the ADC project.",
    )
    location: str = Field(default="global", description="Ranking API location")
    ranking_config: str = Field(default="default_ranking_config", description="Ranking config id")
    timeout_s: float = Field(
        default=3.0, gt=0,
        description="Per-request timeout. A search must not wait on a slow ranker; on timeout the call raises.",
    )
    max_records_per_request: int = Field(
        default=200, ge=1, le=200,
        description="Records per request (the API accepts at most 200). Larger pools are split.",
    )
