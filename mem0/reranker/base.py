from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseReranker(ABC):
    """Abstract base class for all rerankers.

    SCORE CONTRACT (DeepMem0): ``rerank_score`` is an ABSOLUTE relevance in
    [0, 1] -- not a raw logit, and not relative to the candidate set.

    Both halves matter to consumers:

    * **Absolute.** Post-rerank ranking adjustments subtract constants from this
      score (the superseded penalty) and compare differences against fixed tie
      bands. A set-relative score (min-max, where the worst candidate is pinned
      to 0.0 and a single candidate collapses to 0.0) makes those constants mean
      a different thing for every query.
    * **In [0, 1].** A raw cross-encoder logit spans roughly [-10, +10], so the
      same constants would be ~10x weaker against it.

    A provider that cannot honour this must say so at construction time; the
    consumer clamps out-of-contract values, which preserves ORDER (clamping is
    monotonic and the sort is stable) but silently flattens the penalty and the
    tie bands.
    """

    @abstractmethod
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to the query.

        Args:
            query: The search query
            documents: List of documents to rerank, each with 'memory' field
            top_k: Number of top documents to return (None = return all)

        Returns:
            List of reranked documents with added 'rerank_score' field,
            an absolute relevance in [0, 1] (see the class docstring).
        """
        pass