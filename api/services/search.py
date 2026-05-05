from typing import Any, Dict, List, Optional

from models.search import SearchResponse, SearchResult
from qdrant_client import QdrantClient, models

from services.embeddings import EmbeddingService


class SearchService:
    def __init__(self, qdrant_url: str, qdrant_api_key: str, collection_name: str):
        self.qdrant = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        self.collection_name = collection_name
        self.embedding_service = EmbeddingService()

    def _build_qdrant_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict]:
        if not filters:
            return None

        must_conditions = []
        for key, value in filters.items():
            must_conditions.append(
                {
                    "key": f"metadata.{key}",
                    "match": {"value": value},
                }
            )
        return {
            "must": must_conditions,
        }

    def search(
        self, query: str, limit: int = 3, filters: Optional[Dict[str, Any]] = None
    ) -> SearchResponse:
        query_dense, query_sparse, query_colbert = self.embedding_service.embed_query(
            query
        )

        query_filter = self._build_qdrant_filter(filters)

        results = self.qdrant.query_points(
            collection_name=self.collection_name,
            prefetch=[
                # type: ignore Pylance isn't recognizing the dict structure type,
                # expects Dict[str, float] but numpy is returning Dict[str, NumpyArray]
                {
                    "prefetch": [
                        {
                            "query": query_dense,
                            "using": "dense",
                            "limit": 20,
                        },
                        {
                            "query": query_sparse,
                            "using": "sparse",
                            "limit": 20,
                        },
                    ],
                    "query": models.FusionQuery(fusion=models.Fusion.RRF),
                    "limit": 15,
                }
            ],
            query=query_colbert,
            using="colbert",
            limit=3,
            query_filter=query_filter,  # type: ignore Pylance isn't recognizing the dict structure type,
        )

        max_score = max(p.score for p in results.points)
        search_results = [
            SearchResult(
                score=item.score / max_score if max_score > 0 else 0,
                text=item.payload.get("text", "") if item.payload else "",
                metadata=item.payload.get("metadata", {}) if item.payload else {},
            )
            for item in results.points
        ]

        return SearchResponse(results=search_results)
