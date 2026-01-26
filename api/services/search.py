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

    def search(self, query: str, limit: int = 3) -> SearchResponse:
        query_dense, query_sparse, query_colbert = self.embedding_service.embed_query(
            query
        )

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
