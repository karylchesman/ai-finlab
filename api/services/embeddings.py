from config.settings import settings
from fastembed import LateInteractionTextEmbedding, SparseTextEmbedding, TextEmbedding


class EmbeddingService:
    def __init__(self):
        self.dense_model = TextEmbedding(model_name=settings.dense_model)
        self.sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)
        self.colbert_model = LateInteractionTextEmbedding(
            model_name=settings.colbert_model
        )

    def embed_query(self, query: str):
        dense_embedding = list(self.dense_model.query_embed([query]))[0].tolist()
        sparse_embedding = list(self.sparse_model.query_embed([query]))[0].as_object()
        colbert_embedding = list(self.colbert_model.query_embed([query]))[0].tolist()

        return (dense_embedding, sparse_embedding, colbert_embedding)
