from config.settings import settings
from fastapi import FastAPI
from models.rag import RAGRequest, RAGResponse
from models.search import SearchRequest, SearchResponse
from services.rag import RAGService
from services.search import SearchService

app = FastAPI(title="AI Financial Search API")

search_service = SearchService(
    qdrant_url=settings.qdrant_url,
    qdrant_api_key=settings.qdrant_api_key,
    collection_name=settings.collection_name,
)

rag_service = RAGService(search_service=search_service)


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    return search_service.search(query=request.query, limit=request.limit)


@app.get("/")
def root():
    return {"status": "online"}


@app.post("/rag", response_model=RAGResponse)
def rag(request: RAGRequest):
    return rag_service.generate_answer(query=request.query, limit=request.limit)
