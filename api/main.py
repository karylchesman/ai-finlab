from config.settings import settings
from fastapi import FastAPI
from models.search import SearchRequest, SearchResponse
from services.search import SearchService

app = FastAPI(title="AI Financial Search API")

search_service = SearchService(
    qdrant_url=settings.qdrant_url,
    qdrant_api_key=settings.qdrant_api_key,
    collection_name=settings.collection_name,
)


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    return search_service.search(query=request.query, limit=request.limit)


@app.get("/")
def root():
    return {"status": "online"}
