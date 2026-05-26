import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .search import HybridSearchEngine

_engine: HybridSearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    _engine = HybridSearchEngine(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        chroma_dir=os.environ.get("CHROMA_DIR", "./data/chroma"),
    )
    yield


app = FastAPI(
    title="AI Document Search",
    description="Hybrid BM25 + semantic search with RRF fusion.",
    version="1.0.0",
    lifespan=lifespan,
)


class IndexRequest(BaseModel):
    documents: list[dict] = Field(..., description="List of {id, text, metadata} dicts")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(5, ge=1, le=20)


class SearchResult(BaseModel):
    text: str
    metadata: dict
    score: float
    retrieval: str


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
    query: str


@app.post("/index")
async def index_documents(request: IndexRequest) -> dict:
    n = _engine.index(request.documents)
    return {"indexed": n, "total_in_index": _engine.document_count}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    if _engine.document_count == 0:
        raise HTTPException(status_code=404, detail="Index is empty. POST to /index first.")
    results = _engine.search(request.query, k=request.k)
    return SearchResponse(
        results=[SearchResult(**r) for r in results],
        total=len(results),
        query=request.query,
    )


@app.get("/status")
async def status() -> dict:
    return {"status": "ok", "documents_indexed": _engine.document_count}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
