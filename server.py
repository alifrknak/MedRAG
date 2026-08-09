import os
import sys
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import config
from ollama_embedder import OllamaEmbedder
from vector_db import LocalVectorDB

# Set stdout encoding to UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedRAG.Server")

app = FastAPI(
    title="MedRAG Search & Chat API",
    description="REST API & Web UI for Medical Vector Search, Hybrid Retrieval, Reranking & Safety Gate Filtering",
    version="1.0.0"
)

# Global lazy singletons
embedder_instance: Optional[OllamaEmbedder] = None
vector_db_instance: Optional[LocalVectorDB] = None

def get_embedder() -> OllamaEmbedder:
    global embedder_instance
    if embedder_instance is None:
        logger.info("Initializing OllamaEmbedder singleton...")
        embedder_instance = OllamaEmbedder()
    return embedder_instance

def get_vector_db() -> LocalVectorDB:
    global vector_db_instance
    if vector_db_instance is None:
        logger.info("Initializing LocalVectorDB singleton...")
        vector_db_instance = LocalVectorDB()
    return vector_db_instance

class SearchRequest(BaseModel):
    query: str = Field(..., description="User search text or medical question", example="Diyabet hastalığının belirtileri ve tedavisi nedir?")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of results to return")
    similarity_threshold: Optional[float] = Field(default=config.SIMILARITY_THRESHOLD, ge=0.0, le=1.0, description="Similarity cutoff threshold")
    use_hybrid: bool = Field(default=config.USE_HYBRID_SEARCH, description="Enable BM25 + Dense Vector RRF fusion")
    use_reranker: bool = Field(default=config.USE_RERANKER, description="Enable Cross-Encoder Reranker")

class SearchResponse(BaseModel):
    query: str
    total_results: int
    similarity_threshold: float
    safety_gate_triggered: bool
    safety_gate_message: Optional[str] = None
    results: List[Dict[str, Any]]

@app.on_event("startup")
def startup_event():
    logger.info("Starting MedRAG FastAPI Application...")
    # Pre-warm singletons
    try:
        get_vector_db()
        logger.info("Vector DB initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing Vector DB: {e}")

@app.get("/api/v1/stats")
def get_stats():
    """Returns Vector DB document count and system configuration."""
    vdb = get_vector_db()
    return {
        "status": "online",
        "total_records": vdb.count(),
        "collection_name": config.COLLECTION_NAME,
        "ollama_url": config.OLLAMA_URL,
        "embedding_model": config.MODEL_NAME,
        "reranker_model": config.RERANKER_MODEL_NAME,
        "default_similarity_threshold": config.SIMILARITY_THRESHOLD,
        "use_hybrid_search": config.USE_HYBRID_SEARCH,
        "use_reranker": config.USE_RERANKER
    }

@app.post("/api/v1/search", response_model=SearchResponse)
def api_search(req: SearchRequest):
    """
    Executes hybrid search + reranking over ChromaDB medical articles.
    Enforces the Similarity Threshold Safety Gate filter.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Search query cannot be empty.")

    try:
        embedder = get_embedder()
        vdb = get_vector_db()

        # Step 1: Generate query vector
        query_vector = embedder.get_embedding(req.query)

        # Step 2: Execute search
        threshold = req.similarity_threshold if req.similarity_threshold is not None else config.SIMILARITY_THRESHOLD

        results = vdb.search(
            query_vector=query_vector,
            query_text=req.query,
            top_k=req.top_k,
            similarity_threshold=threshold,
            use_hybrid=req.use_hybrid,
            use_reranker=req.use_reranker
        )

        safety_triggered = (len(results) == 0)
        safety_msg = None
        if safety_triggered:
            safety_msg = (
                f"Tıbbi Bilgi Filtresi: Aradığınız konu tıbbi veritabanımızdaki makalelerle uyuşmamaktadır. "
                f"MedRAG yalnızca doğrulanmış hastane makaleleri üzerinden yanıt verir."
            )

        # Format results (clean numpy floats/types for JSON)
        formatted_results = []
        for r in results:
            formatted_results.append({
                "chunk_id": str(r.get("chunk_id", "")),
                "url": r.get("url"),
                "chunk_text": r.get("chunk_text", ""),
                "similarity_score": float(r.get("similarity_score", 0.0)),
                "distance": float(r.get("distance", 1.0)),
                "rrf_score": float(r["rrf_score"]) if "rrf_score" in r else None,
                "rerank_score": float(r["rerank_score"]) if "rerank_score" in r else None
            })

        return SearchResponse(
            query=req.query,
            total_results=len(formatted_results),
            similarity_threshold=threshold,
            safety_gate_triggered=safety_triggered,
            safety_gate_message=safety_msg,
            results=formatted_results
        )
    except Exception as e:
        logger.error(f"Search API Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Mount static folder if exists
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def serve_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"message": "MedRAG API active. Create static/index.html for Web UI."})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
