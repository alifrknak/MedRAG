import sys
import logging
import config
from ollama_embedder import OllamaEmbedder
from vector_db import LocalVectorDB

# Ensure UTF-8 console output reconfiguration for Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_query(
    query_text: str,
    embedder: OllamaEmbedder,
    vector_db: LocalVectorDB,
    top_k: int = 3,
    threshold: float = config.SIMILARITY_THRESHOLD
):
    """
    Converts query text to vector via Ollama and searches ChromaDB.
    Filters out results below the specified similarity threshold.
    """
    print(f"\n[+] Query: '{query_text}' (Similarity Threshold: >= {threshold})")
    query_vector = embedder.get_embedding(query_text)
    results = vector_db.search(query_vector, top_k=top_k, similarity_threshold=threshold)

    if not results:
        print(f"   [!] No relevant document found exceeding the similarity threshold ({threshold}).")
        print("-" * 70)
        return

    print(f"   Top Relevant Results (Found: {len(results)} items):")
    print("-" * 70)
    for idx, res in enumerate(results, 1):
        print(f"   Result #{idx}:")
        print(f"   • Similarity Score : {res['similarity_score']}")
        print(f"   • Source URL       : {res['url'] if res['url'] else '(None / Null)'}")
        print(f"   • Chunk Content    : {res['chunk_text']}")
        print("-" * 70)

def main():
    print("=" * 70)
    print(" Vector Search Service (ChromaDB + Ollama) ")
    print(f" Similarity Threshold Filter Active (>= {config.SIMILARITY_THRESHOLD}) ")
    print("=" * 70)

    # 1. Establish connections
    embedder = OllamaEmbedder()
    vector_db = LocalVectorDB()

    total_records = vector_db.count()
    print(f"\n[+] Total Records in Vector DB: {total_records}")

    if total_records == 0:
        print("\n[!] Vector database is empty! Please ingest articles first (e.g., python ingest.py).")
        return

    # 2. Command-line query argument if provided
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        search_query(user_query, embedder, vector_db)
        return

    # 3. Default sample queries
    test_queries = [
        "Diyabet hastalığının belirtileri ve tedavisi nedir?",
        "siber güvenlik"
    ]

    print("\n--- Running Sample Test Queries ---")
    for q in test_queries:
        search_query(q, embedder, vector_db)

if __name__ == "__main__":
    main()
