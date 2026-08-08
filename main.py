import sys
import logging
import config
from ollama_embedder import OllamaEmbedder
from vector_db import LocalVectorDB

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
    Sorgu metnini Ollama ile embedding'e dönüştürür ve ChromaDB'de arar.
    Belirlenen eşik değerinin (threshold) altındaki sonuçlar filtrelenir.
    """
    print(f"\n🔍 Sorgu: '{query_text}' (Eşik Değeri / Threshold: >= {threshold})")
    query_vector = embedder.get_embedding(query_text)
    results = vector_db.search(query_vector, top_k=top_k, similarity_threshold=threshold)

    if not results:
        print(f"   ⚠️ Eşik değerini ({threshold}) geçen hiçbir alakalı doküman bulunamadı.")
        print("-" * 70)
        return

    print(f"   En Alakalı Sonuçlar (Bulunan: {len(results)} adet):")
    print("-" * 70)
    for idx, res in enumerate(results, 1):
        print(f"   Sonuç #{idx}:")
        print(f"   • Benzerlik Skoru (Similarity) : {res['similarity_score']}")
        print(f"   • Kaynak URL                    : {res['url'] if res['url'] else '(Yok / Null)'}")
        print(f"   • Metin (chunk_text)            : {res['chunk_text']}")
        print("-" * 70)

def main():
    print("=" * 70)
    print(" Vektör Veritabanı Sorgulama Servisi (ChromaDB + Ollama) ")
    print(" Eşik Değeri (Similarity Threshold) Aktif ")
    print("=" * 70)

    # 1. Bağlantıların kurulması
    embedder = OllamaEmbedder()
    vector_db = LocalVectorDB()

    total_records = vector_db.count()
    print(f"\n[+] Veritabanında Mevcut Toplam Kayıt Sayısı: {total_records}")

    if total_records == 0:
        print("\n⚠️ Veritabanı boş! Lütfen önce veritabanına kayıt ekleyin (örn: python ingest.py).")
        return

    # 2. Komut satırından sorgu parametresi verilmişse onu çalıştır
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        search_query(user_query, embedder, vector_db)
        return

    # 3. Varsayılan Örnek Sorgular
    test_queries = [
        "Diyabet hastalığının belirtileri ve tedavisi nedir?",
        "siber güvenlik"
    ]

    print("\n--- Örnek Sorgular Çalıştırılıyor ---")
    for q in test_queries:
        search_query(q, embedder, vector_db)

if __name__ == "__main__":
    main()
