import sys
import logging
from ollama_embedder import OllamaEmbedder
from vector_db import LocalVectorDB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def search_query(query_text: str, embedder: OllamaEmbedder, vector_db: LocalVectorDB, top_k: int = 3):
    """
    Sorgu metnini Ollama ile embedding'e dönüştürür ve ChromaDB'de arar.
    """
    print(f"\n🔍 Sorgu: '{query_text}'")
    query_vector = embedder.get_embedding(query_text)
    results = vector_db.search(query_vector, top_k=top_k)

    if not results:
        print("   ❌ Hiçbir eşleşen kayıt bulunamadı.")
        return

    print(f"   En Alakalı Sonuçlar (Top {len(results)}):")
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
        "Kanser tedavisinde kullanılan gen düzenleme teknolojisi nedir?",
        "Mammografi ve MR analizinde yapay zeka nasıl kullanılır?",
        "Hipertansiyon ve kalp sağlığı için beslenme önerileri"
    ]

    print("\n--- Örnek Sorgular Çalıştırılıyor ---")
    for q in test_queries:
        search_query(q, embedder, vector_db)

if __name__ == "__main__":
    main()
