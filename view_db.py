import sys
import logging
import chromadb
import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def view_chroma_contents(
    persist_dir: str = config.CHROMA_PATH,
    collection_name: str = config.COLLECTION_NAME
):
    """
    Yerel ChromaDB veritabanındaki tüm koleksiyonları ve verileri görüntüler.
    """
    print("=" * 70)
    print(f" ChromaDB Veritabanı Görüntüleyici ({persist_dir}) ")
    print("=" * 70)

    client = chromadb.PersistentClient(path=persist_dir)
    collections = client.list_collections()

    print(f"\n[+] Mevcut Koleksiyon Sayısı: {len(collections)}")
    for col in collections:
        print(f" - Koleksiyon Adı: {col.name} | Öge Sayısı: {col.count()}")

    print("\n" + "-" * 70)
    print(f" Target Koleksiyon: '{collection_name}' ")
    print("-" * 70)

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"[-] '{collection_name}' isimli koleksiyon bulunamadı: {e}")
        return

    total_count = collection.count()
    print(f"Toplam Kayıt Sayısı: {total_count}\n")

    if total_count == 0:
        print("Veritabanında henüz kayıt bulunmamaktadır.")
        return

    # Tüm kayıtları getir (documents, metadatas, embeddings)
    all_data = collection.get(include=["documents", "metadatas", "embeddings"])

    ids = all_data.get("ids", [])
    documents = all_data.get("documents", [])
    metadatas = all_data.get("metadatas", [])
    embeddings = all_data.get("embeddings", [])

    for i in range(len(ids)):
        chunk_id = ids[i]
        doc = documents[i] if documents else "N/A"
        meta = metadatas[i] if metadatas else {}
        url = meta.get("url") if meta else None
        vector = embeddings[i] if embeddings is not None and len(embeddings) > i else None
        vec_dim = len(vector) if vector is not None else "Yok"

        print(f"[+] Kayıt #{i+1}")
        print(f"  • ID            : {chunk_id}")
        print(f"  • Kaynak URL    : {url if url else '(Yok / Null)'}")
        print(f"  • Metin (Chunk) : {doc}")
        print(f"  • Vektör Boyutu : {vec_dim}")
        if vector is not None:
            print(f"  • Vektör (İlk 5): {vector[:5]}...")
        print("-" * 70)

if __name__ == "__main__":
    view_chroma_contents()
