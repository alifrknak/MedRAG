import sys
import logging
from typing import List
from datasets import load_dataset, get_dataset_split_names
from ollama_embedder import OllamaEmbedder
from vector_db import LocalVectorDB
from semantic_chunker import SemanticChunker

# Windows konsolunda UTF-8 karakterlerin düzgün yazdırılmasını sağla
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATASET_NAME = "alibayram/turkish-hospital-medical-articles"

def ingest_hf_dataset(limit: int = 100, reset_db: bool = False):
    """
    Hugging Face repo'sundan ('alibayram/turkish-hospital-medical-articles') 
    hastane split'lerini gezerek 100 adet tıbbi makale çeker,
    Semantic Chunker ile parçalar ve ChromaDB veritabanına kaydeder.
    """
    print("=" * 70)
    print(f" Hugging Face Dataset Ingestion: '{DATASET_NAME}' ")
    print("=" * 70)

    # 1. Servislerin Başlatılması
    embedder = OllamaEmbedder(batch_size=32, timeout=120)
    vector_db = LocalVectorDB()
    semantic_chunker = SemanticChunker(embedder=embedder)

    if reset_db:
        print("\n[!] Veritabanı sıfırlanıyor...")
        vector_db.reset()

    # 2. Mevcut Split'leri Alma
    try:
        splits = get_dataset_split_names(DATASET_NAME)
        logger.info(f"Mevcut hastane split'leri: {splits}")
    except Exception as e:
        logger.warning(f"Split listesi alınamadı, varsayılan liste kullanılacak: {e}")
        splits = ['acibadem', 'medicana', 'memorial', 'medipol', 'medicalpark', 'florence']

    processed_articles = 0
    skipped_articles = 0
    failed_articles = 0
    all_chunks_to_insert = []

    print(f"\n[1] Hugging Face üzerinden {limit} makale çekiliyor ve Semantic Chunker ile işleniyor...\n")

    for split_name in splits:
        if processed_articles >= limit:
            break

        logger.info(f"--- '{split_name}' hastane grubu verileri yükleniyor ---")
        try:
            ds = load_dataset(DATASET_NAME, split=split_name, streaming=True)
        except Exception as e:
            logger.warning(f"Split '{split_name}' yüklenemedi: {e}")
            continue

        for item in ds:
            if processed_articles >= limit:
                break

            url = item.get("url", "")
            title = str(item.get("title", "") or "").strip()
            
            # İçerik alanını farklı sütun isimlerini destekleyecek şekilde al (content, article, text)
            raw_content = item.get("content") or item.get("article") or item.get("text") or ""
            article_text = str(raw_content).strip()

            # Boş veya çok kısa metinleri atla
            if not article_text or len(article_text) < 100:
                skipped_articles += 1
                continue

            processed_articles += 1
            full_text = f"{title}\n{article_text}" if title else article_text

            logger.info(f"[{processed_articles}/{limit}] ({split_name}) Makale işleniyor: '{title[:45]}...'")

            try:
                # Semantic Chunking uygula
                chunks = semantic_chunker.chunk_document(full_text, url=url)
                all_chunks_to_insert.extend(chunks)
            except Exception as err:
                logger.error(f"Makale işlenirken hata oluştu ({title[:30]}): {err}")
                failed_articles += 1
                continue

            # Her 25-30 chunk toplandığında veritabanına kaydet
            if len(all_chunks_to_insert) >= 25:
                vector_db.add_chunks_batch(all_chunks_to_insert)
                logger.info(f"   [+] {len(all_chunks_to_insert)} adet chunk ChromaDB'ye aktarıldı.")
                all_chunks_to_insert = []

    # Kalan son chunk'ları veritabanına kaydet
    if all_chunks_to_insert:
        vector_db.add_chunks_batch(all_chunks_to_insert)
        logger.info(f"   [+] Kalan {len(all_chunks_to_insert)} adet chunk ChromaDB'ye aktarıldı.")

    print("\n" + "=" * 70)
    print(" [OK] INGESTION TAMAMLANDI ")
    print("=" * 70)
    print(f" • İşlenen Toplam Makale Sayısı : {processed_articles}")
    print(f" • Başarısız/Hatalı Makaleler   : {failed_articles}")
    print(f" • Atlanan Kısa Makale Sayısı   : {skipped_articles}")
    print(f" • Güncel Veritabanı Kayıt Sayısı: {vector_db.count()}")
    print("=" * 70)

if __name__ == "__main__":
    ingest_hf_dataset(limit=20, reset_db=False)
