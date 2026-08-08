import logging
from ollama_embedder import OllamaEmbedder
from vector_db import LocalVectorDB
from semantic_chunker import SemanticChunker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_sample_articles():
    """
    Uzun makaleleri Semantic Chunker ile anlamsal parçalara ayırır
    ve ChromaDB veritabanına kaydeder.
    """
    print("=" * 70)
    print(" Makale Yükleme ve Vektörleştirme (Ingestion) ")
    print("=" * 70)

    embedder = OllamaEmbedder()
    vector_db = LocalVectorDB()
    semantic_chunker = SemanticChunker(embedder=embedder)

    # Örnek Uzun Tıbbi ve Teknolojik Makaleler
    long_article_1 = """
    Diyabet (şeker hastalığı), insülin hormonunun eksikliği veya hücrelerin insüline direnç göstermesi sonucu gelişen kronik bir metabolizma hastalığıdır.
    Kan şekerinin sürekli yüksek seyretmesi böbrekler, gözler ve kardiyovasküler sistem üzerinde kalıcı hasarlara yol açabilir.
    Tip 1 diyabet otoimmün kaynaklıdır ve hastaların ömür boyu insülin tedavisi alması gerekir.
    Tip 2 diyabet ise genellikle obezite, hareketli olmayan yaşam ve beslenme bozuklukları ile ilişkilidir ve yaşam tarzı değişiklikleri ile kontrol altına alınabilir.
    
    Tıbbi biyoteknoloji alanında son yıllarda yaşanan gelişmeler gen tedavileri ve m-RNA aşıları üzerine yoğunlaşmıştır.
    CRISPR-Cas9 gen düzenleme teknolojisi sayesinde kalıtsal hastalıkların kökenindeki mutasyonlar hedeflenebilmektedir.
    Kanser immünoterapisi de hastanın kendi bağışıklık hücrelerini tümörle savaşmak üzere eğitmeyi amaçlamaktadır.
    
    Kalp damar sağlığı için haftada en az 150 dakika orta düzeyde egzersiz önerilmektedir.
    Doymuş yağ oranı yüksek gıdalardan kaçınmak ve Akdeniz tipi beslenmek koroner arter hastalığı riskini %30 oranında düşürmektedir.
    Sigara kullanımı ve kronik stres hipertansiyonu tetikleyen en önemli çevresel faktörler arasındadır.
    """

    long_article_2 = """
    Radyolojide Yapay Zeka Uygulamaları ve Derin Öğrenme Modelleri.
    Bilgisayarlı tomografi (BT) ve manyetik rezonans (MR) görüntülerinin analizinde evrişimli sinir ağları (CNN) insan gözünün kaçırabileceği mikro lezyonları tespit edebilmektedir.
    Özellikle meme kanseri taramalarında mammografi görüntülerinin yapay zeka ile ön değerlendirmeye tabi tutulması yanlış negatif oranlarını düşürmüştür.
    Buna karşın tıbbi verilerin gizliliği ve algoritmaların etik kullanımı klinik entegrasyondaki en büyük engellerdir.
    """

    articles = [
        {"url": "https://saglik.gov.tr/makale/diyabet-ve-saglik", "text": long_article_1},
        {"url": "https://medikalai.org/radyoloji-yapay-zeka", "text": long_article_2}
    ]

    all_chunks = []
    for idx, art in enumerate(articles, 1):
        print(f"\n📄 Makale #{idx} işleniyor (URL: {art['url']})...")
        chunks = semantic_chunker.chunk_document(art["text"], url=art["url"])
        all_chunks.extend(chunks)

    print(f"\n[+] Toplam {len(all_chunks)} parça ChromaDB'ye kaydediliyor...")
    vector_db.add_chunks_batch(all_chunks)
    print(f"✅ Yükleme tamamlandı. Güncel veritabanı kayıt sayısı: {vector_db.count()}")

if __name__ == "__main__":
    ingest_sample_articles()
