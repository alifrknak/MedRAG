import sys
import logging
import numpy as np
import config
from ollama_embedder import OllamaEmbedder
from vector_db import LocalVectorDB

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_benchmark():
    print("=" * 75)
    print(" SIMILARITY THRESHOLD EŞİK DEĞERİ BENCHMARK VE KALİBRASYON TESTİ ")
    print("=" * 75)

    embedder = OllamaEmbedder()
    vector_db = LocalVectorDB()

    total_docs = vector_db.count()
    print(f"\n[+] Veritabanındaki Toplam Kayıt (Chunk) Sayısı: {total_docs}")

    if total_docs == 0:
        print("❌ Veritabanı boş! Lütfen önce 'python ingest.py' çalıştırın.")
        return

    # 20 Adet Alakalı (Pozitif) Tıbbi Sorgu
    relevant_queries = [
        "Diyabet hastalığının belirtileri ve tedavisi nedir?",
        "Glike hemoglobin HGB testi ne için yapılır?",
        "Hipoksemi kanda oksijen düşüklüğü belirtileri nelerdir?",
        "Bazofil BASO yüksekliği ve düşüklüğü ne anlama gelir?",
        "Endoskopik sleeve gastroplasti mide küçültme işlemi nasıl uygulanır?",
        "Sleeve gastrektomi ameliyatı kaç saat sürer?",
        "Açık kalp ameliyatı riskleri ve iyileşme süreci",
        "Sigara kullanımı ve stresin hipertansiyon üzerine etkileri",
        "Çocuklarda yüksek ateş durumunda yapılması gerekenler",
        "Çocuk acil servise hangi durumlarda başvurulmalıdır?",
        "Bebeklerde ve çocuklarda burun tıkanıklığı nasıl giderilir?",
        "Solunum sıkıntısı belirtileri ve tedavisi",
        "Ablasyon tedavisi nasıl yapılır ve kalbe etkileri",
        "Adenit lenf bezi iltihabı nedenleri ve belirtileri",
        "Kol ağrısı sebepleri ve hangi doktora gidilmeli?",
        "Kardiyovasküler hastalıkları önlemek için beslenme ve egzersiz",
        "Grip ve soğuk algınlığı arasındaki farklar ve tedavi",
        "Mide bulantısı neden olur ve diyabetik ketoasidoz ilişkisi",
        "Obezite tedavisinde cerrahi olmayan mide dikişleme yöntemi",
        "Çocuk göğüs hastalıkları ve hırıltılı solunum"
    ]

    # 10 Adet Alakasız (Negatif) Sorgu
    irrelevant_queries = [
        "Siber güvenlik saldırılarından korunma yöntemleri nelerdir?",
        "Kripto para ve blokzincir teknolojisi nasıl çalışır?",
        "Araba motorunda yağ değişimi nasıl yapılır?",
        "Python yazılım dilinde döngüler ve fonksiyonlar",
        "Uzay seyahatleri ve Mars keşif araçları",
        "Evde lezzetli napoliten pizza tarifi nasıl hazırlanır?",
        "Futbolda 4-3-3 taktiği ve ofsayt kuralı nedir?",
        "Borsa İstanbul hisse senedi alım satım işlemleri",
        "Fotoğraf makinesinde diyafram ve enstantane ayarı",
        "İkinci el araba alırken dikkat edilmesi gereken ekspertiz noktaları"
    ]

    print(f"\n[1] Pozitif Sorgular Çalıştırılıyor ({len(relevant_queries)} adet)...")
    positive_scores = []
    for idx, q in enumerate(relevant_queries, 1):
        q_vec = embedder.get_embedding(q)
        # Threshold koymadan ham skorları al (top_k=1)
        results = vector_db.search(q_vec, top_k=1, similarity_threshold=None)
        score = results[0]["similarity_score"] if results else 0.0
        positive_scores.append(score)
        print(f"  • P{idx:02d}: '{q[:45]}...' -> Max Skor: {score:.4f}")

    print(f"\n[2] Negatif (Alakasız) Sorgular Çalıştırılıyor ({len(irrelevant_queries)} adet)...")
    negative_scores = []
    for idx, q in enumerate(irrelevant_queries, 1):
        q_vec = embedder.get_embedding(q)
        results = vector_db.search(q_vec, top_k=1, similarity_threshold=None)
        score = results[0]["similarity_score"] if results else 0.0
        negative_scores.append(score)
        print(f"  • N{idx:02d}: '{q[:45]}...' -> Max Skor: {score:.4f}")

    # İstatistiksel Hesaplamalar
    pos_min = float(np.min(positive_scores))
    pos_max = float(np.max(positive_scores))
    pos_mean = float(np.mean(positive_scores))
    pos_median = float(np.median(positive_scores))

    neg_min = float(np.min(negative_scores))
    neg_max = float(np.max(negative_scores))
    neg_mean = float(np.mean(negative_scores))
    neg_median = float(np.median(negative_scores))

    print("\n" + "=" * 75)
    print(" ISTATISTIKSEL BENCHMARK OZETI ")
    print("=" * 75)
    print("📊 ALAKALI (POZITIF) SORGU SKORLARI:")
    print(f"   - Minimum Skor : {pos_min:.4f}")
    print(f"   - Maksimum Skor: {pos_max:.4f}")
    print(f"   - Ortalama Skor: {pos_mean:.4f}")
    print(f"   - Medyan Skor  : {pos_median:.4f}")

    print("\n📊 ALAKASIZ (NEGATIF) SORGU SKORLARI:")
    print(f"   - Minimum Skor : {neg_min:.4f}")
    print(f"   - Maksimum Skor: {neg_max:.4f}")
    print(f"   - Ortalama Skor: {neg_mean:.4f}")
    print(f"   - Medyan Skor  : {neg_median:.4f}")

    # Optimal Eşik Değeri Arayışı (0.20 ile 0.70 arası simülasyon)
    best_threshold = 0.50
    best_accuracy = -1.0
    best_fp = 999
    best_fn = 999

    print("\n" + "-" * 75)
    print(" EŞİK DEĞERİ (THRESHOLD) SİMÜLASYONU VE OPTİMİZASYON ")
    print("-" * 75)
    print(f"{'Eşik (T)':<10} | {'Doğru Kabul (TP)':<16} | {'Yanlış Kabul (FP)':<17} | {'Yanlış Red (FN)':<15} | {'Başarı (Acc)':<12}")
    print("-" * 75)

    candidate_thresholds = np.arange(0.25, 0.65, 0.025)
    for t in candidate_thresholds:
        t = round(float(t), 4)
        tp = sum(1 for s in positive_scores if s >= t)  # Doğru bilinen alakalılar
        fn = sum(1 for s in positive_scores if s < t)   # Kaçırılan alakalılar
        fp = sum(1 for s in negative_scores if s >= t)  # İçeri sızan alakasızlar
        tn = sum(1 for s in negative_scores if s < t)   # Doğru reddedilen alakasızlar

        total = len(positive_scores) + len(negative_scores)
        acc = (tp + tn) / total

        print(f"  {t:<8.3f} | {tp:<16} | {fp:<17} | {fn:<15} | %{acc*100:<10.1f}")

        # Optimum threshold: En yüksek accuracy, en az False Positive
        if acc > best_accuracy or (acc == best_accuracy and fp < best_fp):
            best_accuracy = acc
            best_threshold = t
            best_fp = fp
            best_fn = fn

    print("=" * 75)
    print(f" 🏆 TAVSİYE EDİLEN OPTİMAL EŞİK DEĞERİ (OPTIMAL THRESHOLD): {best_threshold:.3f}")
    print(f" • Genel Doğruluk Oranı (Accuracy) : %{best_accuracy*100:.1f}")
    print(f" • Yanlış Pozitif (FP - Yanlış Kabul): {best_fp} adet")
    print(f" • Yanlış Negatif (FN - Kaçırılan)   : {best_fn} adet")
    print("=" * 75)

    return {
        "best_threshold": best_threshold,
        "best_accuracy": best_accuracy,
        "pos_mean": pos_mean,
        "pos_min": pos_min,
        "neg_mean": neg_mean,
        "neg_max": neg_max,
        "positive_scores": positive_scores,
        "negative_scores": negative_scores
    }

if __name__ == "__main__":
    run_benchmark()
