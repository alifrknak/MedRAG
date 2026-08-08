# Similarity Threshold Kalibrasyon Raporu (Yöntem A)

Bu rapor, yerel **Ollama (`embeddinggemma:300m`)** modeli ve **ChromaDB** vektör veritabanınızdaki **446 adet** tıbbi chunk üzerinde **20 Pozitif (Alakalı)** ve **10 Negatif (Alakasız)** sorgu ile gerçekleştirilen empirik (deneysel) kalibrasyon testinin sonuçlarını içerir.

---

## 📊 1. İstatistiksel Skor Dağılımı

| İstatistik Metric | Alakalı (Pozitif - 20 Sorgu) | Alakasız (Negatif - 10 Sorgu) | Marjin / Fark |
| :--- | :---: | :---: | :---: |
| **Minimum Skor** | **0.4465** | 0.2320 | **+0.2145** |
| **Maksimum Skor** | 0.8790 | **0.4901** | **+0.3889** |
| **Ortalama (Mean)** | **0.6365** | **0.3720** | **+0.2645** |
| **Medyan (Median)** | 0.6151 | 0.3765 | +0.2386 |

### 📌 Kritik Bulgular:
1. En yüksek alakasız sorgu skoru **`0.4901`** (*Siber güvenlik*) çıkmıştır.
2. En düşük alakalı sorgu skoru **`0.4465`** (*Bebeklerde burun tıkanıklığı*) çıkmıştır.
3. Alakalı sorguların ortalama skoru **`0.6365`** olup, alakasız sorguların ortalaması **`0.3720`** seviyesindedir.

---

## 🧪 2. Eşik Değeri (Threshold) Simülasyonu ve Başarı Oranları

| Eşik (Threshold) | Doğru Kabul (TP) | Yanlış Kabul (FP) | Kaçırılan (FN) | Genel Başarı (Accuracy) |
| :---: | :---: | :---: | :---: | :---: |
| **0.300** | 20 | 8 | 0 | %73.3 |
| **0.350** | 20 | 7 | 0 | %76.7 |
| **0.400** | 20 | 3 | 0 | %90.0 |
| **0.425** | 20 | 2 | 0 | **%93.3** |
| **🏆 0.450** | **19** | **1** | **1** | **%93.3** |
| **🏆 0.480** | **18** | **1** | **2** | **%90.0** |
| **0.500** | 16 | 0 | 4 | %86.7 |
| **0.550** | 16 | 0 | 4 | %86.7 |

---

## 🏆 3. Karar ve Optimizasyon

- **`0.500` ve üzeri** değerler, zayıf ifade edilmiş alakalı bazı soruları kaçırmaya başlamaktadır (%86.7 başarı).
- **`0.400` ve altı** değerler, alakasız soruları içeri alma riski taşımaktadır (False Positive: 3+).
- **En Dengeli ve İdeal Eşik Değeri (Optimal Threshold): `0.450 - 0.480`**

### Neden `0.450 - 0.480`?
1. Alakasız soruların %90'ını engeller.
2. Alakalı tıbbi soruların %90-%95'ini başarıyla yakalar.

> `config.py` dosyanızdaki **`SIMILARITY_THRESHOLD = 0.48`** olarak güncellenmiştir.
