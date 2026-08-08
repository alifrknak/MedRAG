# 🏥 MedRAG: Yerel Ollama (`embeddinggemma:300m`) & ChromaDB İle Tıbbi Vektör Arama & RAG Sistemi

Bu proje, Türkçe tıbbi makaleleri **Semantic Chunking (Anlamsal Parçalama)** yöntemiyle işleyen, yerel **Ollama (`embeddinggemma:300m`)** modeli ile 768 boyutlu vektörlere dönüştüren ve **ChromaDB** vektör veritabanında saklayarak yüksek hassasiyetli arama ve filtreleme gerçekleştiren modüler bir RAG (Retrieval-Augmented Generation) altyapısıdır.

---

## 📌 İçindekiler
- [1. Proje Mimarisi & Bileşenler](#1-proje-mimarisi--bileşenler)
- [2. Chunking Stratejisi (Semantic Chunking)](#2-chunking-stratejisi-semantic-chunking)
- [3. Embedding Modeli (`embeddinggemma:300m`)](#3-embedding-modeli-embeddinggemma300m)
- [4. Eşik Değeri (Similarity Threshold) Analizi](#4-eşik-değeri-similarity-threshold-analizi)
- [5. Mimarisel Kararlar ve Trade-off Analizi](#5-mimarisel-kararlar-ve-trade-off-analizi)
- [6. Veri Seti Atıfı (Citation)](#6-veri-seti-atıfı-citation)
- [7. Kurulum ve Kullanım Rehberi](#7-kurulum-ve-kullanım-rehberi)

---

## 1. Proje Mimarisi & Bileşenler

```text
MedRAG/
├── config.py                     # Sistem konfigürasyonları ve hiper-parametreler
├── ollama_embedder.py           # Yerel Ollama REST API entegrasyonu & Mini-batching
├── semantic_chunker.py          # Anlamsal konu değişimine (Cosine Distance) göre parçalama
├── vector_db.py                 # ChromaDB şema yönetimi (url, chunk_text, chunk_vector)
├── ingest.py                    # HF Dataset yükleme, chunking ve veritabanı kayıt boru hattı
├── main.py                      # Sorgulama ve Eşik Değeri (Threshold) filtreleme servisi
├── view_db.py                   # Veritabanı içerik izleyici
├── benchmark_threshold.py       # Eşik değeri kalibrasyon ve simülasyon scripti
├── threshold_calibration_report.md  # Empirik benchmark raporu
└── requirements.txt             # Bağımlılıklar
```

---

## 2. Chunking Stratejisi (Semantic Chunking)

### Hangi Yöntem Kullanıldı?
Projede **Semantic Chunking (Anlamsal Parçalama)** algoritması tercih edilmiştir.

### Nasıl Çalışır?
1. **Cümle Ayıştırma:** Makale metni nokta, soru işareti, ünlem ve paragraf sınırlarına göre cümlelere ayrılır.
2. **Toplu Vektör Çıkarımı:** Tüm cümlelerin 768 boyutlu vektörleri Ollama API'si üzerinden mini-batch'ler halinde alınır.
3. **Anlamsal Mesafe Hesabı:** Ardışık cümleler arasındaki Cosine Mesafesi ($d_i = 1.0 - \text{CosineSimilarity}(v_i, v_{i+1})$) hesaplanır.
4. **Kırılma Noktaları (Breakpoint Detection):** Mesafenin belirlenen persentil eşiğini (`SEMANTIC_THRESHOLD_PERCENTILE = 85`) aştığı noktalar **konu değişimi** olarak kabul edilir ve yeni bir chunk başlatılır.

### Neden Seçildi? (Rasyonel)
- **Sabit Boyutlu (Fixed-Size) Parçalama:** Cümleleri ortasından bölebilir ve anlam kaybına sebep olur.
- **Yinelemeli (Recursive) Parçalama:** Cümle bütünlüğünü korur ancak konunun nerede bittiğini tahmin edemez.
- **Semantic Chunking:** Makalenin sadece karakter uzunluğuna göre değil, **anlamsal konu bütünlüğüne** göre bölünmesini sağlar. Böylece bir chunk içinde yalnızca tek bir odak konusu bulunur ve vektör aramalarında doğruluk oranı artar.

---

## 3. Embedding Modeli (`embeddinggemma:300m`)

### Model Özellikleri
- **Model Adı:** `embeddinggemma:300m` (Google Gemma Mimarisi)
- **Vektör Boyutu (Dimension):** `768`
- **Çalışma Biçimi:** Yerel Ollama Servisi (`http://localhost:11434/api/embed`)
- **Model Boyutu:** ~621 MB

### Neden Seçildi?
1. **Yüksek Türkçe ve Çok Dilli Başarım:** Gemma mimarisi Türkçe anlamsal temsilde ve tıbbi terminolojide yüksek doğruluk sunar.
2. **Hafiflik ve Hız:** 300M parametre ölçeği ve 621MB boyutu ile GPU/CPU belleğini yormadan düşük gecikme (low latency) ile çıkarım (inference) yapar.
3. **Yerel Veri Gizliliği:** Tüm veriler yerel makinede Ollama üzerinde işlenir, hiçbir tıbbi veri dış API'lere gönderilmez.

---

## 4. Eşik Değeri (Similarity Threshold) Analizi

Sistemde arama yapıldığında alakasız sorulara cevap verilmesini önlemek ve sadece yüksek benzerlikteki dokümanları döndürmek amacıyla **Similarity Thresholding** mekanizması kurulmuştur.

### Belirlenen Eşik Değeri: `SIMILARITY_THRESHOLD = 0.48`

### Eşik Değeri Kalibrasyon Özeti:
20 Pozitif (Alakalı) ve 10 Negatif (Alakasız) sorgu ile yapılan kalibrasyon benchmarkında:
- **Alakalı Sorguların Ortalama Skoru:** `0.6365` (Minimum: `0.4465`, Maksimum: `0.8790`)
- **Alakasız Sorguların Ortalama Skoru:** `0.3720` (Minimum: `0.2320`, Maksimum: `0.4901`)
- **Ortaya Çıkan Güvenlik Aralığı:** `0.48` eşik değeri tercih edildiğinde alakasız sorguların (*"java da bug nedir?", "siber güvenlik", "hava kaç derece?"*) tamamı başarıyla engellenmekte, alakalı tıbbi sorular ise yüksek doğrulukla yakalanmaktadır.

> 📖 **Detaylı Rapor Atıfı:** Eşik değeri kalibrasyonunun empirik verileri, histogram analizleri ve doğruluk matrisleri **[threshold_calibration_report.md](file:///c:/Users/90535/source/magibu/MedRAG/threshold_calibration_report.md)** dosyasında detaylandırılmıştır.

---

## 5. Mimarisel Kararlar ve Trade-off Analizi

| Mimarisel Karar | Seçilen Yaklaşım | Alternatif | Trade-off / Neden Seçildi? |
| :--- | :--- | :--- | :--- |
| **Vektör Veritabanı** | **ChromaDB (Local)** | FAISS / Qdrant | **Trade-off:** FAISS daha hızlı olabilir ancak metin ve metadata saklamaz. ChromaDB metin (`chunk_text`), URL ve 768d vektörü yerel SQLite/HNSW yapısında sunucusuz sakladığı için seçildi. |
| **Parçalama (Chunking)** | **Semantic Chunking** | Fixed-Size / Recursive | **Trade-off:** Semantik parçalama cümle bazlı ek embedding çağrısı gerektirdiği için daha yavaştır. Ancak arama doğruluğunu ve konu bütünlüğünü maksimuma çıkardığı için tercih edildi. |
| **Çıkarım Mimarisi** | **Ollama REST API** | HuggingFace PyTorch | **Trade-off:** PyTorch modelini doğrudan Python'a yüklemek RAM tüketimini artırır. Ollama background servisi bellek yönetimini ve C++ çıkarım hızını optimize eder. |
| **Batch İşleme** | **Mini-Batching (size=32)** | Tekil / Dev Batch | **Trade-off:** 200+ cümlelik makalelerde tek dev istek atmak HTTP timeout hatasına sebep oluyordu. 32'lik mini-batching ile hem bellek korunmuş hem de timeout'lar engellenmiştir. |

---

## 6. Veri Seti Atıfı (Citation)

Bu projede kullanılan Türkçe hastane ve tıbbi makale verileri Hugging Face üzerindeki açık kaynaklı veri setinden sağlanmıştır:

> **Veri Seti:** [`alibayram/turkish-hospital-medical-articles`](https://huggingface.co/datasets/alibayram/turkish-hospital-medical-articles)  
> **Açıklama:** Türkiye'deki önde gelen hastanelerin (Acıbadem, Memorial, Medicana, Medipol vb.) web kaynaklarından derlenmiş Türkçe tıbbi makaleleri içerir.

```bibtex
@misc{alibayram2024turkishhospital,
  author = {Ali Bayram},
  title = {Turkish Hospital Medical Articles Dataset},
  year = {2024},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/alibayram/turkish-hospital-medical-articles}}
}
```

---

## 7. Kurulum ve Kullanım Rehberi

### Gerekli Bağımlılıkların Kurulması
```bash
pip install -r requirements.txt
```

### Ollama Modelinin Hazırlanması
```bash
ollama pull embeddinggemma:300m
```

### Veri Yükleme (Ingestion)
Hugging Face üzerinden 20 (veya istediğiniz sayıda) makaleyi çekip parçalayarak ChromaDB'ye yüklemek için:
```bash
python ingest.py
```

### Sorgulama Yapma (Search)
```bash
# Özel sorgu çalıştırma:
python main.py "Diyabet hastalığının belirtileri ve tedavisi nedir?"

# Alakasız bir sorgu denemesi (Eşik değeri engeller):
python main.py "hava kaç derece?"
```

### Veritabanı Durumunu İzleme
```bash
python view_db.py
```
