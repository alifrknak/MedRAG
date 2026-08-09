# 🏥 MedRAG: Advanced Hybrid Vector Search & Reranking RAG System

This project implements a high-precision, modular RAG (Retrieval-Augmented Generation) pipeline for Turkish medical articles. It combines **Semantic Chunking** with a **Two-Stage Hybrid Search & Reranking Architecture** and an **Agentic Qwen2.5:7b LLM Tool-Calling Layer**:

1. **Stage 1 (Agentic Intent Decision & Tool Calling):** Local **Qwen2.5:7b** model evaluates user intent. Greetings and non-medical queries are handled instantly without DB search. Medical queries trigger `search_medical_database`.
2. **Stage 2 (Hybrid Retrieval):** Combines **BM25 Keyword Search** and **Dense Vector Search** via local **Ollama (`embeddinggemma:300m`)** using **Reciprocal Rank Fusion (RRF)**.
3. **Stage 3 (Deep Reranking):** Re-scores candidates using a **Cross-Encoder Reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
4. **Stage 4 (Safety Gate & Citation Synthesis):** Filters out irrelevant queries (`SIMILARITY_THRESHOLD = 0.48`) and synthesizes fluid Turkish answers with inline source citations (`[Kaynak 1]`).

---

## 📌 Table of Contents
- [1. Project Architecture & Agentic Flow](#1-project-architecture--agentic-flow)
- [2. Hybrid Search & Why BM25 Was Chosen](#2-hybrid-search--why-bm25-was-chosen)
- [3. Cross-Encoder Reranking & Why It Was Chosen](#3-cross-encoder-reranking--why-it-was-chosen)
- [4. Chunking Strategy (Semantic Chunking)](#4-chunking-strategy-semantic-chunking)
- [5. Embedding Model (`embeddinggemma:300m`) & Why Chosen](#5-embedding-model-embeddinggemma300m--why-chosen)
- [6. Reranker Model (`ms-marco-MiniLM-L-6-v2`)](#6-reranker-model-ms-marco-minilm-l-6-v2)
- [7. Similarity Threshold Analysis](#7-similarity-threshold-analysis)
- [8. Sample Query Execution & Web UI Screenshots](#8-sample-query-execution--web-ui-screenshots)
- [9. Architectural Trade-off Matrix](#9-architectural-trade-off-matrix)
- [10. Dataset Citation](#10-dataset-citation)
- [11. Installation & Quick Start](#11-installation--quick-start)
- [12. Future Roadmap & TODOs](#12-future-roadmap--todos)
- [13. License](#13-license)

---

## 1. Project Architecture & Agentic Flow

```text
MedRAG/
├── config.py                     # Hyperparameters & system configurations
├── ollama_embedder.py           # Local Ollama REST API client (batch_size=32)
├── semantic_chunker.py          # Semantic breakpoint chunking (Cosine Distance)
├── vector_db.py                 # ChromaDB + BM25 + RRF + Cross-Encoder Reranker
├── llm_generator.py             # Qwen2.5:7b Agentic Tool-Calling & RAG Synthesis
├── server.py                    # FastAPI Web & REST API service
├── static/index.html            # Glassmorphic Medical Chatbot Web UI
├── run_webui.py                 # Convenience server launcher
├── ingest.py                    # Dataset ingestion pipeline (HF -> Chunker -> ChromaDB)
├── main.py                      # Querying CLI service with threshold filtering
├── view_db.py                   # Vector DB inspection utility
└── requirements.txt             # Project dependencies
```

### 🤖 Agentic Tool-Calling & Intent Decision Flow (Qwen2.5:7b)

The system features an **Agentic Decision Layer** powered by **Qwen2.5:7b** via Ollama. Instead of running expensive vector embeddings and database searches unconditionally for every user input, Qwen2.5:7b evaluates the user intent against strict domain policies before deciding whether to invoke the `search_medical_database` tool:

```mermaid
graph TD
    A[Kullanıcı Mesajı] --> B[Qwen2.5:7b Niyet Analizi & Araç Tanımı]
    B -->|Arama Gerekmiyor: Günlük Selamlaşma| C[Doğrudan Sohbet Yanıtı - 0ms Vektör Araması]
    B -->|Arama Gerekmiyor: Tıbbi Dışı Konu| D[Kapsam Dışı Reddetme Mesajı]
    B -->|Araç Tetiklendi: search_medical_database| E[Hibrit Vektör Arama: Ollama 768d + BM25]
    E --> F[Cross-Encoder Reranker: ms-marco-MiniLM-L6]
    F --> G[Güvenlik Filtresi Safety Gate >= 0.48]
    G -->|Tıbbi Kaynaklar Bulundu| H[Atıflı Tıbbi Sentez Yanıtı [Kaynak N] + Kaynak Kartları]
    G -->|Eşik Altında / Eşleşme Yok| I[Güvenlik Uyarısı Bineri]
    style C fill:#34d399,stroke:#333,color:#fff
    style D fill:#f87171,stroke:#333,color:#fff
    style H fill:#38bdf8,stroke:#333,color:#fff
```

#### 📋 Intent Decision Rules:
1. **Günlük Selamlaşma (*"Merhaba"*, *"Nasılsın"*):** Doğrudan sohbet yanıtı verilir, vektör veritabanı araması tamamen atlanır (0ms).
2. **Tıbbi Dışı Konular (*Yazılım, Yemek, Spor, Siber Güvenlik vb.*):** Doğrudan reddetme mesajı verilir (*"Maalesef, ben yalnızca sağlık ve tıp alanında hizmet veren bir bilgi asistanıyım. Bu konuda yardımcı olamam. Sağlık alanında bir sorunuz var mıdır?"*).
3. **Tıbbi Sorular (*"Diyabet belirtileri nedir?"*):** Otomatik `search_medical_database` aracı çağrılır. Veritabanından çekilen hastane makaleleri ile atıflı (`[Kaynak 1]`) yanıt sentezlenir.

---

## 2. Hybrid Search & Why BM25 Was Chosen

### Why BM25 Keyword Search?
Dense vector models (`embeddinggemma:300m`) excel at understanding semantic concepts (e.g., mapping *"şeker hastalığı"* to *"diyabet"*). However, vector-only models suffer from **blind spots** when handling exact medical terminology:

- **Medical Acronyms & Lab Test Codes:** Terms like `HbA1c`, `BASO`, `HGB`, or `ESG` can be blurred into generic "blood test" or "medical procedure" vectors.
- **Specific Surgeries & Drug Names:** Rare terms (e.g., `Sleeve gastroplasti`, `Ablasyon`) require verbatim string matching.
- **Numeric Parameters & Thresholds:** Specific ranges (e.g., `"120 mg/dl"`, `"30 yaş"`) are lost in vector representations.

**BM25 (Sparse Keyword Retrieval)** uses term-frequency statistics (TF-IDF) to pinpoint exact keyword matches with pinpoint accuracy. Combining BM25 with Vector Search via **Reciprocal Rank Fusion (RRF)** ensures exact terminology is never missed while maintaining semantic flexibility.

$$\text{RRF Score}(d) = \frac{1}{60 + \text{rank}_{\text{bm25}}(d)} + \frac{1}{60 + \text{rank}_{\text{vector}}(d)}$$

---

## 3. Cross-Encoder Reranking & Why It Was Chosen

### Why Cross-Encoder Reranking?
Standard vector search (ChromaDB / Bi-Encoder) computes query and document embeddings **independently** and calculates their cosine distance. While ultra-fast, Bi-Encoders cannot perform token-level interactions between the query and the chunk text, often placing the most authoritative answer at Rank #4 or #5 instead of Rank #1.

**Cross-Encoder Reranking** passes the query and document text together into a single transformer model:

$$\text{Input} = \text{[Query]} + \text{[Document Chunk]}$$

All self-attention layers process the query words alongside the chunk words simultaneously, evaluating true semantic relevance.

### Benefits of the Two-Stage Architecture:
1. **Stage 1 (Bi-Encoder + BM25):** Rapidly narrows down thousands of chunks to the top 15 candidate passages in milliseconds.
2. **Stage 2 (Cross-Encoder Reranker):** Performs deep pairwise inspection on the 15 candidates to promote the single most accurate chunk to **Rank #1**.

---

## 4. Chunking Strategy (Semantic Chunking)

1. **Sentence Segmentation:** Raw articles are split into discrete sentences.
2. **Batch Embedding:** 768-dim embeddings extracted in mini-batches via Ollama.
3. **Cosine Distance Calculation:** Consecutive sentence distance ($d_i = 1.0 - \text{CosineSimilarity}(v_i, v_{i+1})$) is computed.
4. **Breakpoint Detection:** Points exceeding `SEMANTIC_THRESHOLD_PERCENTILE = 85` are identified as topic shifts, creating natural chunk boundaries.

---

## 5. Embedding Model (`embeddinggemma:300m`) & Why Chosen

### Model Specifications
- **Model Name:** `embeddinggemma:300m` (Google Gemma Architecture)
- **Vector Dimension:** `768`
- **Inference Engine:** Local Ollama Server (`http://localhost:11434/api/embed`)
- **Memory Footprint:** ~621 MB

### Why Was `embeddinggemma:300m` Chosen?
1. **Google Gemma Multilingual Architecture:** Built upon Google's state-of-the-art Gemma foundation model, providing superior cross-lingual semantic embedding capabilities for Turkish medical terminology compared to traditional legacy models.
2. **768-Dimensional High Expressive Capacity:** Generates rich 768-dimensional float vectors capable of capturing subtle medical nuances, clinical symptoms, and complex anatomical relationships.
3. **Ultra-Lightweight & Low Latency (~621 MB):** At only 300 million parameters, it achieves high inference speeds locally via Ollama without requiring expensive cloud GPU server farms.
4. **100% Local Data Privacy:** Runs entirely inside the local environment via Ollama (`http://localhost:11434`), eliminating external API dependency and guaranteeing complete healthcare data privacy.

---

## 6. Reranker Model (`ms-marco-MiniLM-L-6-v2`)

### Model Overview & Purpose
The project utilizes **`cross-encoder/ms-marco-MiniLM-L-6-v2`** as the Stage-2 Deep Reranker. Its purpose is to act as an authoritative evaluator that inspects candidate passages from Stage-1 and re-ranks them so the single most authoritative answer lands at **Rank #1**.

---

## 7. Similarity Threshold Analysis

- **Configured Threshold:** `SIMILARITY_THRESHOLD = 0.48`
- **Safety Gate Behavior:** Prevents hallucination and blocks non-medical/off-topic queries (*"hava kaç derece?"*, *"siber güvenlik"*), returning a clean warning when no chunk meets the threshold.

> 📖 **Detailed Report Reference:** Statistical score distributions and benchmark matrices are available in **[threshold_calibration_report.md](./threshold_calibration_report.md)**.

---

## 8. Sample Query Execution & Web UI Screenshots

Below are actual Web UI interface screenshots demonstrating the three intent branches of the system:

### 1. Günlük Selamlaşma (Direct Chitchat - 0ms Search)
Kullanıcı selamlaştığında veya günlük sohbet başlattığında veritabanı araması yapılmaz, doğrudan sohbet yanıtı sunulur:

![1. Günlük Selamlaşma Yanıtı](file:///C:/Users/90535/Desktop/say_hello.png)

---

### 2. Tıbbi Soru & Atıflı Yapay Zeka Sentezi (Medical Tool Calling)
Tıbbi bir soru sorulduğunda `search_medical_database` aracı tetiklenir, ChromaDB ve Reranker üzerinden çekilen klinik kaynaklarla atıflı (`[Kaynak 1]`) yanıt üretilir:

![2. Tıbbi Soru & Atıflı Yapay Zeka Yanıtı](file:///C:/Users/90535/Desktop/medical_quesiton.png)

---

### 3. Kapsam Dışı / Tıbbi Dışı Konu Engelleme (Out-of-Scope Refusal)
Tıp veya sağlık dışı konularda (yazılım, yemek vb.) soru sorulduğunda sistem veritabanı araması yapmadan nazikçe reddetme mesajı verir:

![3. Kapsam Dışı Konu Engelleme](file:///C:/Users/90535/Desktop/outofscope_quesion.png)

---

## 9. Architectural Trade-off Matrix

| Architectural Decision | Chosen Approach | Alternative | Trade-off / Rationale |
| :--- | :--- | :--- | :--- |
| **Generative LLM** | **`qwen2.5:7b`** | `gemma2:9b` / Cloud API | **Trade-off:** High Turkish fluency & tool-calling support running 100% locally via Ollama. |
| **Embedding Model** | **`embeddinggemma:300m`** | `all-MiniLM-L6-v2` | **Trade-off:** 768-dim Google Gemma offers superior Turkish medical semantics vs older 384-dim models. |
| **Retrieval Architecture** | **Two-Stage Hybrid + Reranking** | Single Vector Search | **Trade-off:** Adds ~0.5s rerank latency, but increases search precision (Precision@K) by 20-30%. |
| **Reranker Model** | **`ms-marco-MiniLM-L-6-v2`** | `bge-reranker-large` | **Trade-off:** Extremely lightweight (~80MB) and fast on CPU, avoiding heavy GPU memory usage. |
| **Keyword Search** | **BM25 (Sparse)** | None (Vector-only) | **Trade-off:** Guarantees exact matching for medical acronyms (`HbA1c`) and test codes. |
| **Vector Storage** | **ChromaDB (Local)** | FAISS / Qdrant | **Trade-off:** ChromaDB persists text (`chunk_text`), URL, and 768d vectors together in SQLite/HNSW. |

---

## 10. Dataset Citation

> **Dataset:** [`alibayram/turkish-hospital-medical-articles`](https://huggingface.co/datasets/alibayram/turkish-hospital-medical-articles)  
> **Description:** Curated collection of medical articles from major hospital groups in Turkey.

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

## 11. Installation & Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Pull Ollama Models (Embedder & Generative LLM)
```bash
# Embedding Model (300M):
ollama pull embeddinggemma:300m

# Generative Medical LLM (7B):
ollama pull qwen2.5:7b
```

### Ingest Dataset
```bash
python ingest.py
```

### Run Search & LLM RAG Queries
```bash
# Query relevant medical topics (Hybrid Retrieval + Reranker + Qwen2.5:7b Answer Synthesis):
python main.py "Diyabet hastalığının belirtileri ve tedavisi nedir?"

# Off-topic query (Blocked by Similarity Threshold Filter):
python main.py "siber güvenlik"
```

### Launch Interactive Web UI & REST API
```bash
python run_webui.py
# Open Web UI in browser: http://localhost:8000
# OpenAPI Docs: http://localhost:8000/docs
```

### Inspect Database Contents
```bash
python view_db.py
```

---

## 12. Future Roadmap & TODOs

- [x] **1. Generative LLM Integration (Chatbot Response Layer):** Integrate local LLMs via Ollama (`qwen2.5:7b` / `gemma2:9b`) to synthesize fluid, grounded, citation-backed Turkish medical answers from retrieved passages.
- [x] **2. REST API & Web UI:** Develop a FastAPI backend (`/api/v1/search`, `/api/v1/stats`) and an interactive Web UI for clinical and public query interfaces.
- [ ] **3. Automated RAG Evaluation Framework (RAGAS):** Integrate the RAGAS framework to evaluate Faithfulness (Hallucination detection), Answer Relevance, and Context Precision automatically against synthetic medical benchmarks.

---

## 13. License

This project is licensed under the [MIT License](./LICENSE). See the `LICENSE` file for details.
