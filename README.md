# 🏥 MedRAG: Medical Vector Search & RAG System with Local Ollama (`embeddinggemma:300m`) & ChromaDB

This project implements a modular RAG (Retrieval-Augmented Generation) pipeline for Turkish medical articles. It utilizes **Semantic Chunking** to preserve contextual integrity, generates 768-dimensional embeddings using a local **Ollama (`embeddinggemma:300m`)** model, and persists/queries vectors efficiently with **ChromaDB** using Similarity Threshold filtering.

---

## 📌 Table of Contents
- [1. Project Architecture & Components](#1-project-architecture--components)
- [2. Chunking Strategy (Semantic Chunking)](#2-chunking-strategy-semantic-chunking)
- [3. Embedding Model (`embeddinggemma:300m`)](#3-embedding-model-embeddinggemma300m)
- [4. Similarity Threshold Analysis](#4-similarity-threshold-analysis)
- [5. Architectural Decisions & Trade-off Analysis](#5-architectural-decisions--trade-off-analysis)
- [6. Dataset Citation](#6-dataset-citation)
- [7. Installation & Quick Start](#7-installation--quick-start)
- [8. License](#8-license)

---

## 1. Project Architecture & Components

```text
MedRAG/
├── config.py                     # System configuration & hyperparameters
├── ollama_embedder.py           # Local Ollama REST API client with mini-batching
├── semantic_chunker.py          # Semantic breakpoint chunking (Cosine Distance)
├── vector_db.py                 # ChromaDB schema & vector store manager
├── ingest.py                    # Dataset ingestion pipeline (HF -> Chunker -> ChromaDB)
├── main.py                      # Querying CLI service with threshold filtering
├── view_db.py                   # Vector DB inspection utility
├── benchmark_threshold.py       # Threshold calibration & simulation script
├── threshold_calibration_report.md  # Empirical calibration report
└── requirements.txt             # Project dependencies
```

---

## 2. Chunking Strategy (Semantic Chunking)

### Selected Approach
The project employs **Semantic Chunking** rather than fixed-character or arbitrary recursive splitting.

### How It Works
1. **Sentence Segmentation:** Raw articles are split into discrete sentences based on punctuation and line boundaries.
2. **Batch Embedding:** Sentence embeddings (768-dim) are extracted in mini-batches via Ollama.
3. **Cosine Distance Calculation:** Consecutive sentence distance ($d_i = 1.0 - \text{CosineSimilarity}(v_i, v_{i+1})$) is computed.
4. **Breakpoint Detection:** Points where the semantic distance exceeds the percentile threshold (`SEMANTIC_THRESHOLD_PERCENTILE = 85`) are identified as topic shifts, creating natural chunk boundaries.

### Why Semantic Chunking?
- **Fixed-Size Chunking:** Arbitrarily cuts sentences mid-word/mid-sentence, leading to severe context loss.
- **Recursive Chunking:** Preserves sentences but cannot detect where a topic actually ends.
- **Semantic Chunking:** Ensures every chunk focuses on a **single semantic topic**, maximizing vector retrieval precision.

---

## 3. Embedding Model (`embeddinggemma:300m`)

### Model Specifications
- **Model Name:** `embeddinggemma:300m` (Google Gemma Architecture)
- **Vector Dimension:** `768`
- **Inference Service:** Local Ollama Server (`http://localhost:11434/api/embed`)
- **Memory Footprint:** ~621 MB

### Why `embeddinggemma:300m`?
1. **Multilingual & Medical Performance:** Gemma architecture excels at semantic representations in Turkish medical domain text.
2. **Efficiency & Low Latency:** With 300M parameters (~621MB), it provides fast local inference without heavy GPU memory demands.
3. **Data Privacy:** Runs 100% locally via Ollama, ensuring sensitive medical data never leaves the local environment.

---

## 4. Similarity Threshold Analysis

To prevent hallucination and block irrelevant queries (e.g., non-medical or off-topic questions), a **Similarity Thresholding** filter is enforced during retrieval.

### Configured Threshold: `SIMILARITY_THRESHOLD = 0.48`

### Benchmark Summary:
Conducted on 20 Relevant (Medical) and 10 Irrelevant (Non-medical) queries across 446 chunks:
- **Relevant Query Average Similarity Score:** `0.6365` (Min: `0.4465`, Max: `0.8790`)
- **Irrelevant Query Average Similarity Score:** `0.3720` (Min: `0.2320`, Max: `0.4901`)
- **Safety Margin:** A threshold of `0.48` achieves **90-93.3% accuracy**, completely filtering out off-topic queries (*"java da bug nedir?", "siber güvenlik"*) while retrieving genuine medical queries with high recall.

> 📖 **Detailed Report Reference:** Complete statistical distributions, simulation tables, and calibration metrics are documented in **[threshold_calibration_report.md](./threshold_calibration_report.md)**.

---

## 5. Architectural Decisions & Trade-off Analysis

| Architectural Decision | Chosen Approach | Alternative | Trade-off / Rationale |
| :--- | :--- | :--- | :--- |
| **Vector Storage** | **ChromaDB (Local)** | FAISS / Qdrant | **Trade-off:** FAISS is fast but requires separate metadata storage. ChromaDB persists text (`chunk_text`), URL, and 768d vectors together in a serverless SQLite/HNSW structure. |
| **Chunking Method** | **Semantic Chunking** | Fixed-Size / Recursive | **Trade-off:** Semantic chunking requires sentence-level embedding calls (slightly slower), but guarantees topic coherence and superior search precision. |
| **Inference Engine** | **Ollama REST API** | HuggingFace PyTorch | **Trade-off:** Loading PyTorch models directly consumes Python process RAM. Ollama operates as an optimized C++ background engine. |
| **Request Batching** | **Mini-Batching (size=32)** | Single Huge Payload | **Trade-off:** Large articles (200+ sentences) caused HTTP timeouts when sent in a single payload. Mini-batching (size=32) eliminates timeouts while preserving throughput. |

---

## 6. Dataset Citation

This project utilizes open-source Turkish hospital medical articles sourced from Hugging Face:

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

## 7. Installation & Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Pull Ollama Model
```bash
ollama pull embeddinggemma:300m
```

### Ingest Dataset
To fetch, semantically chunk, and store articles into ChromaDB:
```bash
python ingest.py
```

### Run Search Queries
```bash
# Query relevant medical topics:
python main.py "Diyabet hastalığının belirtileri ve tedavisi nedir?"

# Query off-topic questions (blocked by similarity threshold):
python main.py "hava kaç derece?"
```

### Inspect Database Contents
```bash
python view_db.py
```

---

## 8. License

This project is licensed under the [MIT License](./LICENSE). See the `LICENSE` file for details.
