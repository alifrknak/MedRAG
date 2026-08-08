import os

# Ollama Servisi Konfigürasyonu
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "embeddinggemma:300m")

# ChromaDB Vektör Veritabanı Konfigürasyonu
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "medical_chunks")

# Semantic Chunking (Anlamsal Parçalama) Konfigürasyonu
SEMANTIC_THRESHOLD_PERCENTILE = float(os.getenv("SEMANTIC_THRESHOLD_PERCENTILE", "85"))
DEFAULT_DISTANCE_THRESHOLD = float(os.getenv("DEFAULT_DISTANCE_THRESHOLD", "0.35"))
MIN_CHUNK_CHAR_LEN = int(os.getenv("MIN_CHUNK_CHAR_LEN", "150"))
MAX_CHUNK_CHAR_LEN = int(os.getenv("MAX_CHUNK_CHAR_LEN", "1200"))
