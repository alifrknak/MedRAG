import logging
from typing import List, Union
import requests
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaEmbedder:
    """
    Yerel Ollama servisi üzerinden metinlerden embedding vektörü üreten sınıf.
    Varsayılan model: embeddinggemma:300m (768-boyutlu vektörler üretir)
    """

    def __init__(
        self,
        ollama_url: str = config.OLLAMA_URL,
        model_name: str = config.MODEL_NAME
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model_name = model_name
        self.embed_endpoint = f"{self.ollama_url}/api/embed"
        self._verify_connection()

    def _verify_connection(self):
        """Ollama servisinin çalışıp çalışmadığını ve modelin varlığını kontrol eder."""
        try:
            res = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if res.status_code == 200:
                models = [m.get("name") for m in res.json().get("models", [])]
                logger.info(f"Ollama servisi aktif. Yüklü modeller: {models}")
                if not any(self.model_name in m for m in models):
                    logger.warning(
                        f"Dikkat: '{self.model_name}' modeli Ollama listesinde görünmüyor. "
                        f"Lütfen 'ollama pull {self.model_name}' komutunu çalıştırdığınızdan emin olun."
                    )
            else:
                logger.warning(f"Ollama bağlantı uyarısı: HTTP Status {res.status_code}")
        except Exception as e:
            logger.error(f"Ollama servisine bağlanılamadı ({self.ollama_url}): {e}")

    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Bir veya birden fazla metin için embedding vektörü listesi üretir.
        """
        if isinstance(texts, str):
            input_texts = [texts]
        else:
            input_texts = texts

        payload = {
            "model": self.model_name,
            "input": input_texts
        }

        try:
            response = requests.post(
                self.embed_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama API hatası (Status {response.status_code}): {response.text}"
                )

            data = response.json()
            embeddings = data.get("embeddings", [])

            if not embeddings:
                raise RuntimeError("Ollama boş embedding yanıtı döndürdü.")

            return embeddings

        except Exception as e:
            logger.error(f"Embedding üretilirken hata oluştu: {e}")
            raise e

    def get_embedding(self, text: str) -> List[float]:
        """Tek bir metin string'i için embedding vektörü döndürür."""
        embeddings = self.get_embeddings([text])
        return embeddings[0]


if __name__ == "__main__":
    embedder = OllamaEmbedder()
    test_text = "Tıbbi makale araması ve yerel vektör saklama"
    vec = embedder.get_embedding(test_text)
    print(f"Model: {embedder.model_name}")
    print(f"Metin: '{test_text}'")
    print(f"Vektör boyutu: {len(vec)}")
    print(f"Vektör ilk 5 elemanı: {vec[:5]}")
