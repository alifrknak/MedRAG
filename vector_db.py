import os
import uuid
import logging
from typing import List, Dict, Any, Optional
import chromadb
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalVectorDB:
    """
    Yerel ChromaDB vektör veritabanı yöneticisi.
    İstenen Şema (Nullable Destekli):
    - url (datasourceURL -> metadata)
    - chunk_text (Metin içeriği -> document)
    - chunk_vector (Float liste / vektör -> embedding)
    """

    def __init__(
        self,
        persist_dir: str = config.CHROMA_PATH,
        collection_name: str = config.COLLECTION_NAME
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        os.makedirs(self.persist_dir, exist_ok=True)
        
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            f"ChromaDB koleksiyonu hazır: '{self.collection_name}' "
            f"(Mevcut öge sayısı: {self.collection.count()})"
        )

    def add_chunk(
        self,
        chunk_text: str,
        chunk_vector: List[float],
        url: Optional[str] = None,
        chunk_id: Optional[str] = None
    ) -> str:
        """Tek bir chunk kaydını veritabanına ekler."""
        cid = chunk_id or str(uuid.uuid4())
        metadata = {"url": url if url is not None else ""}

        self.collection.add(
            ids=[cid],
            embeddings=[chunk_vector],
            documents=[chunk_text],
            metadatas=[metadata]
        )
        logger.info(f"Chunk başarıyla eklendi ID: {cid}")
        return cid

    def add_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Toplu halde chunk listesi ekler.
        Her chunk dict öğesi şunları barındırabilir:
        - chunk_text (zorunlu)
        - chunk_vector (zorunlu)
        - url (opsiyonel / nullable)
        - chunk_id (opsiyonel)
        """
        if not chunks:
            return []

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for item in chunks:
            cid = item.get("chunk_id") or str(uuid.uuid4())
            url_val = item.get("url")
            
            ids.append(cid)
            documents.append(item["chunk_text"])
            embeddings.append(item["chunk_vector"])
            metadatas.append({"url": url_val if url_val is not None else ""})

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        logger.info(f"{len(chunks)} adet chunk veritabanına toplu olarak yüklendi.")
        return ids

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Sorgu vektörüne göre Cosine Similarity araması yapar.
        Sonuç olarak url, chunk_text, chunk_vector ve similarity_score döndürür.
        """
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances", "embeddings"]
        )

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        embeddings = results["embeddings"][0] if results.get("embeddings") else [None] * len(ids)

        matches = []
        for i in range(len(ids)):
            distance = distances[i]
            similarity_score = 1.0 - distance
            url_val = metadatas[i].get("url", "")
            
            matches.append({
                "chunk_id": ids[i],
                "url": url_val if url_val != "" else None,
                "chunk_text": documents[i],
                "chunk_vector": embeddings[i],
                "similarity_score": round(float(similarity_score), 4),
                "distance": round(float(distance), 4)
            })

        return matches

    def count(self) -> int:
        """Koleksiyondaki toplam kayıt sayısını döner."""
        return self.collection.count()

    def reset(self):
        """Koleksiyonu sıfırlar ve temizler."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Koleksiyon '{self.collection_name}' sıfırlandı.")
        except Exception as e:
            logger.warning(f"Sıfırlama uyarısı: {e}")
