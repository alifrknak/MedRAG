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
    Local ChromaDB Manager for storing and querying medical article chunks.
    Required Schema (Nullable Supported):
    - url (datasourceURL -> metadata)
    - chunk_text (Document content -> document)
    - chunk_vector (Float list / vector -> embedding)
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
            f"ChromaDB collection ready: '{self.collection_name}' "
            f"(Current record count: {self.collection.count()})"
        )

    def add_chunk(
        self,
        chunk_text: str,
        chunk_vector: List[float],
        url: Optional[str] = None,
        chunk_id: Optional[str] = None
    ) -> str:
        """Inserts a single chunk into ChromaDB."""
        cid = chunk_id or str(uuid.uuid4())
        metadata = {"url": url if url is not None else ""}

        self.collection.add(
            ids=[cid],
            embeddings=[chunk_vector],
            documents=[chunk_text],
            metadatas=[metadata]
        )
        logger.info(f"Chunk inserted successfully ID: {cid}")
        return cid

    def add_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Inserts a batch of chunks into ChromaDB.
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
        logger.info(f"Batch inserted {len(chunks)} chunks into ChromaDB.")
        return ids

    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        similarity_threshold: Optional[float] = config.SIMILARITY_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Executes Cosine Similarity search against chunk vectors.
        Filters out results below similarity_threshold if specified (e.g. 0.48).
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
            similarity_score = round(float(1.0 - distance), 4)
            
            # Similarity threshold filtering
            if similarity_threshold is not None and similarity_score < similarity_threshold:
                continue

            url_val = metadatas[i].get("url", "")
            
            matches.append({
                "chunk_id": ids[i],
                "url": url_val if url_val != "" else None,
                "chunk_text": documents[i],
                "chunk_vector": embeddings[i],
                "similarity_score": similarity_score,
                "distance": round(float(distance), 4)
            })

        return matches

    def count(self) -> int:
        """Returns total record count in ChromaDB collection."""
        return self.collection.count()

    def reset(self):
        """Clears and recreates the ChromaDB collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{self.collection_name}' reset successfully.")
        except Exception as e:
            logger.warning(f"Reset collection warning: {e}")
