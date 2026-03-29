from __future__ import annotations
import hashlib
import chromadb
import config
from core.model_manager import models
 
class VectorStore:
    def __init__(self, collection_name: str = 'antimony_rag'):
        self._client     = chromadb.PersistentClient(str(config.CHROMA_PATH))
        self._collection = self._client.get_or_create_collection(collection_name)
 
    def add_chunks(self, chunks: list[dict]) -> None:
        """chunks: list of {text, source, chunk_id}"""
        texts = [c['text'] for c in chunks]
        vecs  = models.encode(texts)
        ids   = [c.get('chunk_id',
                        hashlib.md5(c['text'].encode()).hexdigest()[:16])
                 for c in chunks]
        metas = [{'source': c['source']} for c in chunks]
        self._collection.upsert(
            documents=texts, embeddings=vecs, ids=ids, metadatas=metas
        )
 
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self._collection.count() == 0:
            return []
        vec  = models.encode([query])[0]
        res  = self._collection.query(
            query_embeddings=[vec], n_results=min(top_k, self._collection.count()),
            include=['documents', 'metadatas', 'distances'],
        )
        results = []
        for i, (doc, meta, dist) in enumerate(zip(
                res['documents'][0], res['metadatas'][0], res['distances'][0])):
            results.append({
                'doc':         {'text': doc, 'source': meta.get('source', '?')},
                'dense_score': float(1 - dist / 2),
                'rank':        i,
            })
        return results
