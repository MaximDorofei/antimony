from __future__ import annotations
import re, pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
import config
 
_STOPWORDS = {
    'the','a','an','is','are','was','were','in','on','at','to','of',
    'and','or','for','with','by','from','that','this','it','be',
    'si','sau','in','la','pe','de','cu','un','o','cel','cea',
    'i','v','na','po','za','ot','iz', 'kak', 'eto',
}
 
def tokenise(text: str) -> list[str]:
    tokens = re.findall(r'\b[a-zA-ZÀ-ÿА-яёЁ]{2,}\b', text.lower())
    return [t for t in tokens if t not in _STOPWORDS]
 
class BM25Index:
    def __init__(self):
        self._docs:   list[dict] = []
        self._index:  BM25Okapi | None = None
        self._cache   = config.CHROMA_PATH.parent / 'bm25_cache.pkl'
 
    def add_documents(self, docs: list[dict]) -> None:
        self._docs.extend(docs)
        self._index = BM25Okapi([tokenise(d['text']) for d in self._docs])
 
    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if not self._index:
            return []
        tokens = tokenise(query)
        scores = self._index.get_scores(tokens)
        ranked = sorted(zip(scores, range(len(scores))),
                        key=lambda x: x[0], reverse=True)[:top_k]
        return [{'doc': self._docs[i], 'bm25_score': float(s), 'rank': r}
                for r, (s, i) in enumerate(ranked)]
 
    def save(self) -> None:
        with open(self._cache, 'wb') as f:
            pickle.dump({'docs': self._docs}, f)
 
    def load(self) -> bool:
        if not self._cache.exists(): return False
        with open(self._cache, 'rb') as f:
            data = pickle.load(f)
        self.add_documents(data['docs'])
        return True
