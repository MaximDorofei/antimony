from __future__ import annotations
import re
from pathlib import Path
import config
from rag.bm25_index  import BM25Index
from rag.vector_store import VectorStore
from rag.reranker    import rrf_merge
 
_CHUNK_SIZE    = 400   # words
_CHUNK_OVERLAP = 80
 
def _chunk_text(text: str, source: str) -> list[dict]:
    words  = text.split()
    chunks = []
    start  = 0
    idx    = 0
    while start < len(words):
        end   = min(start + _CHUNK_SIZE, len(words))
        chunk = ' '.join(words[start:end])
        chunks.append({
            'text':     chunk,
            'source':   source,
            'chunk_id': f'{Path(source).stem}_{idx}',
        })
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
        idx   += 1
    return chunks
 
class HybridRAG:
    def __init__(self):
        self.bm25   = BM25Index()
        self.vector = VectorStore()
        self._ingested: set[str] = set()
 
    def ingest_text(self, text: str, source: str) -> None:
        if source in self._ingested:
            return
        chunks = _chunk_text(text, source)
        self.bm25.add_documents(chunks)
        self.vector.add_chunks(chunks)
        self._ingested.add(source)
 
    def ingest_education_folder(self) -> int:
        ingested = 0
        for path in config.EDUCATION_DIR.iterdir():
            if path.suffix.lower() not in config.ALLOWED_EDU_EXTENSIONS:
                continue
            if str(path) in self._ingested:
                continue
            text = _extract_text(path)
            if text.strip():
                self.ingest_text(text, str(path))
                ingested += 1
        return ingested
 
    def query(self, query: str, top_k: int = 6) -> list[dict]:
        bm25_res  = self.bm25.search(query, top_k=top_k * 2)
        dense_res = self.vector.search(query, top_k=top_k * 2)
        return rrf_merge(bm25_res, dense_res, top_k=top_k)
 
    def build_context_string(self, chunks: list[dict]) -> str:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f'[{i}] (source: {Path(c["doc"]["source"]).name})\n'
                         f'{c["doc"]["text"]}')
        return '\n\n'.join(parts)
 
def _extract_text(path: Path) -> str:
    if path.suffix.lower() == '.pdf':
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            return ' '.join(page.extract_text() or '' for page in reader.pages)
        except Exception: return ''
    elif path.suffix.lower() == '.rtf':
        try:
            from striprtf.striprtf import rtf_to_text
            return rtf_to_text(path.read_text(errors='ignore'))
        except Exception: return ''
    return ''
