from __future__ import annotations
import json, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import chromadb
import config
from core.model_manager import models
 
ACTIVE_WINDOW   = 6    # turns kept verbatim
COMPRESS_AFTER  = 4    # turns to compress when window overflows
ARCHIVAL_THRESH = 0.72 # cosine similarity gate for archival recall
 
@dataclass
class Turn:
    role:    str
    content: str
    ts:      float = field(default_factory=time.time)
 
class EnergiaMemory:
    def __init__(self, session_id: str = 'default'):
        self.session_id   = session_id
        self._active:     list[Turn] = []
        self._summaries:  list[str]  = []
        self._chroma      = chromadb.PersistentClient(str(config.CHROMA_PATH))
        self._collection  = self._chroma.get_or_create_collection('antimony_archival')

    def add(self, role: str, content: str) -> None:
        self._active.append(Turn(role, content))
        if len(self._active) > ACTIVE_WINDOW:
            self._compress()
 
    def _compress(self) -> None:
        to_compress = self._active[:COMPRESS_AFTER]
        self._active = self._active[COMPRESS_AFTER:]
 
        turns_text = '\n'.join(
            f'{t.role.upper()}: {t.content[:600]}' for t in to_compress
        )
        prompt = (
            f'Summarise the following conversation turns concisely in under 200 words.\n'
            f'Preserve named entities, decisions, and open questions.\n\n'
            f'{turns_text}\n\nSUMMARY:'
        )
        summary = models.complete_light(prompt, max_tokens=220)
        self._summaries.append(summary)
        vec = models.encode([summary])[0]
        doc_id = f'{self.session_id}_{int(time.time())}_{len(self._summaries)}'
        self._collection.add(
            documents=[summary],
            embeddings=[vec],
            ids=[doc_id],
            metadatas=[{'session': self.session_id, 'ts': time.time()}],
        )

    def recall(self, query: str, top_k: int = 2) -> list[str]:
        if self._collection.count() == 0:
            return []
        vec = models.encode([query])[0]
        results = self._collection.query(
            query_embeddings=[vec], n_results=min(top_k, self._collection.count()),
            include=['documents', 'distances'],
        )
        docs, dists = results['documents'][0], results['distances'][0]
        return [d for d, dist in zip(docs, dists) if (1 - dist / 2) >= ARCHIVAL_THRESH]
 
    def build_context(self, query: str) -> str:
        parts = []
        if self._summaries:
            parts.append('<memory_summary>\n'
                         + ' | '.join(self._summaries[-3:]) + '\n</memory_summary>')
        archival = self.recall(query)
        if archival:
            parts.append('<archival_memory>\n'
                         + '\n---\n'.join(archival) + '\n</archival_memory>')
        for turn in self._active:
            parts.append(f'{turn.role.upper()}: {turn.content}')
 
        return '\n\n'.join(parts)
 
    def clear_session(self) -> None:
        self._active.clear()
        self._summaries.clear()
