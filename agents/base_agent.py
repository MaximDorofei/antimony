from __future__ import annotations
from abc import ABC, abstractmethod
from core.memoria import EnergiaMemory
from core.model_manager import ModelManager
from rag.hybrid_rag import HybridRAG
 
class BaseAgent(ABC):
    def __init__(
        self,
        models: ModelManager,
        memory: EnergiaMemory,
        rag:    HybridRAG,
    ):
        self.models = models
        self.memory = memory
        self.rag    = rag
 
    @abstractmethod
    def run(self, query: str, context: str = '') -> dict:
        """
        Execute the agent task.
        Returns: {response: str, confidence?: float, sources?: list[str]}
        """
 
    def _build_prompt(self, system: str, context: str, query: str) -> str:
        mem = self.memory.build_context(query)
        return (
            f'<|im_start|>system\n{system}<|im_end|>\n'
            f'<|im_start|>user\n'
            + (f'{mem}\n\n' if mem else '')
            + (f'CONTEXT:\n{context}\n\n' if context else '')
            + f'{query}<|im_end|>\n<|im_start|>assistant\n'
        )
