from __future__ import annotations
import threading, logging
from typing import Iterator
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import config
 
log = logging.getLogger('antimony.models')
 
class ModelManager:
    def __init__(self):
        self._lock      = threading.Lock()
        self._light:  Llama | None = None
        self._medium: Llama | None = None
        self._embed:  SentenceTransformer | None = None
 
    def light(self) -> Llama:
        with self._lock:
            if self._light is None:
                log.info('Loading light model...')
                self._light = Llama(
                    model_path=str(config.LIGHT_MODEL),
                    n_ctx=config.LIGHT_CTX,
                    n_gpu_layers=config.N_GPU_LAYERS,
                    verbose=False,
                    logits_all=True,
                )
            return self._light
 
    def medium(self) -> Llama:
        with self._lock:
            if self._medium is None:
                log.info('Loading medium model...')
                self._medium = Llama(
                    model_path=str(config.MEDIUM_MODEL),
                    n_ctx=config.MEDIUM_CTX,
                    n_gpu_layers=config.N_GPU_LAYERS,
                    verbose=False,
                    logits_all=True,
                )
            return self._medium
 
    def embed(self) -> SentenceTransformer:
        with self._lock:
            if self._embed is None:
                log.info('Loading embedding model...')
                self._embed = SentenceTransformer(
                    config.EMBED_MODEL,
                    device='mps' if config.IS_MAC_M1 else 'cpu',
                )
            return self._embed
 
    def complete_light(self, prompt: str, max_tokens: int = 512) -> str:
        resp = self.light()(
            prompt, max_tokens=max_tokens, stop=['</s>', '<|im_end|>'],
            echo=False,
        )
        return resp['choices'][0]['text'].strip()
 
    def complete_medium(
        self, prompt: str, max_tokens: int = 1024,
        stream: bool = False
    ) -> str | Iterator[str]:
        llm = self.medium()
        if stream:
            return (tok['choices'][0]['text']
                    for tok in llm(prompt, max_tokens=max_tokens,
                                  stop=['</s>', '<|im_end|>'],
                                  stream=True))
        resp = llm(prompt, max_tokens=max_tokens,
                   stop=['</s>', '<|im_end|>'], echo=False)
        return resp['choices'][0]['text'].strip()
 
    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.embed().encode(texts, normalize_embeddings=True).tolist()
 
    def unload_medium(self):
        with self._lock:
            self._medium = None
            import gc; gc.collect()
models = ModelManager()
