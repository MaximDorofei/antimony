from agents.base_agent import BaseAgent
from core.safety import complete_medium_with_confidence
 
_SYSTEM = (
    'You are ANTIMONY, a precise local AI assistant. '
    'Answer using ONLY the provided context. '
    'If the context does not contain enough information, say so clearly. '
    'Cite sources using [N] notation matching the context labels.'
)
 
class RAGAgent(BaseAgent):
    def run(self, query: str, context: str = '') -> dict:
        chunks = self.rag.query(query, top_k=6)
        ctx    = self.rag.build_context_string(chunks)
        prompt = self._build_prompt(_SYSTEM, ctx, query)
        result = complete_medium_with_confidence(prompt, self.models)
        return {
            'response':   result['text'],
            'confidence': result['confidence'],
            'low_conf':   result['low_confidence_flag'],
            'sources':    [c['doc']['source'] for c in chunks],
        }
