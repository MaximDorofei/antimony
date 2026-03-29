from __future__ import annotations
import re
from core.model_manager  import models as _models
from core.memoria        import EnergiaMemory
from core.safety         import sanitise
from rag.hybrid_rag      import HybridRAG
from agents.rag_agent    import RAGAgent
from agents.research_agent  import ResearchAgent
from agents.education_agent import EducationAgent
 
_INTENTS = {
    'research':  r'search|find|look up|what is|who is|when did|latest|news|discover|investigate',
    'education': r'test|quiz|exam|study|learn|textbook|homework|exercise|exercitiu|задание|тест',
    'rag':       r'according to|from the doc|in the book|based on|explain from',
}
_COMPILED_INTENTS = {k: re.compile(v, re.IGNORECASE)
                    for k, v in _INTENTS.items()}
 
class Orchestrator:
    def __init__(self):
        self.memory = EnergiaMemory()
        self.rag    = HybridRAG()
        self.rag.ingest_education_folder()
 
        self._agents = {
            'rag':       RAGAgent(_models, self.memory, self.rag),
            'research':  ResearchAgent(_models, self.memory, self.rag),
            'education': EducationAgent(_models, self.memory, self.rag),
        }
 
    def classify_intent(self, query: str) -> str:
        """Classify intent using pattern matching first, light model fallback."""
        for intent, pattern in _COMPILED_INTENTS.items():
            if pattern.search(query):
                return intent
        # Light model fallback — single-word classification
        prompt = (
            'Classify this query into exactly one word: research, education, or chat.\n'
            f'Query: {query[:200]}\nAnswer:'
        )
        label = _models.complete_light(prompt, max_tokens=5).strip().lower()
        if 'research'  in label: return 'research'
        if 'education' in label: return 'education'
        return 'rag'
 
    def run(self, raw_query: str) -> dict:
        query, warnings = sanitise(raw_query)
        if warnings:
            return {
                'response':       'Input was modified due to safety concerns.',
                'warnings':       warnings,
                'intent':         'blocked',
            }
 
        intent = self.classify_intent(query)
        result = self._agents[intent].run(query)
        result['intent'] = intent
        result['query']  = query
 
        # Persist to memory
        self.memory.add('user',      query)
        self.memory.add('assistant', result.get('response', ''))
 
        return result
