from __future__ import annotations
import re
from agents.base_agent  import BaseAgent
from tools.test_creator import create_test, detect_language
from tools.test_solver  import solve_test
from tools.test_taker   import run_test_session
 
_CREATE = re.compile(r'create|make|generate|write|creat|genereaz',
                     re.IGNORECASE)
_SOLVE  = re.compile(r'solve|answer|rezolv|реши',  re.IGNORECASE)
_TAKE   = re.compile(r'take|start|begin|start.*test|ia.*test|начать',
                     re.IGNORECASE)
 
class EducationAgent(BaseAgent):
    def run(self, query: str, context: str = '') -> dict:
        lang = detect_language(query)
 
        if _CREATE.search(query):
            # Extract topic after 'about/despre/о'
            m = re.search(r'(?:about|despre|о|on)\s+(.+)', query, re.IGNORECASE)
            topic = m.group(1).strip() if m else query
            test  = create_test(topic, rag=self.rag, lang=lang)
            return {'response': test, 'tool': 'test_creator'}
 
        if _SOLVE.search(query):
            # Assume test content follows the keyword
            test_body = re.sub(r'^.*(solve|answer|rezolv|реши)\s*:?\s*',
                               '', query, flags=re.IGNORECASE)
            solution  = solve_test(test_body or query, rag=self.rag)
            return {'response': solution, 'tool': 'test_solver'}
 
        if _TAKE.search(query):
            # We need a test to run — retrieve from RAG or create one
            m = re.search(r'on\s+(.+)', query, re.IGNORECASE)
            topic = m.group(1).strip() if m else 'general knowledge'
            test  = create_test(topic, n_questions=5, rag=self.rag, lang=lang)
            result = run_test_session(test)
            return {'response': result.get('raw_feedback', ''),
                    'tool': 'test_taker'}
 
        # Fallback — general educational Q&A via RAG
        chunks = self.rag.query(query, top_k=5)
        ctx    = self.rag.build_context_string(chunks)
        prompt = self._build_prompt(
            'You are a patient educational assistant. '
            'Explain clearly and use examples appropriate to the student level.',
            ctx, query
        )
        from core.safety import complete_medium_with_confidence
        result = complete_medium_with_confidence(prompt, self.models)
        return {'response': result['text'], 'tool': 'edu_qa',
                'confidence': result['confidence']}
