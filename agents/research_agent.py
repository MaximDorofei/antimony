from __future__ import annotations
import re, time
from duckduckgo_search import DDGS
from agents.base_agent  import BaseAgent
from core.safety        import complete_medium_with_confidence
import config
 
_SYSTEM = (
    'You are ANTIMONY, a research-grade local AI. '
    'Synthesise the web search snippets below into a clear, factual answer. '
    'Distinguish established facts from uncertain claims. '
    'Cite sources as [1], [2], etc. from the provided snippet list.'
)
 
_NEEDS_CONFIRMATION = re.compile(
    r'(plan|strategy|should\s+I|help\s+me|how\s+do\s+I|design|build|create'
    r'|write|make|develop|improve)',
    re.IGNORECASE
)
 
class ResearchAgent(BaseAgent):
 
    def _expand_queries(self, query: str) -> list[str]:
        prompt = (
            f'Rewrite this query in two alternative ways for web search.\n'
            f'Query: {query}\n'
            f'Output exactly two lines, one query per line.'
        )
        raw    = self.models.complete_light(prompt, max_tokens=80)
        lines  = [ln.strip().lstrip('12.-) ') for ln in raw.strip().split('\n')]
        return [query] + [l for l in lines if l][:2]
 
    def _ddg_search(self, query: str) -> list[dict]:
        results = []
        try:
            with DDGS() as ddgs:
                for r in ddgs.text(query,
                                   max_results=config.DDG_MAX_RESULTS):
                    results.append({
                        'title': r.get('title', ''),
                        'body':  r.get('body', ''),
                        'href':  r.get('href', ''),
                    })
        except Exception as e:
            pass
        return results
 
    def _deduplicate(self, snippets: list[dict]) -> list[dict]:
        """Remove near-duplicate snippets using simple prefix dedup."""
        seen   = set()
        unique = []
        for s in snippets:
            key = s['body'][:100].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(s)
        return unique
 
    def _ask_confirmation(self, query: str) -> str | None:
        """Medium model asks one clarifying question for complex goals.
        Returns the clarification question string, or None if not needed.
        """
        if not _NEEDS_CONFIRMATION.search(query):
            return None
        prompt = (
            '<|im_start|>system\n'
            'You are a helpful assistant. Ask exactly ONE short clarifying question '
            'to better understand the user goal before you begin. Be direct.\n'
            '<|im_end|>\n'
            f'<|im_start|>user\n{query}<|im_end|>\n'
            '<|im_start|>assistant\nQuestion:'
        )
        q = self.models.complete_light(prompt, max_tokens=60).strip()
        return q if q else None
 
    def run(self, query: str, context: str = '') -> dict:
        confirm_q = self._ask_confirmation(query)
 
        queries  = self._expand_queries(query)
        snippets = []
        for q in queries[:config.RESEARCH_MAX_ITER]:
            snippets.extend(self._ddg_search(q))
            time.sleep(0.3)   # polite rate limiting
 
        snippets  = self._deduplicate(snippets)[:10]
        ctx_lines = [
            f'[{i+1}] {s["title"]}\n{s["body"]}\n(url: {s["href"]})'
            for i, s in enumerate(snippets)
        ]
        ctx = '\n\n'.join(ctx_lines)
 
        prompt = self._build_prompt(_SYSTEM, ctx, query)
        result = complete_medium_with_confidence(prompt, self.models)
 
        return {
            'response':        result['text'],
            'confidence':      result['confidence'],
            'low_conf':        result['low_confidence_flag'],
            'confirmation_q':  confirm_q,
            'sources':         [s['href'] for s in snippets],
            'snippets_used':   len(snippets),
        }
