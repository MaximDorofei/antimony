from __future__ import annotations
import re, math, html
from pathlib import Path
import config
 
_INJECTION_PATTERNS = [
    r'ignore\s+(all\s+)?previous\s+instructions?',
    r'you\s+are\s+now\s+(a|an)',
    r'forget\s+(everything|all)',
    r'act\s+as\s+(if\s+you\s+are|a|an)',
    r'<\s*(system|SYS|INST|s)\b',
    r'\[\s*(INST|SYS|system)\s*\]',
    r'<<\s*SYS\s*>>',
    r'jailbreak|DAN\b|do\s+anything\s+now',
    r'reveal\s+(your\s+)?(system|prompt|instruction)',
    r'disregard\s+(your|all)',
]
 
_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL)
             for p in _INJECTION_PATTERNS]
 
_TAG_STRIP = re.compile(
    r'</?\s*(system|SYS|INST|s|im_start|im_end)\b[^>]*>',
    re.IGNORECASE
)
 
MAX_INPUT_CHARS = 4000
 
def sanitise(text: str) -> tuple[str, list[str]]:
    warnings = []
    text = html.unescape(text)
    text = _TAG_STRIP.sub('', text)
 
    for pattern in _COMPILED:
        if pattern.search(text):
            warnings.append(f'Injection pattern detected: {pattern.pattern[:40]}')
            text = pattern.sub('[REDACTED]', text)
 
    if len(text) > MAX_INPUT_CHARS:
        warnings.append(f'Input truncated from {len(text)} to {MAX_INPUT_CHARS} chars')
        text = text[:MAX_INPUT_CHARS]
 
    return text.strip(), warnings

def score_confidence(logprobs: list[float | None]) -> float:
    valid = [lp for lp in logprobs if lp is not None]
    if not valid:
        return 0.5   # unknown
    mean_lp = sum(valid) / len(valid)
    return float(min(1.0, max(0.0, 1.0 + mean_lp / 5.0)))
 
 
def complete_medium_with_confidence(
    prompt: str,
    models,
    max_tokens: int = 1024,
) -> dict:
    llm  = models.medium()
    resp = llm(
        prompt, max_tokens=max_tokens,
        stop=['</s>', '<|im_end|>'],
        logprobs=1, echo=False,
    )
    text = resp['choices'][0]['text'].strip()
    lps  = [
        tp.get('logprob')
        for tp in (resp['choices'][0].get('logprobs', {}).get('top_logprobs') or [])
        if tp
    ]
    conf = score_confidence(lps)
    return {
        'text':                text,
        'confidence':          round(conf, 3),
        'low_confidence_flag': conf < config.CONFIDENCE_THRESHOLD,
    }
