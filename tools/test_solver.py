from __future__ import annotations
import re
from core.model_manager import models
from rag.hybrid_rag import HybridRAG
 
def solve_test(test_text: str, rag: HybridRAG | None = None) -> str:
    """
    Solve all questions in a test, providing answers with reasoning.
    Returns a solution sheet as a formatted string.
    """
    context = ''
    if rag:
        chunks  = rag.query(test_text[:500], top_k=4)
        context = rag.build_context_string(chunks)
 
    prompt = (
        '<|im_start|>system\n'
        'You are an expert tutor. Solve every question in the test provided. '
        'For MCQ questions, state the letter and explain why it is correct. '
        'For short-answer questions, give a complete answer with justification. '
        'Be concise but thorough. Indicate if any question is ambiguous.\n'
        '<|im_end|>\n<|im_start|>user\n'
        + (f'Reference material:\n{context[:1500]}\n\n' if context else '')
        + f'TEST TO SOLVE:\n{test_text}\n'
        + '<|im_end|>\n<|im_start|>assistant\n'
    )
    return models.complete_medium(prompt, max_tokens=2000)
