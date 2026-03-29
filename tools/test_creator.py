from __future__ import annotations
from core.model_manager import models
from rag.hybrid_rag import HybridRAG
 
_LANG_LABELS = {'en': 'English', 'ro': 'Romanian', 'ru': 'Russian'}
 
def detect_language(text: str) -> str:
    ro_markers = ['și','sau','că','cu','pe','la','din','sunt','este','mai','cum','ce','dacă','care','pentru','despre','între','într','într-un','într-o']
    ru_markers = ['и','в','на','по','за','от','из','это','как','что','если','или','но','для']
    words = text.lower().split()
    ro_score = sum(1 for w in words if w in ro_markers)
    ru_score = sum(1 for w in words if w in ru_markers)
    if ru_score > ro_score and ru_score > 1: return 'ru'
    if ro_score > 1: return 'ro'
    return 'en'
 
def create_test(
    topic: str,
    n_questions: int = 10,
    q_type: str = 'MIXED',
    rag: HybridRAG | None = None,
    lang: str | None = None,
) -> str:
    lang = lang or detect_language(topic)
    lang_label = _LANG_LABELS.get(lang, 'English')
 
    context = ''
    if rag:
        chunks = rag.query(topic, top_k=4)
        context = rag.build_context_string(chunks)
 
    lang_instruction = {
        'ro': 'Scrie testul în limba română.',
        'ru': 'Напишите тест на русском языке.',
        'en': 'Write the test in English.',
    }.get(lang, 'Write the test in English.')
 
    prompt = (
        '<|im_start|>system\n'
        'You are an expert educator and test designer. '
        'Create tests exactly in the ANTIMONY TEST FORMAT shown below. '
        'Always include the answer key after the "--- ANSWER KEY ---" sentinel.\n'
        '=== ANTIMONY TEST FORMAT ===\n'
        '=== ANTIMONY TEST ===\n'
        'Topic: [topic]\nLanguage: [LANG]\nQuestions: [N]\nType: [TYPE]\n---\n'
        '1. [question]\n   a) ...\n   b) ...\n   c) ...\n   d) ...\n\n'
        '--- ANSWER KEY (do not show to student in ANY MEANING) ---\n'
        '1. c\n=== END TEST ===\n'
        '<|im_end|>\n'
        '<|im_start|>user\n'
        f'{lang_instruction}\n'
        f'Topic: {topic}\n'
        f'Number of questions: {n_questions}\n'
        f'Question type: {q_type}\n'
        + (f'Reference material:\n{context[:2000]}\n' if context else '')
        + '<|im_end|>\n<|im_start|>assistant\n'
    )
    return models.complete_medium(prompt, max_tokens=2000)
