from __future__ import annotations
import re, sys
from core.model_manager import models
 
_ANSWER_KEY_SENTINEL = '--- ANSWER KEY'
_Q_PATTERN = re.compile(r'^\s*(\d+)\.\s*(.+)', re.MULTILINE)
 
def parse_questions(test_text: str) -> list[dict]:
    body = test_text.split(_ANSWER_KEY_SENTINEL)[0]
    questions = []
    blocks = re.split(r'(?=^\s*\d+\.)', body, flags=re.MULTILINE)
    for block in blocks:
        block = block.strip()
        if not block or not re.match(r'^\d+\.', block):
            continue
        lines = block.split('\n')
        q_text = lines[0].strip()
        options = [l.strip() for l in lines[1:] if re.match(r'\s+[a-d]\)', l)]
        questions.append({'text': q_text, 'options': options})
    return questions
 
def run_test_session(
    test_text: str,
    input_fn=input,
    output_fn=print,
) -> dict:
    questions = parse_questions(test_text)
    if not questions:
        output_fn('Could not parse test questions. Check the format.')
        return {}
 
    answers  = []
    output_fn(f'\n=== TEST SESSION — {len(questions)} questions ===\n')
 
    for i, q in enumerate(questions, 1):
        output_fn(f'Q{i}: {q["text"]}')
        for opt in q['options']:
            output_fn(f'  {opt}')
        answer = input_fn('Your answer: ').strip()
        answers.append({'q': i, 'answer': answer})
        output_fn('')
 
    answer_block = '\n'.join(f'Q{a["q"]}: {a["answer"]}' for a in answers)
    prompt = (
        '<|im_start|>system\n'
        'You are a grading assistant. Compare the student answers to the test. '
        'Return a JSON-like block: {score: N, total: M, feedback: [...]}. '
        'For each question give brief feedback on whether the answer was correct.\n'
        '<|im_end|>\n<|im_start|>user\n'
        f'TEST:\n{test_text[:2000]}\n\n'
        f'STUDENT ANSWERS:\n{answer_block}\n'
        '<|im_end|>\n<|im_start|>assistant\n'
    )
    feedback = models.complete_medium(prompt, max_tokens=800)
    output_fn('\n=== RESULTS ===\n')
    output_fn(feedback)
    return {'raw_feedback': feedback, 'answers': answers}
