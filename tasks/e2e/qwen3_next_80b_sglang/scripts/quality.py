"""Frozen answer-based model regression check; independent of token determinism.

Uses the pinned InferenceX GSM8K prompt and strict final-answer convention.
This small, fixed subset is a regression gate, not a full benchmark score.
"""
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def answer(text):
    # A draft inside an unfinished reasoning block is not a final answer. Some
    # chat templates put the opening tag in the prompt, outside returned text.
    if '<think>' in text and '</think>' not in text:
        return None
    text = text.rsplit('</think>', 1)[-1]
    match = re.search(r'####\s*(-?[0-9][0-9,.]*)\s*\Z', text)
    if not match:
        return None
    try:
        value = Decimal(match[1].replace(',', '').rstrip('.'))
        return str(value.normalize()) if value.is_finite() else None
    except InvalidOperation:
        return None


def load_cases(lock):
    rule = lock['correctness']['quality']
    content = (ROOT / 'quality.json').read_bytes()
    if hashlib.sha256(content).hexdigest() != rule['dataset_sha256']:
        raise ValueError('Model quality dataset differs from lock')
    data = json.loads(content)
    if len(data['samples']) != rule['samples'] or len(data['fewshot']) != rule['fewshot']:
        raise ValueError('Model quality sample coverage changed')
    if len({row['id'] for row in data['samples']}) != rule['samples']:
        raise ValueError('Model quality samples are duplicated')
    for row in [*data['samples'], *data['fewshot']]:
        if not isinstance(row['question'], str) or not row['question'] or answer(row['answer']) is None:
            raise ValueError('Invalid model quality case')
    return data


def completion_answer(text, finish, reasoning):
    # This pinned Instruct template ends at the assistant role marker; it does
    # not open a reasoning block. Normal final answers need no </think> tag.
    # answer() still rejects an explicit unfinished reasoning block, and a
    # truncated completion is never accepted even if it contains the answer.
    if finish != 'stop':
        return None
    return answer(text)


def run(port, lock, request, label):
    rule = lock['correctness']['quality']
    data = load_cases(lock)
    prefix = '\n\n'.join('Question: ' + row['question'] + '\nAnswer: ' + row['answer']
                         for row in data['fewshot']) + '\n\n'

    def evaluate(row):
        prompt = (prefix + 'Question: ' + row['question'] +
                  '\nEnd your response with the answer on the last line, formatted as: #### [number]\nAnswer:')
        # Use the model's pinned chat template, as InferenceX's quality client
        # does. Keep the model's default reasoning mode; serving flags are fixed.
        response = request(port, 'v1/chat/completions', dict(model=str(ROOT / 'model'),
            messages=[dict(role='user', content=prompt)], temperature=0, top_p=1,
            max_tokens=rule['max_new_tokens'], stop=['</s>', '<|im_end|>']))
        choice = response['choices'][0]
        text = choice['message']['content']
        reasoning = choice['message'].get('reasoning_content')
        tokens = response['usage']['completion_tokens']
        finish = choice['finish_reason']
        if not isinstance(text, str) or type(tokens) is not int or not 0 < tokens <= rule['max_new_tokens']:
            raise ValueError('Incomplete model quality response')
        if finish not in ('stop', 'length'):
            raise ValueError('Unexpected model quality termination')
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError('Invalid model reasoning response')
        expected = answer(row['answer'])
        actual = completion_answer(text, finish, reasoning)
        return dict(id=row['id'], expected=expected, actual=actual, correct=actual == expected,
                    completion_tokens=tokens, finish_reason=finish, response=text,
                    reasoning_content=reasoning)

    with ThreadPoolExecutor(max_workers=rule['concurrency']) as pool:
        rows = list(pool.map(evaluate, data['samples']))
    correct = sum(row['correct'] for row in rows)
    report = dict(schema_version='1.0', provider='task-owned-gsm8k', status='COMPLETED',
                  task='gsm8k_fixed_subset', metric='exact_match,strict-match',
                  score=correct / len(rows), samples=len(rows), correct=correct,
                  dataset_sha256=rule['dataset_sha256'], results=rows)
    (ROOT / (label + '_accuracy_report.json')).write_text(json.dumps(report, indent=2))
    return report


def validate(candidate, reference, lock):
    rule = lock['correctness']['quality']
    cases = load_cases(lock)['samples']
    expected_ids = [row['id'] for row in cases]
    for report in (candidate, reference):
        if (report.get('status') != 'COMPLETED' or report.get('samples') != rule['samples']
                or report.get('dataset_sha256') != rule['dataset_sha256']):
            raise ValueError('Incomplete model accuracy report')
        rows = report['results']
        if [row['id'] for row in rows] != expected_ids:
            raise ValueError('Model quality case identities changed')
        if any(type(row['correct']) is not bool or row['correct'] != (row['actual'] == row['expected'])
               or row['expected'] != answer(case['answer'])
               or row.get('finish_reason') not in ('stop', 'length')
               or type(row.get('completion_tokens')) is not int
               or not 0 < row['completion_tokens'] <= rule['max_new_tokens']
               or (row.get('reasoning_content') is not None and not isinstance(row['reasoning_content'], str))
               or row['actual'] != completion_answer(row['response'], row['finish_reason'], row.get('reasoning_content'))
               for row, case in zip(rows, cases)):
            raise ValueError('Invalid model quality answer evidence')
        score = sum(row['correct'] for row in rows) / rule['samples']
        if type(report['score']) not in (int, float) or not math.isfinite(report['score']) or report['score'] != score:
            raise ValueError('Model quality score differs from answer evidence')
        if score < rule['minimum_accuracy']:
            raise ValueError(f'Model accuracy {score:.4f} is below the locked minimum')
    if candidate['score'] + rule['maximum_accuracy_drop'] < reference['score']:
        raise ValueError('Candidate model accuracy regressed beyond the locked limit')
    return dict(candidate_accuracy=candidate['score'], reference_accuracy=reference['score'],
                samples=rule['samples'], dataset_sha256=rule['dataset_sha256'])
