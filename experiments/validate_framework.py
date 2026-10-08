import asyncio
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.eara.agents import validate_judgment
from experiments.eara.audit import export_proposals, machine_label, make_blind_audit, human_labels
from experiments.eara.common import config_load, read_jsonl, write_json, write_jsonl
from experiments.eara.data import corpus_import, freeze_splits, normalize_dataset
from experiments.eara.gateway import Context, Router, RateLimited
from experiments.eara.metrics import agreement, answer_scores, paired_cluster_bootstrap
from experiments.eara.retrieval import build_bm25, Retriever, fuse
from experiments.eara.runner import make_grid, run_study


Path('tmp/experiments').mkdir(parents=True, exist_ok=True)
root = Path(tempfile.mkdtemp(prefix='framework_validation_', dir='tmp/experiments')).resolve()
config = config_load('experiments/configs/paper.yaml')
source = root / 'corpus.tsv'
with source.open('w') as stream:
    writer = csv.writer(stream, delimiter='\t')
    writer.writerow(['id', 'text', 'title'])
    writer.writerow(['p1', 'The fictional book Silver Orchard was written by Ada Vale.', 'Silver Orchard'])
    writer.writerow(['p2', 'Ada Vale was born in Stonehaven.', 'Ada Vale'])
    for index in range(6):
        writer.writerow([f'p{index + 3}', f'Unrelated mineral type {index} is blue.', f'Mineral {index}'])
config['retrieval'].update({'corpus_db': str(root / 'corpus.sqlite'), 'bm25_index': str(root / 'bm25'), 'corpus_identity': 'synthetic_validation_only'})
metadata = corpus_import(source, config['retrieval']['corpus_db'], 'synthetic_validation_only')
build_bm25(config['retrieval']['corpus_db'], config['retrieval']['bm25_index'], config['retrieval']['bm25'])
retriever = Retriever(config['retrieval']['corpus_db'], config['retrieval']['bm25_index'])
assert retriever.search('Silver Orchard')['passages'][0]['id'] == 'p1'
assert len(retriever.search('Unrelated mineral blue')['passages']) == 5
assert not retriever.search('xyznonsenseword')['passages']
for weight in range(11):
    ranking = fuse([(1, 5), (2, 3)], [(2, .9), (3, .7)], weight, 10 - weight)
    assert len(ranking) == (2 if weight in {0, 10} else 3)
    if weight == 0:
        assert 1 not in dict(ranking)
    if weight == 10:
        assert 3 not in dict(ranking)

for dataset in ['hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'morehopqa']:
    input_file = root / (dataset + '.json')
    write_json(input_file, [{'id': 'one', 'question': 'Where was Ada Vale born?', 'answer': 'Stonehaven', 'answer_aliases': ['Stone Haven'], 'context': [['GOLD', ['SECRET GOLD CONTEXT']]]}])
    destination = root / (dataset + '.normalized.jsonl')
    normalize_dataset(dataset, input_file, destination, 'https://example.invalid/synthetic-fixture')
    row = list(read_jsonl(destination))[0]
    assert row['answers'] == ['Stonehaven', 'Stone Haven']
    assert 'SECRET GOLD CONTEXT' not in destination.read_text()

dataset_file = root / 'synthetic.jsonl'
write_jsonl(dataset_file, [{'id': f'synthetic:{index}', 'dataset': 'synthetic', 'question': f'Where was the author of Silver Orchard born? Example {index}.', 'language': 'en', 'answers': ['Stonehaven', 'DO_NOT_LEAK_GOLD_SENTINEL'], 'parent_ids': [], 'parent_mapping_reviewed': True} for index in range(4)])
config['datasets'] = {'synthetic': {'input': str(dataset_file), 'dev': 3, 'test': 1}}
config['splits']['directory'] = str(root / 'splits')
config['output']['runs'] = str(root / 'runs')
config['eara']['plan_cache'] = str(root / 'plans')
config['eara']['requested_subquestions'] = 2
config['study']['credentials_file_env'] = 'EARA_VALIDATION_CREDENTIALS_UNUSED'
config['execution']['max_api_requests_per_run'] = 500
for role in ['main_generator', 'frontier', 'small_reviewer']:
    config['models'][role] = {'transport': 'local', 'endpoint_env': 'EARA_VALIDATION_URL', 'model': role, 'revision': 'synthetic_mock_only', 'seed_supported': True}
os.environ['EARA_VALIDATION_URL'] = 'http://127.0.0.1:9999/v1'
freeze_splits(config)
dev = list(read_jsonl(root / 'splits/dev.jsonl'))
test = list(read_jsonl(root / 'splits/test.jsonl'))
assert not {row['id'] for row in dev} & {row['id'] for row in test}
assert answer_scores('The Stonehaven!', ['Stonehaven']) == {'em': 1.0, 'f1': 1.0}
assert answer_scores(None, ['Stonehaven'])['em'] == 0
assert agreement([True, True], [True, True])['kappa'] is None
assert agreement([True, False], [True, False])['kappa'] == 1
for family, expected in [('main', 16), ('decomposition', 24), ('rewriting', 16), ('retrieval', 11), ('transfer', 40)]:
    assert len(make_grid(config, family)) == expected

requests = []
async def response(request):
    payload = json.loads(request.content)
    assert 'DO_NOT_LEAK_GOLD_SENTINEL' not in json.dumps(payload)
    requests.append(payload)
    messages = payload['messages']
    instruction = messages[0]['content']
    data = json.loads(messages[1]['content'])
    if 'Decompose the question' in instruction:
        result = {'subquestions': [{'id': 's1', 'question': 'Who wrote Silver Orchard?', 'dependencies': []}, {'id': 's2', 'question': 'Where was the writer born?', 'dependencies': ['s1']}]}
    elif 'Rewrite the selected' in instruction:
        result = {'query': 'Ada Vale birthplace born', 'passage_ids': ['p1']}
    elif 'Judge whether' in instruction:
        assert set(data) == {'question', 'candidate', 'passages'}
        identifiers = {item['id'] for item in data['passages']}
        supported = {'p1', 'p2'}.issubset(identifiers)
        result = {'score': .95 if supported else .1, 'sufficient': supported, 'passage_ids': sorted(identifiers), 'missing_links': [] if supported else ['birthplace'], 'rationale': 'synthetic validation response'}
    else:
        ids = {item['id'] for item in data.get('passages', [])}
        updates = [{'id': 's1', 'status': 'supported', 'answer': 'Ada Vale', 'passage_ids': ['p1']}] if 'p1' in ids else []
        if 'p2' in ids:
            updates.append({'id': 's2', 'status': 'supported', 'answer': 'Stonehaven', 'passage_ids': ['p2']})
        result = {'answer': 'Stonehaven', 'propose': True, 'abstain': False, 'subquestion_updates': updates}
    await asyncio.sleep(.005)
    return httpx.Response(200, json={'model': payload['model'], 'choices': [{'message': {'content': json.dumps(result)}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 20, 'completion_tokens': 30, 'total_tokens': 50}})


async def check():
    async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
        condition = {'id': 'eara', 'method': 'eara', 'subquestions': 2}
        directory, summary = await run_study(config, condition, 'dev', 'pilot', [17], transport_client=client)
        rows = list(read_jsonl(directory / 'results.jsonl'))
        assert summary['overall']['answered'] == 3, rows
        assert all(row['em'] == 1 and row['logical_retrieval_calls'] == 2 for row in rows), rows
        assert all(len(row['proposals']) == 2 and row['proposals'][-1]['is_terminal'] and not row['proposals'][0]['is_terminal'] for row in rows)
        assert all(not row['benchmark_eligible'] for row in rows)
        before = len(requests)
        resumed, repeated = await run_study(config, condition, 'dev', 'pilot', [17], transport_client=client)
        assert resumed == directory and len(requests) == before
        comparison = paired_cluster_bootstrap(rows, rows, repeats=100)
        assert comparison['delta_yield_em'] == 0 and comparison['cluster_bootstrap_95_ci'] == [0, 0]
        tuples = root / 'tuples.jsonl'
        assert export_proposals([directory / 'results.jsonl'], tuples) == 6
        labels_dir = root / 'model_labels'
        result = await machine_label(config, tuples, ['frontier', 'small_reviewer'], labels_dir, transport_client=client)
        assert result['successes'] == 12 and result['human_labels_created'] == 0
        audit = root / 'audit'
        result = make_blind_audit(tuples, labels_dir / 'machine_labels.jsonl', audit, 'frontier', 'small_reviewer', .5, .5, 1.0)
        assert result['selected'] == 3
        blind = list(read_jsonl(audit / 'tasks_A.jsonl'))
        assert all(set(row) == {'task_id', 'question', 'candidate', 'passages'} for row in blind)
        try:
            human_labels(labels_dir / 'machine_labels.jsonl')
        except ValueError:
            pass
        else:
            raise AssertionError('Machine labels were accepted as human labels')
        try:
            await run_study(config, condition, 'test', 'pilot', [17], transport_client=client)
        except ValueError:
            pass
        else:
            raise AssertionError('Pilot accessed test data')
        try:
            validate_judgment({'score': .9, 'sufficient': True, 'passage_ids': ['fabricated']}, {'p1'})
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid evidence ID accepted')
        cap_router = Router(config, root / 'concurrency_ledger.jsonl', client=client)
        contexts = [Context(str(index), {'id': str(index), 'question': 'synthetic'}, config, 17, lambda event: None) for index in range(25)]
        await asyncio.gather(*(cap_router.complete(context, 'main_generator', [{'role': 'system', 'content': 'Return JSON answer'}, {'role': 'user', 'content': '{}'}], 'agent') for context in contexts))
        assert cap_router.peak == 20
        assert len({json.dumps(request) for request in requests[-25:]}) == 1
    counters = {'network_calls': 0}
    async def limited(request):
        counters['network_calls'] += 1
        return httpx.Response(429, json={'error': 'rate limited'}, headers={'Retry-After': '60'})
    async with httpx.AsyncClient(transport=httpx.MockTransport(limited)) as client:
        router = Router(config, root / 'rate_limit_ledger.jsonl', client=client)
        for index in range(2):
            context = Context('rate-' + str(index), {'id': 'rate', 'question': 'test'}, config, 17, lambda event: None)
            try:
                await router.complete(context, 'main_generator', [{'role': 'user', 'content': 'test'}], 'agent')
            except RateLimited:
                pass
            else:
                raise AssertionError('Rate limit was not raised')
        assert counters['network_calls'] == 1
    return directory


directory = asyncio.run(check())
report = {'status': 'passed', 'fixture': 'synthetic_only_not_paper_results', 'real_BM25s_index_tested': True, 'model_transport': 'httpx.MockTransport', 'checks': ['five_dataset_normalizers_exclude_gold_context', 'deterministic_disjoint_splits', 'top5_BM25_mmap_OOV', 'all_11_RRF_weights', 'two_hop_CRP_rewrite_CAC_rejection_then_acceptance', 'no_gold_answer_sent_to_models', 'terminal_proposal_tracking', 'resume_does_not_repeat_successes', 'paired_cluster_bootstrap_identity', 'independent_machine_review_inputs', 'blind_audit_export', 'machine_labels_rejected_as_human', 'pilot_test_split_block', 'invalid_passage_ID_rejection', 'global_20_request_concurrency_cap', '429_cooldown_no_key_rotation'], 'validation_directory': str(root), 'example_run_directory': str(directory)}
write_json('tmp/experiments/framework_validation.json', report)
print(json.dumps(report, ensure_ascii=False, indent=2))
