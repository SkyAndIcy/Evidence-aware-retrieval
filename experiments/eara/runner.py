import asyncio
import copy
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import sqlite3

from .agents import Agent, CHECKPOINT_METHODS, IMPLEMENTATIONS, PROMPT_VERSION, UNIMPLEMENTED
from .common import digest, file_digest, now, path, public_question, read_jsonl, write_json, write_jsonl
from .gateway import Context, InfrastructureFailure, RateLimited, Router, RunBudgetExceeded
from .metrics import answer_scores, summarize
from .protocol import validate_protocol
from .retrieval import Retriever


class ExclusiveRun:
    def __init__(self, directory):
        self.directory = Path(directory)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / '.active_run.lock').open('a')
        try:
            fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.stream.close()
            raise ValueError('Another run or labeling process is active in this run root; the concurrency cap is shared') from None
        return self

    def __exit__(self, *args):
        fcntl.flock(self.stream, fcntl.LOCK_UN)
        self.stream.close()


def code_revision():
    root = Path(__file__).parent
    return digest({filename.name: file_digest(filename) for filename in sorted(root.glob('*.py'))})


def retrieval_from(config, condition):
    options = config['retrieval']
    weights = condition.get('retrieval_weights', [options['lexical_weight'], options['dense_weight']])
    backend = 'hybrid' if 'retrieval_weights' in condition else options['backend']
    return Retriever(path(options['corpus_db']), path(options['bm25_index']), path(options['dense_index']), backend=backend, lexical_weight=weights[0], dense_weight=weights[1], candidate_depth=options['candidate_depth'], top_k=options['top_k'], rrf_constant=options['rrf_constant'], scan_batch_size=options['dense']['scan_batch_size'])


def load_threshold(config, condition, mode, router, split_manifest_hash):
    if not condition.get('cac', condition['method'] == 'eara'):
        return None, None
    filename = config['eara'].get('threshold_artifact')
    if filename:
        artifact = json.loads(path(filename).read_text())
        if 'threshold' not in artifact or artifact.get('prompt_version') != PROMPT_VERSION:
            raise ValueError('Threshold selection failed or its reviewer prompt version differs')
        reviewer = condition.get('reviewer', config['roles']['reviewer'])
        if artifact['reviewer_identity'] != router.identity(reviewer):
            raise ValueError('Threshold artifact is for a different reviewer')
        if artifact['split_manifest_sha256'] != split_manifest_hash or artifact['selection_split'] != 'dev':
            raise ValueError('Threshold artifact was not selected from this development split')
        if mode == 'paper' and not artifact.get('closed_loop_validated'):
            raise ValueError('Sequential controller must be rerun and validated on development data before held-out use')
        return artifact['threshold'], file_digest(path(filename))
    if mode == 'paper':
        raise ValueError('No development-selected threshold artifact is configured')
    return config['eara']['pilot_threshold'], None


def make_grid(config, family):
    rows = []
    if family == 'main':
        methods = ['direct', 'cot', 'tir', 'rag', 'self_ask', 'crag_local', 'react', 'react_abstain', 'ircot', 'search_o1', 'eara', 'r1_base', 'r1_instruct', 'search_r1', 'zerosearch_base', 'zerosearch_instruct']
        rows = [{'id': method, 'method': method, 'family': 'main'} for method in methods]
    elif family == 'decomposition':
        for role in config['ablations']['model_roles']:
            for count in config['ablations']['subquestion_counts']:
                rows.append({'id': f'{role}_m{count}', 'method': 'eara', 'family': family, 'generator': role, 'planner': role, 'subquestions': count, 'decompose': count != 1, 'crp': True, 'cac': True})
    elif family == 'planner_crossed':
        # Sec. 6.1 secondary crossed test: vary the decomposition model while
        # holding the answer generator and reviewer fixed, as a separately
        # labeled complement to the same-model decomposition grid above.
        for role in config['ablations']['model_roles']:
            for count in config['ablations']['subquestion_counts']:
                rows.append({'id': f'crossed_{role}_m{count}', 'method': 'eara', 'family': family, 'generator': config['roles']['generator'], 'planner': role, 'subquestions': count, 'decompose': count != 1, 'crp': True, 'cac': True})
    elif family == 'rewriting':
        for role in config['ablations']['model_roles']:
            for decompose in [False, True]:
                for rewrite in [False, True]:
                    rows.append({'id': f'{role}_D{int(decompose)}W{int(rewrite)}', 'method': 'eara', 'family': family, 'generator': role, 'planner': role, 'crp': True, 'cac': True, 'decompose': decompose, 'rewrite': rewrite, 'subquestions': 3 if decompose else 1})
    elif family == 'rewriting_control':
        # Sec. 6.2 separate control: the D/W factorial rerun with CAC disabled,
        # so rewriting effects can be read without the release gate.
        for role in config['ablations']['model_roles']:
            for decompose in [False, True]:
                for rewrite in [False, True]:
                    rows.append({'id': f'{role}_D{int(decompose)}W{int(rewrite)}_noCac', 'method': 'eara', 'family': family, 'generator': role, 'planner': role, 'crp': True, 'cac': False, 'decompose': decompose, 'rewrite': rewrite, 'subquestions': 3 if decompose else 1})
    elif family == 'retrieval':
        for lexical, dense in config['ablations']['retrieval_ratios']:
            rows.append({'id': f'bm25_{lexical}_dense_{dense}', 'method': 'eara', 'family': family, 'retrieval_weights': [lexical, dense]})
    elif family == 'transfer':
        for host in config['ablations']['transfer_hosts']:
            for crp, cac in config['ablations']['transfer_conditions']:
                rows.append({'id': f'{host}_CRP{int(crp)}CAC{int(cac)}', 'method': host, 'family': family, 'crp': crp, 'cac': cac})
    else:
        raise ValueError('Unknown grid family')
    return [{**row, 'implementation': IMPLEMENTATIONS.get(row['method']), 'blocked_reason': UNIMPLEMENTED.get(row['method']), 'seeds': config['study']['inference_seeds']} for row in rows]


async def run_study(config, condition, split, mode, seeds, limit=None, retry_failures=False, plans_only=False, transport_client=None, additional_request_budget=0):
    validate_protocol(config, condition, mode)
    if mode not in {'pilot', 'development', 'paper'}:
        raise ValueError('Invalid run mode')
    if additional_request_budget < 0:
        raise ValueError('Additional request budget must be nonnegative')
    if mode in {'pilot', 'development'} and split != 'dev':
        raise ValueError('Pilots and development runs must not access held-out test data')
    if mode == 'paper' and (limit is not None or transport_client is not None):
        raise ValueError('Paper mode does not permit sampling caps or injected test transports')
    if mode == 'paper' and (config['retrieval']['corpus_identity'] != 'dpr_december_2018_wikipedia_psgs_w100' or set(config['datasets']) != {'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'morehopqa'}):
        raise ValueError('Paper mode requires the declared full Wikipedia identity and all five primary English datasets')
    if condition['method'] in UNIMPLEMENTED:
        raise ValueError(UNIMPLEMENTED[condition['method']])
    if condition['method'] not in IMPLEMENTATIONS:
        raise ValueError('Unknown method')
    config = copy.deepcopy(config)
    config['runtime_mode'] = mode
    config['plans_only'] = plans_only
    split_root = path(config['splits']['directory'])
    split_manifest = json.loads((split_root / 'manifest.json').read_text())
    split_hash = file_digest(split_root / 'manifest.json')
    data_file = split_root / (split + '.jsonl')
    if file_digest(data_file) != split_manifest['files'][split]:
        raise ValueError('Frozen split has been modified')
    questions = list(read_jsonl(data_file))
    if any(record.get('split') != split for record in questions):
        raise ValueError('Split field mismatch')
    if limit is not None:
        if limit < 1:
            raise ValueError('Question cap must be positive')
        questions = questions[:limit]
    if not questions:
        raise ValueError('Selected split is empty')
    if len({record['id'] for record in questions}) != len(questions):
        raise ValueError('Duplicate question identifiers')
    if any(seed not in config['study']['inference_seeds'] for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Seeds must be distinct members of the frozen seed list')
    run_root = path(config['output']['runs'])
    with ExclusiveRun(run_root):
        bootstrap_router = Router(config, run_root / '.readiness_ledger.jsonl', client=transport_client)
        generator = condition.get('generator', config['roles']['generator'])
        if condition['method'] in CHECKPOINT_METHODS:
            generator = condition['method']
        roles = {generator}
        if condition.get('crp', condition['method'] == 'eara'):
            roles.add(condition.get('planner', config['roles']['planner'] if condition['method'] in CHECKPOINT_METHODS else generator))
        if condition.get('cac', condition['method'] == 'eara') and not plans_only:
            roles.add(condition.get('reviewer', config['roles']['reviewer']))
        try:
            identities = {role: bootstrap_router.identity(role) for role in sorted(roles)}
            threshold, threshold_hash = (None, None) if plans_only else load_threshold(config, condition, mode, bootstrap_router, split_hash)
        finally:
            await bootstrap_router.close()
        if mode == 'paper':
            if any(not identity['revision'] for identity in identities.values()):
                raise ValueError('Freeze all participating model revisions before paper runs')
            if any(config['models'][role].get('protocol_change') for role in roles):
                raise ValueError('Resolve and document pending model/protocol mappings before paper runs')
            if condition.get('family', 'main') == 'main' and generator != 'main_generator' and condition['method'] not in CHECKPOINT_METHODS:
                raise ValueError('Main fixed-generator comparison requires the configured main_generator; use a labeled ablation or pilot instead')
        needs_retrieval = condition['method'] not in {'direct', 'cot', 'r1_base', 'r1_instruct'} and not plans_only
        retriever = retrieval_from(config, condition) if needs_retrieval else None
        if mode == 'paper' and retriever and (retriever.metadata['identity'] != config['retrieval']['corpus_identity'] or retriever.metadata['is_partial']):
            raise ValueError('A diagnostic or partial corpus cannot be used as the paper corpus')
        fingerprint = {'config': config, 'condition': condition, 'split': split, 'split_manifest_sha256': split_hash, 'model_identities': identities, 'code_revision': code_revision(), 'prompt_version': PROMPT_VERSION, 'retriever_revision': retriever.revision if retriever else None, 'threshold_sha256': threshold_hash, 'threshold': threshold, 'seeds': seeds, 'question_ids': [record['id'] for record in questions], 'mode': mode, 'plans_only': plans_only, 'injected_validation_transport': transport_client is not None}
        run_id = digest(fingerprint)
        directory = run_root / run_id[:20]
        directory.mkdir(exist_ok=True)
        manifest_file = directory / 'run_manifest.json'
        existing_run = manifest_file.exists()
        if additional_request_budget and not existing_run:
            raise ValueError('Budget extensions apply only to an existing identical run')
        if manifest_file.exists():
            if json.loads(manifest_file.read_text())['fingerprint'] != fingerprint:
                raise ValueError('Run directory collision')
        else:
            write_json(manifest_file, {'run_id': run_id, 'created_at': now(), 'fingerprint': fingerprint, 'implementation': IMPLEMENTATIONS[condition['method']], 'benchmark_eligible': mode == 'paper' and not plans_only})
        connection = sqlite3.connect(directory / 'jobs.sqlite')
        connection.execute('PRAGMA journal_mode=WAL')
        connection.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, question_id TEXT NOT NULL, seed INTEGER NOT NULL, status TEXT NOT NULL, attempt INTEGER NOT NULL, result TEXT)')
        for question in questions:
            for seed in seeds:
                job_id = digest([run_id, question['id'], seed])
                connection.execute('INSERT OR IGNORE INTO jobs VALUES (?,?,?,?,?,?)', [job_id, question['id'], seed, 'assigned', 0, None])
        connection.execute("UPDATE jobs SET status='assigned' WHERE status='running'")
        if retry_failures:
            connection.execute("UPDATE jobs SET status='assigned' WHERE status='execution_failure' AND attempt < ?", [config['execution']['max_infrastructure_attempts_per_question']])
        connection.commit()
        lookup = {record['id']: record for record in questions}
        pending = asyncio.Queue()
        for job in connection.execute("SELECT id,question_id,seed,attempt FROM jobs WHERE status='assigned' ORDER BY rowid"):
            pending.put_nowait(job)
        extension_file = directory / 'budget_extensions.jsonl'
        if additional_request_budget:
            with extension_file.open('a') as stream:
                stream.write(json.dumps({'additional_requests': additional_request_budget, 'explicit_cli_authorization': True, 'timestamp': now()}) + '\n')
        extended = sum(row['additional_requests'] for row in read_jsonl(extension_file)) if extension_file.exists() else 0
        router_config = copy.deepcopy(config)
        router_config['execution']['max_api_requests_per_run'] += extended
        router = Router(router_config, directory / 'api_ledger.jsonl', client=transport_client)
        agent = Agent(config, router, retriever, condition, threshold)
        stop = asyncio.Event()
        async def worker():
            while not pending.empty() and not stop.is_set():
                job_id, question_id, seed, previous_attempt = pending.get_nowait()
                attempt = previous_attempt + 1
                connection.execute("UPDATE jobs SET status='running',attempt=? WHERE id=?", [attempt, job_id])
                connection.commit()
                trace = directory / 'traces' / job_id / f'attempt_{attempt}.jsonl'
                trace.parent.mkdir(parents=True, exist_ok=True)
                def sink(event):
                    with trace.open('a') as stream:
                        stream.write(json.dumps(router.sanitize(event), ensure_ascii=False, allow_nan=False) + '\n')
                question = lookup[question_id]
                context = Context(job_id, public_question(question), config, seed, sink)
                base = {'run_id': run_id, 'job_id': job_id, 'question_id': question_id, 'cluster_id': question_id, 'dataset': question['dataset'], 'split': split, 'split_manifest_sha256': split_hash, 'condition': condition, 'seed': seed, 'attempt': attempt, 'generator': generator, 'model_identities': identities, 'benchmark_eligible': mode == 'paper' and not plans_only, 'mode': mode, 'corpus_sha256': retriever.metadata['corpus_sha256'] if retriever else None, 'retriever_revision': retriever.revision if retriever else None}
                try:
                    if plans_only:
                        plan = await agent.plan(context)
                        result = {'status': 'plan_frozen', 'realized_subquestions': len(plan), 'answer': None, 'proposals': [], **context.metrics()}
                    else:
                        async with asyncio.timeout(config['execution']['per_question_seconds']):
                            result = await agent.run(context)
                except TimeoutError:
                    result = agent.finish(context, None, 'abstained', 'wall_clock_limit')
                except InfrastructureFailure as error:
                    result = {'status': 'execution_failure', 'answer': None, 'error_status': str(error), 'proposals': context.proposals, **context.metrics()}
                    stop.set()
                except Exception as error:
                    result = {'status': 'execution_failure', 'answer': None, 'error_status': type(error).__name__, 'proposals': context.proposals, **context.metrics()}
                    stop.set()
                scored = answer_scores(result['answer'] if result['status'] == 'answered' else None, question['answers'], question['language'])
                record = router.sanitize({**base, **result, **scored, 'reference_answers_sha256': digest(question['answers'])})
                prior_attempts = [json.loads(filename.read_text()) for filename in trace.parent.glob('result_attempt_*.json')]
                costs = [*prior_attempts, record]
                record['cost_including_prior_attempts'] = {key: sum(item.get(key, 0) for item in costs) for key in ['logical_retrieval_calls', 'physical_retrieval_calls', 'generation_tokens_upper_bound', 'elapsed_seconds']}
                record['cost_including_prior_attempts']['complete_attempt_records'] = len(prior_attempts) == attempt - 1
                record['cost_including_prior_attempts']['missing_usage_calls'] = sum(item.get('missing_usage_calls', 0) for item in costs)
                write_json(trace.parent / f'result_attempt_{attempt}.json', record)
                connection.execute('UPDATE jobs SET status=?,result=? WHERE id=?', [result['status'], json.dumps(record, ensure_ascii=False), job_id])
                connection.commit()
                pending.task_done()
        try:
            await asyncio.gather(*(worker() for _ in range(min(config['execution']['question_workers'], pending.qsize()))))
        finally:
            await router.close()
        records = []
        for job_id, question_id, seed, status, attempt, result in connection.execute('SELECT * FROM jobs ORDER BY rowid'):
            if result and status not in {'assigned', 'running'}:
                records.append(json.loads(result))
            else:
                records.append({'run_id': run_id, 'job_id': job_id, 'question_id': question_id, 'dataset': lookup[question_id]['dataset'], 'seed': seed, 'status': status, 'attempt': attempt, 'split_manifest_sha256': split_hash, 'mode': mode, 'benchmark_eligible': False})
        connection.close()
        write_jsonl(directory / 'results.jsonl', records)
        groups = defaultdict(list)
        for record in records:
            groups[record['dataset']].append(record)
        summary = {'run_id': run_id, 'mode': mode, 'plans_only': plans_only, 'implementation': IMPLEMENTATIONS[condition['method']], 'overall': summarize(records), 'datasets': {name: summarize(items) for name, items in groups.items()}, 'api_requests_total_including_prior_attempts': router.request_count, 'authorized_request_budget': router_config['execution']['max_api_requests_per_run'], 'peak_in_flight_this_invocation': router.peak, 'formal_results_not_automatically_written_to_paper': True}
        write_json(directory / 'summary.json', summary)
        return directory, summary
