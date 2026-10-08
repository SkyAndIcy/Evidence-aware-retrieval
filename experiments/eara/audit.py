import asyncio
from collections import Counter
import json
import math
import os
from pathlib import Path
import random

from .agents import PROMPT_VERSION, SchemaFailure, review_tuple
from .common import digest, file_digest, now, path, read_jsonl, write_json, write_jsonl
from .gateway import Context, InfrastructureFailure, Router
from .metrics import agreement, classifier_metrics, wilson
from .runner import ExclusiveRun


def export_proposals(results_files, destination):
    tuples = {}
    for filename in results_files:
        for result in read_jsonl(filename):
            for proposal in result.get('proposals', []):
                item = {key: proposal[key] for key in ['proposal_id', 'question_id', 'question', 'candidate', 'passages', 'evidence_sha256', 'proposal_index', 'is_terminal']}
                item.update({'run_id': result['run_id'], 'dataset': result['dataset'], 'split': result['split'], 'split_manifest_sha256': result['split_manifest_sha256'], 'cluster_id': result.get('cluster_id', result['question_id']), 'generator': result['generator'], 'condition': result['condition']})
                if item['evidence_sha256'] != digest(item['passages']):
                    raise ValueError('Proposal evidence has changed')
                if item['proposal_id'] in tuples and tuples[item['proposal_id']] != item:
                    raise ValueError('Conflicting versions of the same proposal')
                tuples[item['proposal_id']] = item
    write_jsonl(destination, tuples.values())
    return len(tuples)


async def machine_label(config, tuples_file, roles, destination, transport_client=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    records = list(read_jsonl(tuples_file))
    if not records:
        raise ValueError('No proposal tuples to label')
    if len({record['proposal_id'] for record in records}) != len(records):
        raise ValueError('Duplicate proposal IDs')
    with ExclusiveRun(path(config['output']['runs'])):
        router = Router(config, destination / 'api_ledger.jsonl', client=transport_client)
        identities = {role: router.identity(role) for role in roles}
        fingerprint = {'input_sha256': file_digest(tuples_file), 'identities': identities, 'prompt_version': PROMPT_VERSION, 'execution': config['execution'], 'label_source': 'model'}
        manifest = destination / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
            await router.close()
            raise ValueError('This label directory belongs to a different input/model/prompt configuration')
        write_json(manifest, fingerprint)
        output = destination / 'machine_labels.jsonl'
        previous = list(read_jsonl(output)) if output.exists() else []
        done = {(record['proposal_id'], record['reviewer_role']) for record in previous if record['status'] == 'success'}
        queue = asyncio.Queue()
        for proposal in records:
            for role in roles:
                if (proposal['proposal_id'], role) not in done:
                    queue.put_nowait((proposal, role))
        stopped = asyncio.Event()
        async def worker():
            while not queue.empty() and not stopped.is_set():
                proposal, role = queue.get_nowait()
                if proposal['evidence_sha256'] != digest(proposal['passages']):
                    raise ValueError('Frozen evidence checksum mismatch')
                context = Context(digest([proposal['proposal_id'], role]), {'id': proposal['question_id'], 'question': proposal['question']}, config, config['study']['selection_seed'], lambda event: None)
                result = {'proposal_id': proposal['proposal_id'], 'question_id': proposal['question_id'], 'split': proposal['split'], 'split_manifest_sha256': proposal['split_manifest_sha256'], 'evidence_sha256': proposal['evidence_sha256'], 'reviewer_role': role, 'reviewer_identity': identities[role], 'label_source': 'model', 'prompt_version': PROMPT_VERSION, 'timestamp': now()}
                try:
                    judgment = await review_tuple(router, context, role, proposal['question'], proposal['candidate'], proposal['passages'])
                    result.update({'status': 'success', **judgment})
                except (InfrastructureFailure, SchemaFailure) as error:
                    result.update({'status': 'failed', 'error_status': str(error), 'score': None})
                    if isinstance(error, InfrastructureFailure):
                        stopped.set()
                result.update(context.metrics())
                previous.append(router.sanitize(result))
                write_jsonl(output, previous)
                queue.task_done()
        try:
            await asyncio.gather(*(worker() for _ in range(min(config['execution']['question_workers'], queue.qsize()))))
        finally:
            await router.close()
        return {'label_source': 'model', 'records': len(previous), 'successes': sum(record['status'] == 'success' for record in previous), 'pending': queue.qsize(), 'human_labels_created': 0}


def latest_model_labels(filename):
    output = {}
    for record in read_jsonl(filename):
        if record.get('label_source') != 'model':
            raise ValueError('Expected machine label provenance')
        if record['status'] == 'success':
            key = (record['proposal_id'], record['reviewer_role'])
            if key in output and output[key] != record:
                raise ValueError('More than one successful label for the same proposal and reviewer')
            output[key] = record
    return output


def model_agreement(filename, frontier, small, frontier_threshold, small_threshold):
    if not 0 <= frontier_threshold <= 1 or not 0 <= small_threshold <= 1:
        raise ValueError('Thresholds must lie in [0,1]')
    history = list(read_jsonl(filename))
    labels = latest_model_labels(filename)
    expected = {row['proposal_id'] for row in history if row['reviewer_role'] in {frontier, small}}
    complete = [identifier for identifier in sorted(expected) if (identifier, frontier) in labels and (identifier, small) in labels]
    incomplete = sorted(expected - set(complete))
    if incomplete:
        return {'status': 'incomplete', 'population': len(expected), 'completed_pairs': len(complete), 'incomplete_proposals': incomplete, 'kappa': None, 'human_labels_used': False}
    for identifier in complete:
        if labels[(identifier, frontier)]['evidence_sha256'] != labels[(identifier, small)]['evidence_sha256']:
            raise ValueError('Reviewers did not see identical evidence')
    first = [labels[(identifier, frontier)]['score'] >= frontier_threshold for identifier in complete]
    second = [labels[(identifier, small)]['score'] >= small_threshold for identifier in complete]
    return {'status': 'complete', 'population': len(complete), 'frontier_accepted': sum(first), 'small_accepted': sum(second), 'frontier_threshold': frontier_threshold, 'small_threshold': small_threshold, 'model_model_agreement': agreement(first, second), 'threshold_sweep': [{'threshold': threshold, 'frontier_accepted': sum(labels[(identifier, frontier)]['score'] >= threshold for identifier in complete), 'small_accepted': sum(labels[(identifier, small)]['score'] >= threshold for identifier in complete)} for threshold in [.1, .2, .3, .4, .5]], 'human_labels_used': False, 'model_equivalence_established': False}


def make_blind_audit(tuples_file, labels_file, destination, frontier, small, frontier_threshold, small_threshold, rejected_probability=.2, seed=17):
    if not 0 < rejected_probability <= 1 or not 0 <= frontier_threshold <= 1 or not 0 <= small_threshold <= 1:
        raise ValueError('Invalid sampling probability or thresholds')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Audit package already exists')
    labels = latest_model_labels(labels_file)
    population = [record for record in read_jsonl(tuples_file) if record['is_terminal']]
    if not population:
        raise ValueError('No terminal proposals')
    labeled = []
    for proposal in population:
        first, second = labels.get((proposal['proposal_id'], frontier)), labels.get((proposal['proposal_id'], small))
        if not first or not second:
            raise ValueError('Both reviewers must finish every terminal tuple before sampling; failed cases cannot be silently excluded')
        if first['evidence_sha256'] != proposal['evidence_sha256'] or second['evidence_sha256'] != proposal['evidence_sha256']:
            raise ValueError('Reviewer evidence mismatch')
        positive = first['score'] >= frontier_threshold or second['score'] >= small_threshold
        labeled.append((proposal, 'accepted_union' if positive else 'both_rejected'))
    counts = Counter(stratum for _, stratum in labeled)
    salt = os.urandom(32).hex()
    rng = random.Random(seed)
    tasks, mapping = [], []
    for proposal, stratum in sorted(labeled, key=lambda pair: pair[0]['proposal_id']):
        probability = 1.0 if stratum == 'accepted_union' else rejected_probability
        if rng.random() >= probability:
            continue
        task_id = digest([salt, proposal['proposal_id']])[:24]
        tasks.append({'task_id': task_id, 'question': proposal['question'], 'candidate': proposal['candidate'], 'passages': proposal['passages']})
        mapping.append({'task_id': task_id, 'proposal_id': proposal['proposal_id'], 'question_id': proposal['question_id'], 'cluster_id': proposal.get('cluster_id', proposal['question_id']), 'split': proposal['split'], 'split_manifest_sha256': proposal['split_manifest_sha256'], 'evidence_sha256': proposal['evidence_sha256'], 'sampling_stratum': stratum, 'population_size': counts[stratum], 'inclusion_probability': probability})
    destination.mkdir(parents=True)
    for annotator, offset in [('A', 1), ('B', 2)]:
        shuffled = tasks[:]
        random.Random(seed + offset).shuffle(shuffled)
        write_jsonl(destination / ('tasks_' + annotator + '.jsonl'), shuffled)
    private = destination / 'private'
    private.mkdir(mode=0o700)
    write_jsonl(private / 'mapping.jsonl', mapping)
    write_json(private / 'design.json', {'frontier_role': frontier, 'small_role': small, 'thresholds': [frontier_threshold, small_threshold], 'population_counts': dict(counts), 'selected': len(tasks), 'rejected_probability': rejected_probability, 'seed': seed, 'tuple_sha256': file_digest(tuples_file), 'model_labels_sha256': file_digest(labels_file)})
    for filename in private.iterdir():
        filename.chmod(0o600)
    return {'selected': len(tasks), 'population': len(population), 'human_labels_created': 0}


def human_labels(filename):
    labels = {}
    for record in read_jsonl(filename):
        if record.get('label_source') != 'human' or not record.get('annotator_id') or record.get('independent_review_attested') is not True:
            raise ValueError('Only independently reviewed labels explicitly attributed to a human annotator are accepted')
        if record.get('label') not in {'sufficient', 'insufficient', 'uncertain'}:
            raise ValueError('Invalid human label')
        if record['task_id'] in labels:
            raise ValueError('Duplicate human task ID')
        labels[record['task_id']] = record
    return labels


def audit_agreement(mapping_file, first_file, second_file, labels_file, frontier, small, frontier_threshold, small_threshold, adjudicated_file=None):
    mapping = {record['task_id']: record for record in read_jsonl(mapping_file)}
    first, second = human_labels(first_file), human_labels(second_file)
    if set(first) != set(mapping) or set(second) != set(mapping):
        raise ValueError('Both human raters must label the complete assigned audit population')
    if {row['annotator_id'] for row in first.values()} & {row['annotator_id'] for row in second.values()}:
        raise ValueError('The two human annotators must be distinct')
    usable = [task_id for task_id in mapping if first[task_id]['label'] != 'uncertain' and second[task_id]['label'] != 'uncertain']
    weights = [1 / mapping[task_id]['inclusion_probability'] for task_id in usable]
    result = {'assigned_tasks': len(mapping), 'uncertain_excluded_from_binary_kappa': len(mapping) - len(usable), 'human_human_before_adjudication': agreement([first[task_id]['label'] == 'sufficient' for task_id in usable], [second[task_id]['label'] == 'sufficient' for task_id in usable], weights), 'noninferiority_claim_allowed': False, 'noninferiority_reason': 'Requires predeclared margins, power analysis, clustered confidence intervals, and closed-loop evidence; kappa alone is insufficient'}
    if adjudicated_file:
        adjudicated = human_labels(adjudicated_file)
        if set(adjudicated) != set(mapping):
            raise ValueError('Adjudicated labels must cover all assigned tasks')
        initial_ids = {row['annotator_id'] for row in first.values()} | {row['annotator_id'] for row in second.values()}
        for task_id in mapping:
            disagreement = first[task_id]['label'] != second[task_id]['label'] or first[task_id]['label'] == 'uncertain'
            if disagreement and adjudicated[task_id]['annotator_id'] in initial_ids:
                raise ValueError('Disagreements/uncertainty require an independent third adjudicator')
        model_labels = latest_model_labels(labels_file)
        selected = [task_id for task_id in mapping if adjudicated[task_id]['label'] != 'uncertain']
        for role, threshold in [(frontier, frontier_threshold), (small, small_threshold)]:
            if any((mapping[task_id]['proposal_id'], role) not in model_labels for task_id in selected):
                raise ValueError('Missing model labels for audited tasks')
            result[role + '_vs_adjudicated_human_thresholded'] = classifier_metrics([adjudicated[task_id]['label'] == 'sufficient' for task_id in selected], [model_labels[(mapping[task_id]['proposal_id'], role)]['score'] >= threshold for task_id in selected], [1 / mapping[task_id]['inclusion_probability'] for task_id in selected])
        result['unresolvable_after_adjudication'] = len(mapping) - len(selected)
    return result


def calibration_rows(mapping_file, human_file, labels_file, reviewer, destination):
    mapping = list(read_jsonl(mapping_file))
    humans = human_labels(human_file)
    labels = latest_model_labels(labels_file)
    if set(humans) != {row['task_id'] for row in mapping}:
        raise ValueError('Adjudicated human labels must cover the complete assigned audit')
    rows = []
    for item in mapping:
        human = humans[item['task_id']]
        if human['label'] == 'uncertain':
            raise ValueError('Resolve or explicitly redesign development calibration before excluding uncertain labels')
        label = labels.get((item['proposal_id'], reviewer))
        if not label or label['evidence_sha256'] != item['evidence_sha256']:
            raise ValueError('Missing or mismatched reviewer label')
        rows.append({**item, 'score': label['score'], 'reviewer_identity': label['reviewer_identity'], 'human_sufficient': human['label'] == 'sufficient', 'label_source': 'adjudicated_human', 'human_source_sha256': file_digest(human_file), 'reviewer_source_sha256': file_digest(labels_file)})
    write_jsonl(destination, rows)
    return len(rows)


def validate_closed_loop(artifact_file, results_file, human_rows_file, destination):
    artifact = json.loads(Path(artifact_file).read_text())
    records = list(read_jsonl(results_file))
    humans = {row['proposal_id']: row for row in read_jsonl(human_rows_file)}
    if not records or any(row.get('split') != 'dev' or row.get('mode') != 'development' or row.get('split_manifest_sha256') != artifact['split_manifest_sha256'] for row in records):
        raise ValueError('Closed-loop validation requires this development split and development run mode')
    if any(row['status'] not in {'answered', 'abstained'} for row in records):
        raise ValueError('Complete development execution before validating the operating point')
    if len({row['question_id'] for row in records}) != len(records):
        raise ValueError('Use one run per development question for this Wilson validation')
    accepted, failures = 0, 0
    for record in records:
        if record['status'] != 'answered':
            continue
        proposal = record['proposals'][-1]
        human = humans.get(proposal['proposal_id'])
        if proposal.get('threshold') != artifact['threshold'] or proposal.get('score', -1) < artifact['threshold']:
            raise ValueError('Run does not deploy the selected CAC threshold')
        if not human or human.get('label_source') != 'adjudicated_human' or human.get('inclusion_probability') != 1 or human.get('evidence_sha256') != proposal['evidence_sha256'] or human.get('reviewer_identity') != artifact['reviewer_identity']:
            raise ValueError('Every accepted terminal answer needs matched exhaustive human adjudication')
        if not isinstance(human.get('human_sufficient'), bool):
            raise ValueError('Human evidence label is missing')
        accepted += 1
        failures += not human['human_sufficient']
    interval = wilson(failures, accepted)
    minimum = artifact['minimum_accepted']
    valid = accepted >= minimum and interval[1] is not None and interval[1] <= artifact['risk_target']
    updated = {**artifact, 'closed_loop_validated': valid, 'closed_loop_validation': {'accepted': accepted, 'insufficient': failures, 'risk_95_ci': interval, 'results_sha256': file_digest(results_file), 'human_rows_sha256': file_digest(human_rows_file), 'validated_at': now()}}
    write_json(destination, updated)
    return updated


def calibrate(config, development_rows, destination):
    rows = list(read_jsonl(development_rows))
    target = config['calibration']['risk_target']
    if config['calibration']['confidence'] != .95:
        raise ValueError('This calibration protocol currently fixes 95% Wilson intervals')
    if target is None or not 0 < target < 1:
        raise ValueError('Choose a development risk target before inspecting held-out results')
    if not rows or len({row['question_id'] for row in rows}) != len(rows):
        raise ValueError('Calibration requires one terminal proposal per development question')
    manifest_hash = file_digest(path(config['splits']['directory']) / 'manifest.json')
    dev_ids = {row['id'] for row in read_jsonl(path(config['splits']['directory']) / 'dev.jsonl')}
    identities = {digest(row['reviewer_identity']) for row in rows}
    if len(identities) != 1:
        raise ValueError('Calibrate one reviewer identity at a time')
    for row in rows:
        if row.get('split') != 'dev' or row['question_id'] not in dev_ids or row.get('split_manifest_sha256') != manifest_hash:
            raise ValueError('Calibration contains non-development or mismatched data')
        if row.get('label_source') != 'adjudicated_human' or not isinstance(row.get('human_sufficient'), bool):
            raise ValueError('Model prelabels cannot supply human calibration truth')
        if row.get('inclusion_probability') != 1:
            raise ValueError('This Wilson calibration implementation requires exhaustive labels, not weighted enrichment')
        if isinstance(row['score'], bool) or not isinstance(row['score'], (int, float)) or not math.isfinite(row['score']) or not 0 <= row['score'] <= 1:
            raise ValueError('Invalid development score')
    grid = []
    for threshold in config['calibration']['threshold_grid']:
        selected = [row for row in rows if row['score'] >= threshold]
        insufficient = sum(not row['human_sufficient'] for row in selected)
        interval = wilson(insufficient, len(selected))
        grid.append({'threshold': threshold, 'accepted': len(selected), 'coverage': len(selected) / len(rows), 'risk': insufficient / len(selected) if selected else None, 'risk_95_ci': interval, 'eligible': len(selected) >= config['calibration']['minimum_accepted'] and interval[1] <= target if selected else False})
    eligible = [row for row in grid if row['eligible']]
    if not eligible:
        write_json(destination, {'status': 'no_eligible_threshold', 'grid': grid, 'selection_split': 'dev', 'closed_loop_validated': False})
        return {'status': 'no_eligible_threshold'}
    selected = max(eligible, key=lambda row: (row['coverage'], row['threshold']))
    artifact = {'status': 'selected_requires_closed_loop_validation', 'threshold': selected['threshold'], 'risk_target': target, 'minimum_accepted': config['calibration']['minimum_accepted'], 'selection_split': 'dev', 'split_manifest_sha256': manifest_hash, 'development_labels_sha256': file_digest(development_rows), 'reviewer_identity': rows[0]['reviewer_identity'], 'prompt_version': PROMPT_VERSION, 'grid': grid, 'closed_loop_validated': False, 'not_a_finite_sample_safety_guarantee': True}
    write_json(destination, artifact)
    return artifact
