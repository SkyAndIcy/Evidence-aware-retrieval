from collections import Counter, defaultdict
import math
import random
import re
import statistics
import unicodedata


def normalize_answer(text, language='en'):
    normalized = ''.join(character for character in str(text).casefold() if not unicodedata.category(character).startswith('P'))
    if language == 'en':
        normalized = re.sub(r'\b(a|an|the)\b', ' ', normalized)
    return ' '.join(normalized.split())


def answer_scores(prediction, references, language='en'):
    if prediction is None:
        return {'em': 0.0, 'f1': 0.0}
    predicted = normalize_answer(prediction, language)
    exact, f1 = 0.0, 0.0
    for reference in references:
        gold = normalize_answer(reference, language)
        exact = max(exact, float(predicted == gold))
        if predicted in {'yes', 'no', 'noanswer'} or gold in {'yes', 'no', 'noanswer'}:
            score = float(predicted == gold)
        else:
            predicted_tokens = list(predicted.replace(' ', '')) if language == 'zh' else predicted.split()
            gold_tokens = list(gold.replace(' ', '')) if language == 'zh' else gold.split()
            overlap = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
            score = 2 * overlap / (len(predicted_tokens) + len(gold_tokens)) if predicted_tokens or gold_tokens else 1.0
        f1 = max(f1, score)
    return {'em': exact, 'f1': f1}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    fraction = successes / total
    center = (fraction + z * z / (2 * total)) / (1 + z * z / total)
    radius = z * math.sqrt(fraction * (1 - fraction) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [max(0.0, center - radius), min(1.0, center + radius)]


def agreement(labels_a, labels_b, weights=None):
    if len(labels_a) != len(labels_b):
        raise ValueError('Rater populations must match')
    weights = weights if weights is not None else [1.0] * len(labels_a)
    if len(weights) != len(labels_a) or any(not isinstance(value, bool) for value in [*labels_a, *labels_b]):
        raise ValueError('Agreement requires aligned boolean labels')
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError('Sampling weights must be positive and finite')
    total = sum(weights)
    if not total:
        return {'count': 0, 'kappa': None, 'raw_agreement': None}
    matrix = [[0.0, 0.0], [0.0, 0.0]]
    for first, second, weight in zip(labels_a, labels_b, weights):
        matrix[int(first)][int(second)] += weight
    observed = (matrix[0][0] + matrix[1][1]) / total
    first_positive = (matrix[1][0] + matrix[1][1]) / total
    second_positive = (matrix[0][1] + matrix[1][1]) / total
    chance = first_positive * second_positive + (1 - first_positive) * (1 - second_positive)
    return {'count': len(labels_a), 'weighted_count': total, 'confusion_rows_a_columns_b': matrix, 'raw_agreement': observed, 'prevalence_a': first_positive, 'prevalence_b': second_positive, 'kappa': None if math.isclose(chance, 1.0, abs_tol=1e-12) else (observed - chance) / (1 - chance)}


def classifier_metrics(gold, predicted, weights=None):
    comparison = agreement(gold, predicted, weights)
    if not gold:
        return {**comparison, 'macro_f1': None, 'false_acceptance_rate': None, 'false_rejection_rate': None, 'accepted_precision': None}
    negative, positive = comparison['confusion_rows_a_columns_b']
    true_negative, false_positive = negative
    false_negative, true_positive = positive
    f1_positive = ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative) or 0.0
    f1_negative = ratio(2 * true_negative, 2 * true_negative + false_positive + false_negative) or 0.0
    return {**comparison, 'macro_f1': (f1_positive + f1_negative) / 2, 'false_acceptance_rate': ratio(false_positive, false_positive + true_negative), 'false_rejection_rate': ratio(false_negative, false_negative + true_positive), 'accepted_precision': ratio(true_positive, true_positive + false_positive)}


HARD_LIMIT_REASONS = frozenset({'retrieval_budget_exhausted', 'wall_clock_limit', 'controller_turn_limit', 'generation_token_limit'})


def summarize(records):
    count = len(records)
    answered = [record for record in records if record['status'] == 'answered']
    complete = [record for record in records if record['status'] in {'answered', 'abstained'}]
    failures = [record for record in records if record['status'] == 'execution_failure']
    exact = sum(record.get('em', 0.0) for record in answered)
    incomplete_costs = sum(not record.get('cost_including_prior_attempts', {}).get('complete_attempt_records', True) for record in records)
    abstention_reasons = Counter(record.get('termination_reason') or 'unspecified' for record in records if record['status'] == 'abstained')
    return {
        'assigned': count,
        'answered': len(answered),
        'abstained': sum(record['status'] == 'abstained' for record in records),
        'hard_limit_terminations': sum(abstention_reasons.get(reason, 0) for reason in HARD_LIMIT_REASONS),
        'abstention_reasons': dict(abstention_reasons),
        'execution_failures': len(failures),
        'pending': sum(record['status'] in {'assigned', 'running'} for record in records),
        'completed': len(complete),
        'coverage_all_assigned': ratio(len(answered), count),
        'yield_em_all_assigned': ratio(exact, count),
        'selective_accuracy_em': ratio(exact, len(answered)),
        'answer_error_risk': ratio(len(answered) - exact, len(answered)),
        'evidence_insufficiency_risk': None,
        'f1_all_assigned': ratio(sum(record.get('f1', 0.0) for record in answered), count),
        'failure_rate': ratio(len(failures), count),
        'mean_logical_retrieval_calls_all_assigned': ratio(sum(record.get('cost_including_prior_attempts', record).get('logical_retrieval_calls', 0) for record in records), count) if not incomplete_costs else None,
        'mean_latency_completed': statistics.mean(record.get('cost_including_prior_attempts', record)['elapsed_seconds'] for record in complete) if complete and not incomplete_costs else None,
        'questions_with_incomplete_prior_attempt_costs': incomplete_costs,
        'incomplete_run': any(record['status'] in {'assigned', 'running'} for record in records),
        'metrics_provisional_if_incomplete': True,
    }


def paired_cluster_bootstrap(first, second, repeats=2000, seed=17):
    first_map = {(record['question_id'], record['seed']): record for record in first}
    second_map = {(record['question_id'], record['seed']): record for record in second}
    if len(first_map) != len(first) or len(second_map) != len(second):
        raise ValueError('Duplicate question/seed records')
    if set(first_map) != set(second_map):
        raise ValueError('Paired comparison requires identical assigned question/seed sets')
    clusters = defaultdict(list)
    for key, original in first_map.items():
        comparator = second_map[key]
        if original.get('split_manifest_sha256') != comparator.get('split_manifest_sha256'):
            raise ValueError('Split manifests differ')
        if original['status'] in {'assigned', 'running'} or comparator['status'] in {'assigned', 'running'}:
            raise ValueError('Finish both runs before inference')
        cluster = original.get('cluster_id', original['question_id'])
        clusters[cluster].append((original, comparator))
    groups = list(clusters.values())
    if not groups:
        raise ValueError('No paired observations')
    rng = random.Random(seed)
    differences = []
    for _ in range(repeats):
        paired = [pair for _ in groups for pair in rng.choice(groups)]
        differences.append(statistics.mean(second_record.get('em', 0) - first_record.get('em', 0) for first_record, second_record in paired))
    differences.sort()
    common_completed = [(first_map[key], second_map[key]) for key in first_map if first_map[key]['status'] in {'answered', 'abstained'} and second_map[key]['status'] in {'answered', 'abstained'}]
    return {'paired_assignments': len(first_map), 'clusters': len(groups), 'delta_yield_em': statistics.mean(second_map[key].get('em', 0) - first_map[key].get('em', 0) for key in first_map), 'cluster_bootstrap_95_ci': [differences[int(.025 * repeats)], differences[min(repeats - 1, int(.975 * repeats))]], 'common_completed_count': len(common_completed), 'delta_yield_common_completed_diagnostic': statistics.mean(second_record.get('em', 0) - first_record.get('em', 0) for first_record, second_record in common_completed) if common_completed else None, 'bootstrap_seed': seed, 'repeats': repeats}


TRANSFER_EFFECT_METRICS = ('yield_em_all_assigned', 'coverage_all_assigned', 'selective_accuracy_em', 'answer_error_risk', 'f1_all_assigned', 'mean_logical_retrieval_calls_all_assigned')


def transfer_effects(conditions, metric_keys=TRANSFER_EFFECT_METRICS):
    """Component gains and interaction for one transfer host (Sec. 3.3, Eq. 3-4):
    dC = U10 - U00, dA = U01 - U00, interaction = U11 - U10 - U01 + U00, where
    the subscript bits are the CRP/CAC switches. The four inputs are result
    record lists from the same split and identical assigned question/seed sets;
    metric values that are undefined (None) propagate as undefined rather than
    being silently treated as zero."""
    expected = {'u00', 'u10', 'u01', 'u11'}
    if set(conditions) != expected:
        raise ValueError('Provide exactly the four CRP/CAC conditions: ' + ', '.join(sorted(expected)))
    assigned = None
    for tag, records in conditions.items():
        keys = {(record.get('question_id'), record.get('seed')) for record in records}
        if len(keys) != len(records):
            raise ValueError(tag + ': duplicate question/seed records')
        if any(record.get('status') in {'assigned', 'running'} for record in records):
            raise ValueError(tag + ': finish all four runs before computing effects')
        if assigned is None:
            assigned = keys
        elif keys != assigned:
            raise ValueError('The four transfer conditions must cover identical assigned question/seed sets')
    summaries = {tag: summarize(records) for tag, records in conditions.items()}
    effects = {}
    for key in metric_keys:
        values = {tag: summaries[tag].get(key) for tag in expected}
        if any(value is None for value in values.values()):
            effects[key] = {**values, 'delta_crp': None, 'delta_cac': None, 'interaction': None, 'note': 'undefined for at least one condition'}
        else:
            effects[key] = {**values, 'delta_crp': values['u10'] - values['u00'], 'delta_cac': values['u01'] - values['u00'], 'interaction': values['u11'] - values['u10'] - values['u01'] + values['u00']}
    return {'paired_assignments': len(assigned), 'metric_definitions': {'delta_crp': 'U10-U00', 'delta_cac': 'U01-U00', 'interaction': 'U11-U10-U01+U00'}, 'effects': effects, 'incomplete_costs_in_any_condition': any(summaries[tag].get('questions_with_incomplete_prior_attempt_costs') for tag in expected)}
