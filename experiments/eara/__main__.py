import argparse
import asyncio
import json
from pathlib import Path

from .agents import IMPLEMENTATIONS, UNIMPLEMENTED
from .audit import audit_agreement, calibrate, calibration_rows, export_proposals, machine_label, make_blind_audit, model_agreement, validate_closed_loop
from .common import config_load, path, read_jsonl, write_json
from .data import DATASET_NAMES, corpus_import, download, freeze_splits, normalize_dataset
from .gateway import InfrastructureFailure, Router
from .metrics import paired_cluster_bootstrap, transfer_effects
from .retrieval import build_bm25, build_dense
from .runner import make_grid, run_study
from .protocol import capabilities


async def doctor(config):
    router = Router(config, path(config['output']['runs']) / '.doctor_ledger.jsonl')
    identities = {}
    try:
        for role in config['models']:
            try:
                identity = router.identity(role)
                identities[role] = {'configured': True, 'model': identity['model'], 'revision_frozen': bool(identity['revision']), 'pending_protocol_change': config['models'][role].get('protocol_change')}
            except InfrastructureFailure as error:
                identities[role] = {'configured': False, 'reason': str(error)}
    finally:
        await router.close()
    return {'api_concurrency_cap': config['execution']['api_concurrency'], 'top_k': config['retrieval']['top_k'], 'logical_retrieval_budget': config['retrieval']['logical_budget'], 'dataset_files': {name: path(spec['input']).exists() for name, spec in config['datasets'].items()}, 'corpus_present': path(config['retrieval']['corpus_db']).exists(), 'bm25_index_present': (path(config['retrieval']['bm25_index']) / 'metadata.json').exists(), 'frozen_splits_present': (path(config['splits']['directory']) / 'manifest.json').exists(), 'models': identities, 'implemented_methods': IMPLEMENTATIONS, 'unimplemented_methods': UNIMPLEMENTED, 'human_labels_generated': 0, 'calibration_pending': not bool(config['eara']['threshold_artifact']), 'no_API_requests_made': True}


def arguments():
    parser = argparse.ArgumentParser(description='Evidence-aware retrieval experiments. Diagnostics and paper results are kept separate.')
    parser.add_argument('--config', default='experiments/configs/paper.yaml')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor')
    commands.add_parser('capabilities')
    fetch = commands.add_parser('download-corpus')
    fetch.add_argument('--output', required=True)
    fetch.add_argument('--max-gib', type=float, required=True)
    fetch.add_argument('--sha256')
    corpus = commands.add_parser('import-corpus')
    corpus.add_argument('--input', required=True)
    corpus.add_argument('--output')
    corpus.add_argument('--identity')
    corpus.add_argument('--max-rows', type=int)
    lexical = commands.add_parser('index-bm25')
    lexical.add_argument('--allow-large-index', action='store_true')
    dense = commands.add_parser('index-dense')
    dense.add_argument('--allow-model-download', action='store_true')
    dense.add_argument('--device', default='cpu')
    dataset = commands.add_parser('normalize-dataset')
    dataset.add_argument('--name', choices=sorted(DATASET_NAMES), required=True)
    dataset.add_argument('--input', required=True)
    dataset.add_argument('--source-url', required=True)
    dataset.add_argument('--parent-mapping')
    commands.add_parser('freeze-splits')
    grid = commands.add_parser('make-grid')
    grid.add_argument('--family', choices=['main', 'decomposition', 'planner_crossed', 'rewriting', 'rewriting_control', 'retrieval', 'transfer'], required=True)
    grid.add_argument('--output', required=True)
    for name in ['run', 'freeze-plans']:
        run = commands.add_parser(name)
        selection = run.add_mutually_exclusive_group(required=True)
        selection.add_argument('--method', choices=sorted(IMPLEMENTATIONS))
        selection.add_argument('--grid')
        run.add_argument('--condition-id')
        run.add_argument('--generator')
        run.add_argument('--planner')
        run.add_argument('--split', choices=['dev', 'test'], default='dev')
        run.add_argument('--mode', choices=['pilot', 'development', 'paper'], default='pilot')
        run.add_argument('--seeds', type=int, nargs='+')
        run.add_argument('--limit', type=int)
        run.add_argument('--retry-failures', action='store_true')
        run.add_argument('--additional-request-budget', type=int, default=0)
        run.add_argument('--confirm-api-use', action='store_true')
    export = commands.add_parser('export-proposals')
    export.add_argument('--results', nargs='+', required=True)
    export.add_argument('--output', required=True)
    label = commands.add_parser('label-models')
    label.add_argument('--tuples', required=True)
    label.add_argument('--roles', nargs='+', default=['frontier', 'small_reviewer'])
    label.add_argument('--output', required=True)
    label.add_argument('--confirm-api-use', action='store_true')
    agreement_parser = commands.add_parser('model-agreement')
    agreement_parser.add_argument('--labels', required=True)
    agreement_parser.add_argument('--frontier', default='frontier')
    agreement_parser.add_argument('--small', default='small_reviewer')
    agreement_parser.add_argument('--frontier-threshold', type=float, required=True)
    agreement_parser.add_argument('--small-threshold', type=float, required=True)
    agreement_parser.add_argument('--output', required=True)
    audit = commands.add_parser('make-audit')
    audit.add_argument('--tuples', required=True)
    audit.add_argument('--labels', required=True)
    audit.add_argument('--output', required=True)
    audit.add_argument('--frontier', default='frontier')
    audit.add_argument('--small', default='small_reviewer')
    audit.add_argument('--frontier-threshold', type=float, required=True)
    audit.add_argument('--small-threshold', type=float, required=True)
    audit.add_argument('--rejected-probability', type=float, default=.2)
    audit_stats = commands.add_parser('audit-stats')
    audit_stats.add_argument('--mapping', required=True)
    audit_stats.add_argument('--human-a', required=True)
    audit_stats.add_argument('--human-b', required=True)
    audit_stats.add_argument('--adjudicated')
    audit_stats.add_argument('--labels', required=True)
    audit_stats.add_argument('--frontier', default='frontier')
    audit_stats.add_argument('--small', default='small_reviewer')
    audit_stats.add_argument('--frontier-threshold', type=float, required=True)
    audit_stats.add_argument('--small-threshold', type=float, required=True)
    audit_stats.add_argument('--output', required=True)
    rows = commands.add_parser('calibration-data')
    rows.add_argument('--mapping', required=True)
    rows.add_argument('--human', required=True)
    rows.add_argument('--labels', required=True)
    rows.add_argument('--reviewer', required=True)
    rows.add_argument('--output', required=True)
    calibration = commands.add_parser('calibrate')
    calibration.add_argument('--dev-labels', required=True)
    calibration.add_argument('--output', required=True)
    validate = commands.add_parser('validate-operating-point')
    validate.add_argument('--artifact', required=True)
    validate.add_argument('--results', required=True)
    validate.add_argument('--human-rows', required=True)
    validate.add_argument('--output', required=True)
    compare = commands.add_parser('compare')
    compare.add_argument('--first', required=True)
    compare.add_argument('--second', required=True)
    compare.add_argument('--output', required=True)
    effects = commands.add_parser('transfer-effects')
    effects.add_argument('--u00', required=True, help='results.jsonl for host alone (CRP off, CAC off)')
    effects.add_argument('--u10', required=True, help='results.jsonl for host+CRP (CAC off)')
    effects.add_argument('--u01', required=True, help='results.jsonl for host+CAC (CRP off)')
    effects.add_argument('--u11', required=True, help='results.jsonl for host+CRP+CAC')
    effects.add_argument('--output', required=True)
    return parser.parse_args()


def main():
    args = arguments()
    config = config_load(args.config)
    command = args.command
    if command == 'doctor':
        result = asyncio.run(doctor(config))
    elif command == 'capabilities':
        result = capabilities()
    elif command == 'download-corpus':
        download(config['retrieval']['corpus_download_url'], args.output, args.max_gib, args.sha256)
        result = {'downloaded': args.output}
    elif command == 'import-corpus':
        result = corpus_import(args.input, args.output or path(config['retrieval']['corpus_db']), args.identity or config['retrieval']['corpus_identity'], args.max_rows)
    elif command == 'index-bm25':
        result = build_bm25(path(config['retrieval']['corpus_db']), path(config['retrieval']['bm25_index']), config['retrieval']['bm25'], args.allow_large_index)
    elif command == 'index-dense':
        result = build_dense(path(config['retrieval']['corpus_db']), path(config['retrieval']['dense_index']), config['retrieval']['dense'], args.allow_model_download, args.device)
    elif command == 'normalize-dataset':
        count = normalize_dataset(args.name, args.input, path(config['datasets'][args.name]['input']), args.source_url, args.parent_mapping)
        result = {'normalized_questions': count, 'gold_contexts_used_for_retrieval': False}
    elif command == 'freeze-splits':
        result = freeze_splits(config)
    elif command == 'make-grid':
        result = make_grid(config, args.family)
        write_json(args.output, result)
    elif command in {'run', 'freeze-plans'}:
        if args.grid:
            matches = [row for row in json.loads(Path(args.grid).read_text()) if row['id'] == args.condition_id]
            if len(matches) != 1:
                raise ValueError('Specify exactly one --condition-id from the grid')
            condition = matches[0]
        else:
            condition = {'id': args.method, 'method': args.method, 'family': 'main'}
        if args.generator:
            condition['generator'] = args.generator
        if args.planner:
            condition['planner'] = args.planner
        if not args.confirm_api_use:
            result = {'dry_run_only': True, 'condition': condition, 'mode': args.mode, 'split': args.split, 'api_request_cap': config['execution']['max_api_requests_per_run'], 'next_step': 'Review data/model readiness, then add --confirm-api-use to authorize billable requests'}
        else:
            directory, summary = asyncio.run(run_study(config, condition, args.split, args.mode, args.seeds or config['study']['inference_seeds'], args.limit, args.retry_failures, command == 'freeze-plans', additional_request_budget=args.additional_request_budget))
            result = {'directory': str(directory), **summary}
    elif command == 'export-proposals':
        result = {'frozen_proposals': export_proposals(args.results, args.output)}
    elif command == 'label-models':
        if not args.confirm_api_use:
            result = {'dry_run_only': True, 'label_source': 'model', 'human_labels_created': 0}
        else:
            result = asyncio.run(machine_label(config, args.tuples, args.roles, args.output))
    elif command == 'make-audit':
        result = make_blind_audit(args.tuples, args.labels, args.output, args.frontier, args.small, args.frontier_threshold, args.small_threshold, args.rejected_probability, config['audit']['sampling_seed'])
    elif command == 'model-agreement':
        result = model_agreement(args.labels, args.frontier, args.small, args.frontier_threshold, args.small_threshold)
        write_json(args.output, result)
    elif command == 'audit-stats':
        result = audit_agreement(args.mapping, args.human_a, args.human_b, args.labels, args.frontier, args.small, args.frontier_threshold, args.small_threshold, args.adjudicated)
        write_json(args.output, result)
    elif command == 'calibration-data':
        result = {'rows': calibration_rows(args.mapping, args.human, args.labels, args.reviewer, args.output)}
    elif command == 'calibrate':
        result = calibrate(config, args.dev_labels, args.output)
    elif command == 'validate-operating-point':
        result = validate_closed_loop(args.artifact, args.results, args.human_rows, args.output)
    elif command == 'compare':
        result = paired_cluster_bootstrap(list(read_jsonl(args.first)), list(read_jsonl(args.second)))
        write_json(args.output, result)
    elif command == 'transfer-effects':
        conditions = {'u00': list(read_jsonl(args.u00)), 'u10': list(read_jsonl(args.u10)), 'u01': list(read_jsonl(args.u01)), 'u11': list(read_jsonl(args.u11))}
        result = transfer_effects(conditions)
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, InfrastructureFailure, ImportError) as error:
        raise SystemExit(type(error).__name__ + ': ' + str(error)) from None
