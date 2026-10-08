import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def inventory(root):
    return {str(filename.relative_to(root)): hashlib.sha256(filename.read_bytes()).hexdigest()
            for filename in sorted(root.rglob('*')) if filename.is_file()
            and '__pycache__' not in filename.parts and filename.name != '.DS_Store'}


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(root):
    root = root.resolve()
    for name in ['core', 'data', 'methods', 'prompts']:
        module = ModuleType(name)
        module.__path__ = [str(root / name)]
        sys.modules[name] = module
    for name, attributes in {
        'core.llm': {'LLM': object, 'ALL_TOOLS': []},
        'core.retriever': {'Retriever': object},
        'data.datasets': {'Example': object},
        'methods.baselines': {'REACT_SYSTEM': 'synthetic', 'IRCoT_SYSTEM': 'synthetic'},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module
    legacy = load_module('legacy_v4_audit', root / 'methods/eara_v4.py')
    metrics = load_module('legacy_metrics_audit', root / 'evaluation/metrics.py')

    class ScriptedLLM:
        def __init__(self, outputs):
            self.outputs = iter(outputs)

        def chat(self, *args, **kwargs):
            return next(self.outputs)

    example = SimpleNamespace(question='Synthetic audit question')
    unfinished = legacy.run_eara_v4(ScriptedLLM([{'content': 'Still reasoning.'}]), None, example, max_steps=1)
    forced = legacy.run_eara_v4(ScriptedLLM([{'content': 'FINAL ANSWER: unsupported'}, {'content': '0.0'}]),
                               None, example, max_steps=1, max_rejects=0)
    parsed = legacy.evidence_sufficiency(ScriptedLLM([{'content': 'Step 1: Missing facts. Sufficiency score: 0.0'}]),
                                       example.question, 'unsupported', '')
    assert unfinished['abstained'] is False
    assert forced['abstained'] is False
    assert parsed == 1.0
    assert metrics.is_correct('Not Paris', 'Paris')
    directories = [root.parent / name for name in ['eara_release', 'eara_release 2', 'eara_release 3']]
    inventories = [inventory(directory) for directory in directories]
    return {
        'scope': 'isolated_synthetic_regression_probe_not_formal_experiments',
        'network_calls': 0,
        'credential_configuration_parsed': False,
        'release_copies_identical': all(value == inventories[0] for value in inventories),
        'file_count_per_release': len(inventories[0]),
        'reproduced': {
            'unfinished_reasoning_released_without_judgment': {'answer': unfinished['answer'], 'abstained': unfinished['abstained']},
            'max_rejects_bypasses_low_sufficiency_gate': {'answer': forced['answer'], 'abstained': forced['abstained']},
            'first_number_parser_reads_step_number_as_support': {'intended_score': 0.0, 'parsed_score': parsed},
            'substring_metric_accepts_not_paris': metrics.is_correct('Not Paris', 'Paris'),
        },
        'static_findings': {
            'calibration_import_target_exists': (root / 'methods/metrics.py').exists(),
            'source_hashes': {name: inventories[0][name] for name in [
                'run.py', 'methods/eara.py', 'methods/eara_v4.py', 'core/retriever.py', 'evaluation/metrics.py']},
        },
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read-only offline probes of the supplied legacy EARA release. No real models.')
    parser.add_argument('--legacy-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    report = audit(options.legacy_root)
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
