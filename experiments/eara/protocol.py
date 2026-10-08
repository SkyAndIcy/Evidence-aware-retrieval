import math


def validate_protocol(config, condition, mode='pilot'):
    samples = condition.get('candidate_samples', config['eara'].get('candidate_samples', 1))
    if type(samples) is not int or samples != 1:
        raise ValueError('Only candidate_samples=1 is implemented; five-sample sensitivity is not implemented')
    if config['execution'].get('format_retries') != 1:
        raise ValueError('The JSON contract implements exactly one format-only retry')
    maximum = config['eara']['adaptive_max']
    if type(maximum) is not int or not 1 <= maximum <= 6:
        raise ValueError('adaptive_max must be an integer between 1 and 6')
    count = condition.get('subquestions', config['eara']['requested_subquestions'])
    if count != 'adaptive' and (type(count) is not int or not 1 <= count <= maximum):
        raise ValueError('Subquestion count must be adaptive or an integer within adaptive_max')
    if mode == 'paper':
        if config['retrieval']['top_k'] != 5 or config['retrieval']['logical_budget'] != 10:
            raise ValueError('The current paper protocol requires top-5 and a logical retrieval budget of 10')
        if condition.get('family', 'main') == 'main':
            weights = condition.get('retrieval_weights')
            if config['retrieval']['backend'] != 'bm25' or (weights is not None and weights != [10, 0]):
                raise ValueError('The main table uses BM25s only; label backend variations as retrieval ablations')


def validate_threshold(threshold):
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('CAC requires a finite threshold in [0,1]')


def capabilities():
    return {
        'status': 'implementation_inventory_not_experimental_validation',
        'core': {
            'crp': 'implemented: dependency plan, evidence-grounded rewrite, frozen paired plans',
            'cac': 'implemented: candidate support score, continue retrieval or abstain',
            'retrieval': 'implemented: BM25s, dense optional dependency, eleven weighted-RRF settings',
            'datasets': 'implemented: five English dataset normalizers and frozen splits',
            'execution': 'implemented: global request limit, resume, provenance and cost logs',
            'human_review': 'implemented: blind task export/import; actual humans and labels required',
            'decomposition_grids': 'implemented: same-model grid and the Sec. 6.1 planner-crossed grid (generator/reviewer held fixed)',
            'rewriting_grids': 'implemented: D/W factorial with CAC held fixed, plus the Sec. 6.2 CAC-disabled control grid',
            'transfer_analysis': 'implemented: four-condition grids and Eq. 3-4 component-gain/interaction computation (transfer-effects command)',
            'hard_limit_reporting': 'implemented: hard-limit terminations and abstention reasons summarized separately',
        },
        'unsupported': {
            'candidate_samples_5': 'not implemented; non-one values fail explicitly',
            'chinese_translation_and_ccmor': 'not integrated in the five-English-dataset runner',
            'perturbation_study': 'not integrated',
            'rewrite_human_quality_statistics': 'not integrated',
            'reviewer_noninferiority_and_power': 'not integrated; kappa is not a substitute',
            's2g_rag_chainrag': 'strong-accept plan suggestions, not implemented baselines',
            'tir': 'method identity unresolved',
            'r1_base_instruct': 'checkpoint identity unresolved',
        },
        'external_prerequisites': [
            'licensed corpus and full index', 'benchmark source files and deduplication mapping',
            'immutable model checkpoints and serving endpoints', 'independent human annotations',
            'development-selected and closed-loop-validated threshold',
            'official baseline reproduction/fidelity verification',
        ],
        'formal_results_included': False,
    }
