import copy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.eara.agents import Agent
from experiments.eara.common import config_load
from experiments.eara.gateway import Context, InfrastructureFailure
from experiments.eara.protocol import capabilities, validate_protocol
from experiments.eara.metrics import answer_scores


class ScriptedRouter:
    def __init__(self, responses):
        self.responses = {purpose: list(values) for purpose, values in responses.items()}
        self.calls = []

    def identity(self, role):
        return {'model': 'synthetic-' + role, 'revision': 'offline-test'}

    async def complete(self, context, role, messages, purpose):
        context.check()
        context.turns += 1
        self.calls.append(purpose)
        values = self.responses[purpose]
        value = values.pop(0) if len(values) > 1 else values[0]
        return value if isinstance(value, str) else json.dumps(value)


class SyntheticRetriever:
    def __init__(self):
        self.calls = 0

    def search(self, query):
        self.calls += 1
        return {'physical_calls': 1, 'candidate_counts': {'bm25': 1}, 'passages': [
            {'id': 'p1', 'title': 'Fictional example', 'text': 'The fictional author Ada was born in Stonehaven.'}
        ]}


class ControllerContractTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.config = config_load('experiments/configs/paper.yaml')
        self.config['eara'].update({'decompose': False, 'rewrite': False, 'requested_subquestions': 1,
                                    'plan_cache': self.temporary.name})
        self.config['retrieval']['logical_budget'] = 2
        self.events = []
        self.context = Context('synthetic-contract', {'id': 'fictional', 'question': 'Where was Ada born?'},
                               self.config, 17, self.events.append)
        self.candidate = {'answer': 'Stonehaven', 'propose': True, 'abstain': False, 'subquestion_updates': []}
        self.positive = {'score': 0.8, 'sufficient': True, 'passage_ids': ['p1'], 'missing_links': []}
        self.negative = {'score': 0.1, 'sufficient': False, 'passage_ids': ['p1'], 'missing_links': ['birthplace']}

    async def execute(self, judgment, candidate=None):
        router = ScriptedRouter({'agent': [candidate or self.candidate], 'judge': judgment})
        retriever = SyntheticRetriever()
        agent = Agent(self.config, router, retriever, {'method': 'eara'}, 0.5)
        result = await agent.run(self.context)
        return result, router, retriever

    async def test_insufficient_evidence_never_forces_answer(self):
        result, router, retriever = await self.execute([self.negative])
        self.assertEqual(result['status'], 'abstained')
        self.assertIsNone(result['answer'])
        self.assertEqual(result['logical_retrieval_calls'], 2)
        self.assertEqual(retriever.calls, 1)
        self.assertEqual(router.calls.count('judge'), 2)

    async def test_rejection_then_acceptance(self):
        result, _, _ = await self.execute([self.negative, self.positive])
        self.assertEqual(result['answer'], 'Stonehaven')
        self.assertFalse(result['proposals'][0]['accepted'])
        self.assertTrue(result['proposals'][-1]['is_terminal'])

    async def test_empty_candidate_is_not_released(self):
        candidate = {**self.candidate, 'answer': '', 'propose': False}
        result, router, _ = await self.execute([self.positive], candidate)
        self.assertEqual(result['status'], 'abstained')
        self.assertNotIn('judge', router.calls)

    async def test_malformed_judge_retries_once_then_abstains(self):
        result, router, _ = await self.execute(['not JSON'])
        self.assertEqual(result['status'], 'abstained')
        self.assertEqual(router.calls.count('judge'), 2)
        self.assertEqual(result['termination_reason'], 'reviewer_invalid_after_one_repair')

    async def test_unknown_support_ids_are_rejected(self):
        result, _, _ = await self.execute([{**self.positive, 'passage_ids': ['not-retrieved']}])
        self.assertEqual(result['status'], 'abstained')

    async def test_positive_score_without_citations_is_rejected(self):
        result, _, _ = await self.execute([{**self.positive, 'passage_ids': []}])
        self.assertEqual(result['status'], 'abstained')

    async def test_score_not_binary_verdict_controls_gate(self):
        result, _, _ = await self.execute([{**self.positive, 'sufficient': False}])
        self.assertEqual(result['status'], 'answered')

    async def test_hard_limit_does_not_release_unjudged_candidate(self):
        self.config['execution']['max_controller_turns'] = 1
        result, _, _ = await self.execute([self.positive])
        self.assertEqual(result['status'], 'abstained')
        self.assertIsNone(result['answer'])

    async def test_invalid_plan_falls_back_and_is_frozen(self):
        self.config['eara'].update({'decompose': True, 'requested_subquestions': 2})
        router = ScriptedRouter({'planner': ['invalid']})
        agent = Agent(self.config, router, SyntheticRetriever(), {'method': 'eara'}, 0.5)
        first = await agent.plan(self.context)
        second = await agent.plan(self.context)
        self.assertEqual(first[0]['question'], self.context.question['question'])
        self.assertEqual(first, second)
        self.assertEqual(router.calls, ['planner', 'planner'])
        self.assertEqual(len(list(Path(self.temporary.name).glob('*.json'))), 1)

    def test_unknown_baseline_cannot_be_renamed(self):
        with self.assertRaises(InfrastructureFailure):
            Agent(self.config, None, None, {'method': 'tir'}, None)

    def test_unsupported_sampling_fails_explicitly(self):
        self.config['eara']['candidate_samples'] = 5
        with self.assertRaisesRegex(ValueError, 'not implemented'):
            validate_protocol(self.config, {'method': 'eara'})

    def test_condition_sampling_override_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'not implemented'):
            validate_protocol(self.config, {'method': 'eara', 'candidate_samples': 5})

    def test_invalid_thresholds(self):
        for threshold in [None, True, float('nan'), float('inf'), -0.1, 1.1]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                Agent(self.config, None, None, {'method': 'eara'}, threshold)

    def test_paper_retrieval_protocol(self):
        settings = copy.deepcopy(self.config)
        settings['retrieval']['logical_budget'] = 10
        validate_protocol(settings, {'method': 'eara'}, 'paper')
        settings['retrieval']['top_k'] = 3
        with self.assertRaises(ValueError):
            validate_protocol(settings, {'method': 'eara'}, 'paper')

    def test_main_table_cannot_silently_use_hybrid(self):
        self.config['retrieval']['logical_budget'] = 10
        condition = {'method': 'eara', 'retrieval_weights': [5, 5]}
        with self.assertRaises(ValueError):
            validate_protocol(self.config, condition, 'paper')
        validate_protocol(self.config, {**condition, 'family': 'retrieval'}, 'paper')

    def test_capabilities_do_not_claim_completed_experiments(self):
        self.assertFalse(capabilities()['formal_results_included'])
        self.assertIn('reviewer_noninferiority_and_power', capabilities()['unsupported'])

    def test_substring_match_is_not_exact_match(self):
        self.assertEqual(answer_scores('Not Paris', ['Paris'])['em'], 0.0)
        self.assertEqual(answer_scores('Norway', ['No'])['em'], 0.0)
        self.assertEqual(answer_scores('Paris.', ['Paris'])['em'], 1.0)


if __name__ == '__main__':
    unittest.main()
