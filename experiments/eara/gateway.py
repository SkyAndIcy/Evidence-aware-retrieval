import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx
import yaml

from .common import digest, now, read_jsonl


class HardLimit(Exception):
    pass


class InfrastructureFailure(Exception):
    pass


class RateLimited(InfrastructureFailure):
    pass


class RunBudgetExceeded(InfrastructureFailure):
    pass


class Context:
    def __init__(self, job_id, question, config, seed, event_sink):
        self.job_id = job_id
        self.question = question
        self.config = config
        self.seed = seed
        self.events = event_sink
        self.started = time.monotonic()
        self.deadline = self.started + config['execution']['per_question_seconds']
        self.turns = 0
        self.generated_upper_bound = 0
        self.usage = Counter()
        self.missing_usage_calls = 0
        self.logical_retrieval_calls = 0
        self.physical_retrieval_calls = 0
        self.retrieval_cache = {}
        self.evidence = {}
        self.proposals = []
        self.frozen_plan_cost = {}
        self.frozen_plan_reused = False

    def check(self):
        if time.monotonic() >= self.deadline:
            raise HardLimit('wall_clock_limit')
        if self.turns >= self.config['execution']['max_controller_turns']:
            raise HardLimit('controller_turn_limit')
        if self.generated_upper_bound >= self.config['execution']['max_generated_tokens_per_question']:
            raise HardLimit('generation_token_limit')

    def event(self, kind, **fields):
        self.events({'kind': kind, 'job_id': self.job_id, 'timestamp': now(), **fields})

    def view(self):
        settings = self.config['retrieval']
        selected = list(self.evidence.values())[-settings['evidence_max_passages']:]
        return [{'id': item['id'], 'title': item['title'], 'text': item['text'][:settings['evidence_max_characters_per_passage']]} for item in selected]

    def metrics(self):
        return {'logical_retrieval_calls': self.logical_retrieval_calls, 'physical_retrieval_calls': self.physical_retrieval_calls, 'model_calls_by_purpose': dict(self.usage), 'frozen_initial_plan_cost': self.frozen_plan_cost, 'frozen_plan_reused': self.frozen_plan_reused, 'missing_usage_calls': self.missing_usage_calls, 'generation_tokens_upper_bound': self.generated_upper_bound, 'elapsed_seconds': round(time.monotonic() - self.started, 3)}


class Router:
    def __init__(self, config, ledger_path, client=None):
        self.config = config
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        credential_filename = os.environ.get(config['study']['credentials_file_env'])
        self.external = yaml.safe_load(Path(credential_filename).read_text()) if credential_filename else {'models': {}}
        self.secrets = []
        for item in self.external['models'].values():
            if isinstance(item, dict):
                values = item.get('api_key', [])
                self.secrets.extend(values if isinstance(values, list) else [values])
        for spec in config['models'].values():
            if spec.get('api_key_env') and os.environ.get(spec['api_key_env']):
                self.secrets.append(os.environ[spec['api_key_env']])
        self.secrets = [value for value in self.secrets if isinstance(value, str) and value]
        self.semaphore = asyncio.Semaphore(config['execution']['api_concurrency'])
        self.ledger_lock = asyncio.Lock()
        self.request_count = sum(record['kind'] == 'request_started' for record in read_jsonl(self.ledger_path)) if self.ledger_path.exists() else 0
        self.cooldown = {}
        self.owns_client = client is None
        self.client = client or httpx.AsyncClient(limits=httpx.Limits(max_connections=config['execution']['api_concurrency']), follow_redirects=False)
        self.active = 0
        self.peak = 0

    def sanitize(self, value):
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        for secret in sorted(self.secrets, key=len, reverse=True):
            encoded = encoded.replace(secret, '[REDACTED]')
        return json.loads(encoded)

    def resolve(self, role):
        if role not in self.config['models']:
            raise InfrastructureFailure('unknown_model_role:' + role)
        spec = self.config['models'][role]
        if spec['transport'] == 'friday':
            external = self.external['models'].get(spec['alias'])
            if not isinstance(external, dict):
                raise InfrastructureFailure('credentials_not_configured_for:' + role)
            base = external['base_url']
            keys = external['api_key']
            model = external['model_name']
        else:
            base = os.environ.get(spec['endpoint_env'])
            if not base:
                raise InfrastructureFailure('missing_endpoint_environment:' + spec['endpoint_env'])
            keys = [os.environ.get(spec.get('api_key_env', ''), '')]
            model = spec['model']
        parsed = urlsplit(base)
        if parsed.username or parsed.query or not parsed.hostname:
            raise InfrastructureFailure('invalid_endpoint')
        if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in {'localhost', '127.0.0.1', '::1'}):
            raise InfrastructureFailure('HTTPS_required_except_loopback')
        keys = keys if isinstance(keys, list) else [keys]
        if not keys or any(not isinstance(key, str) for key in keys):
            raise InfrastructureFailure('invalid_credential_list')
        revision = os.environ.get(spec.get('revision_env', ''), spec.get('revision'))
        return {'base_url': base.rstrip('/'), 'model': model, 'keys': keys, 'revision': revision, 'temperature': spec.get('temperature', self.config['execution']['temperature']), 'request_options': spec.get('request_options', {}), 'seed_supported': spec.get('seed_supported', False)}

    def identity(self, role):
        resolved = self.resolve(role)
        return {key: value for key, value in resolved.items() if key != 'keys'}

    async def close(self):
        if self.owns_client:
            await self.client.aclose()

    async def _ledger(self, record):
        async with self.ledger_lock:
            with self.ledger_path.open('a') as stream:
                stream.write(json.dumps(self.sanitize(record), ensure_ascii=False) + '\n')
                stream.flush()

    async def complete(self, context, role, messages, purpose):
        context.check()
        resolved = self.resolve(role)
        route = digest([resolved['base_url'], resolved['model']])
        if self.cooldown.get(route, 0) > time.monotonic():
            raise RateLimited('model_route_is_cooling_down')
        remaining = self.config['execution']['max_generated_tokens_per_question'] - context.generated_upper_bound
        maximum = min(remaining, self.config['execution']['max_completion_tokens'])
        payload = {'model': resolved['model'], 'messages': messages, 'temperature': resolved['temperature'], 'max_tokens': maximum, 'stream': False}
        if resolved['request_options'].keys() & payload.keys():
            raise InfrastructureFailure('request_options_cannot_override_core_request_fields')
        payload.update(resolved['request_options'])
        if resolved['seed_supported']:
            payload['seed'] = context.seed
        slot = int(digest(context.job_id)[:12], 16) % len(resolved['keys'])
        key = resolved['keys'][slot]
        headers = {'Authorization': 'Bearer ' + key} if key else {}
        async with self.semaphore:
            context.check()
            if self.cooldown.get(route, 0) > time.monotonic():
                raise RateLimited('model_route_is_cooling_down')
            async with self.ledger_lock:
                if self.request_count >= self.config['execution']['max_api_requests_per_run']:
                    raise RunBudgetExceeded('run_API_request_limit')
                self.request_count += 1
                request_id = str(self.request_count)
                with self.ledger_path.open('a') as stream:
                    stream.write(json.dumps({'kind': 'request_started', 'request_id': request_id, 'job_id': context.job_id, 'role': role, 'purpose': purpose, 'timestamp': now(), 'credential_slot': slot + 1}) + '\n')
                    stream.flush()
            context.turns += 1
            context.usage[purpose + '_calls'] += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
            started = time.monotonic()
            context.event('model_request', request_id=request_id, role=role, purpose=purpose, messages=messages, max_output_tokens=maximum)
            try:
                timeout = max(0.1, min(self.config['execution']['request_timeout_seconds'], context.deadline - time.monotonic()))
                response = await self.client.post(resolved['base_url'] + '/chat/completions', headers=headers, json=payload, timeout=timeout)
                if response.status_code == 429:
                    try:
                        delay = max(60, float(response.headers.get('retry-after', '60')))
                    except ValueError:
                        delay = 60
                    self.cooldown[route] = time.monotonic() + delay
                    raise RateLimited('HTTP_429_no_key_rotation_or_automatic_retry')
                if response.status_code != 200:
                    raise InfrastructureFailure('HTTP_' + str(response.status_code))
                result = response.json()
                content = result['choices'][0]['message']['content']
                if not isinstance(content, str):
                    raise InfrastructureFailure('completion_has_no_text')
                usage = result.get('usage') or {}
                measured = isinstance(usage.get('completion_tokens'), int) and isinstance(usage.get('prompt_tokens'), int)
                if measured:
                    context.generated_upper_bound += usage['completion_tokens']
                    context.usage[purpose + '_input_tokens'] += usage['prompt_tokens']
                    context.usage[purpose + '_output_tokens'] += usage['completion_tokens']
                else:
                    context.generated_upper_bound += maximum
                    context.missing_usage_calls += 1
                metadata = {'request_id': request_id, 'role': role, 'purpose': purpose, 'requested_model': resolved['model'], 'returned_model': result.get('model'), 'revision': resolved['revision'], 'system_fingerprint': result.get('system_fingerprint'), 'seed_requested': context.seed, 'seed_applied': resolved['seed_supported'], 'finish_reason': result['choices'][0].get('finish_reason'), 'usage': usage if measured else None, 'latency_seconds': round(time.monotonic() - started, 3)}
                context.event('model_response', **metadata, content=self.sanitize(content))
                await self._ledger({'kind': 'request_completed', 'job_id': context.job_id, **metadata})
                return content
            except (httpx.RequestError, ValueError, KeyError, IndexError, TypeError) as error:
                context.generated_upper_bound += maximum
                context.missing_usage_calls += 1
                await self._ledger({'kind': 'request_failed', 'request_id': request_id, 'job_id': context.job_id, 'failure': type(error).__name__, 'usage_unknown': True})
                raise InfrastructureFailure(type(error).__name__) from None
            except InfrastructureFailure as error:
                context.generated_upper_bound += maximum
                context.missing_usage_calls += 1
                await self._ledger({'kind': 'request_failed', 'request_id': request_id, 'job_id': context.job_id, 'failure': str(error), 'usage_unknown': True})
                raise
            except asyncio.CancelledError:
                context.generated_upper_bound += maximum
                context.missing_usage_calls += 1
                await self._ledger({'kind': 'request_failed', 'request_id': request_id, 'job_id': context.job_id, 'failure': 'cancelled_or_question_timeout', 'usage_unknown': True})
                raise
            finally:
                self.active -= 1
