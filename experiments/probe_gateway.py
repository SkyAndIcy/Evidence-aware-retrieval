import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx
import yaml


def load_jobs(config_path, aliases, first_key_only=False):
    config = yaml.safe_load(Path(config_path).read_text())
    jobs = []
    credentials = []
    for model in config['models'].values():
        if isinstance(model, dict):
            keys = model.get('api_key', [])
            keys = keys if isinstance(keys, list) else [keys]
            credentials.extend(key for key in keys if isinstance(key, str) and key)
    credentials = list(dict.fromkeys(credentials))
    for alias in aliases:
        model = config['models'].get(alias)
        if not isinstance(model, dict):
            raise ValueError('Requested model alias is not configured')
        base_url = model['base_url'].rstrip('/')
        parsed = urlsplit(base_url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.query:
            raise ValueError('An HTTPS endpoint without URL credentials is required')
        keys = model['api_key']
        keys = keys if isinstance(keys, list) else [keys]
        if first_key_only:
            keys = keys[:1]
        for key in keys:
            if not isinstance(key, str) or not key.strip():
                raise ValueError('An empty credential was configured')
            jobs.append({
                'alias': alias,
                'model': model['model_name'],
                'url': base_url + '/chat/completions',
                'credential': key,
                'credential_slot': credentials.index(key) + 1,
            })
    return jobs, credentials


def redact(value, credentials):
    encoded = json.dumps(value, ensure_ascii=False)
    for credential in sorted(credentials, key=len, reverse=True):
        encoded = encoded.replace(credential, '[REDACTED]')
    return json.loads(encoded)


async def run_probes(jobs, credentials, concurrency, timeout_seconds):
    semaphore = asyncio.Semaphore(concurrency)
    active = 0
    peak = 0
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    timeout = httpx.Timeout(timeout_seconds, connect=min(15, timeout_seconds))
    async with httpx.AsyncClient(limits=limits, timeout=timeout, follow_redirects=False) as client:
        async def probe(job):
            nonlocal active, peak
            async with semaphore:
                active += 1
                peak = max(peak, active)
                started = time.monotonic()
                record = {
                    'alias': job['alias'],
                    'requested_model': job['model'],
                    'credential_slot': job['credential_slot'],
                    'started_at': datetime.now(timezone.utc).isoformat(),
                    'status': 'failed',
                    'benchmark_result': False,
                }
                try:
                    response = await client.post(
                        job['url'],
                        headers={'Authorization': 'Bearer ' + job['credential']},
                        json={
                            'model': job['model'],
                            'messages': [{'role': 'user', 'content': 'Reply with exactly OK.'}],
                            'max_tokens': 64,
                            'stream': False,
                        },
                    )
                    record['http_status'] = response.status_code
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = None
                    if response.status_code != 200:
                        record['failure_type'] = 'http_error'
                        record['error_excerpt'] = redact(response.text, credentials)[:800]
                        record['retry_after'] = response.headers.get('retry-after')
                    elif not isinstance(payload, dict) or not payload.get('choices'):
                        record['failure_type'] = 'invalid_completion_response'
                    else:
                        choice = payload['choices'][0]
                        content = choice.get('message', {}).get('content')
                        record['returned_model'] = payload.get('model')
                        record['system_fingerprint'] = payload.get('system_fingerprint')
                        record['finish_reason'] = choice.get('finish_reason')
                        record['usage'] = payload.get('usage')
                        record['content_excerpt'] = redact(content, credentials)[:120] if isinstance(content, str) else None
                        record['exact_ok'] = isinstance(content, str) and content.strip() == 'OK'
                        if isinstance(content, str) and content.strip():
                            record['status'] = 'success'
                        else:
                            record['failure_type'] = 'empty_content'
                except httpx.RequestError as error:
                    record['failure_type'] = type(error).__name__
                    record['error_excerpt'] = redact(str(error), credentials)[:800]
                except (KeyError, IndexError, TypeError, AttributeError):
                    record['failure_type'] = 'invalid_completion_response'
                finally:
                    record['elapsed_seconds'] = round(time.monotonic() - started, 3)
                    active -= 1
                return redact(record, credentials)

        records = await asyncio.gather(*(probe(job) for job in jobs))
    return {
        'kind': 'connectivity_probe_not_paper_experiment',
        'requested_concurrency': concurrency,
        'peak_in_flight_attempts': peak,
        'request_count': len(records),
        'success_count': sum(record['status'] == 'success' for record in records),
        'failure_counts': dict(Counter(record.get('failure_type') for record in records if record['status'] != 'success')),
        'automatic_retries': 0,
        'records': records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--first-key-only', action='store_true')
    parser.add_argument('--concurrency', type=int, default=20)
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 20 or args.timeout <= 0:
        parser.error('Concurrency must be 1..20 and timeout must be positive')
    jobs, credentials = load_jobs(args.config, args.models, args.first_key_only)
    if len(jobs) > 100:
        parser.error('Probe mode permits at most 100 short requests')
    report = asyncio.run(run_probes(jobs, credentials, args.concurrency, args.timeout))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    encoded = json.dumps(redact(report, credentials), ensure_ascii=False, indent=2) + '\n'
    assert not any(credential in encoded for credential in credentials)
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(encoded)
    print(json.dumps({key: value for key, value in report.items() if key != 'records'}, ensure_ascii=False))
    return 0 if report['success_count'] == report['request_count'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
