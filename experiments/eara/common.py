from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[2]


def path(value):
    item = Path(value).expanduser()
    return item if item.is_absolute() else ROOT / item


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_digest(filename):
    checksum = hashlib.sha256()
    with Path(filename).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(filename, value):
    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_jsonl(filename):
    with Path(filename).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write_jsonl(filename, records):
    destination = Path(filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def config_load(filename, seen=None):
    filename = Path(filename).resolve()
    seen = set() if seen is None else seen
    if filename in seen:
        raise ValueError('Configuration inheritance cycle')
    seen.add(filename)
    config = yaml.safe_load(filename.read_text())
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a mapping')
    if 'extends' in config:
        base = config_load(filename.parent / config.pop('extends'), seen)
        def merge(original, override):
            combined = dict(original)
            for key, value in override.items():
                combined[key] = merge(combined[key], value) if key != 'datasets' and isinstance(value, dict) and isinstance(combined.get(key), dict) else value
            return combined
        config = merge(base, config)
    if not isinstance(config, dict) or config.get('schema_version') != 1:
        raise ValueError('Unsupported configuration schema')
    if not 1 <= config['execution']['api_concurrency'] <= 20:
        raise ValueError('API concurrency must be between 1 and 20')
    if config['retrieval']['top_k'] != 5 or config['retrieval']['logical_budget'] != 10:
        raise ValueError('This protocol requires top-5 and ten logical retrieval calls')
    if config['execution']['format_retries'] != 1:
        raise ValueError('This protocol permits exactly one schema-only repair')
    if config['audit']['model_labels_are_human']:
        raise ValueError('Model labels cannot be marked as human labels')
    return config


def parse_json(text):
    stripped = text.strip()
    if stripped.startswith('```') and stripped.endswith('```'):
        stripped = stripped.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError('A JSON object is required')
    return value


def public_question(record):
    return {key: record[key] for key in ['id', 'dataset', 'question', 'language'] if key in record}
