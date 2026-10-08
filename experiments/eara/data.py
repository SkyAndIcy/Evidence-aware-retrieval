import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import sys

import httpx

from .common import digest, file_digest, now, path, read_jsonl, write_json, write_jsonl


DATASET_NAMES = {'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle', 'morehopqa'}


def download(url, destination, max_gib, expected_sha256=None):
    if not url.startswith('https://'):
        raise ValueError('Only HTTPS public data downloads are permitted')
    destination = Path(destination)
    if destination.exists():
        if expected_sha256 and file_digest(destination) == expected_sha256:
            return
        raise FileExistsError('Destination already exists; verify it or choose a new path')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.part')
    limit = int(max_gib * 1024 ** 3)
    if limit <= 0 or shutil.disk_usage(destination.parent).free < limit:
        raise ValueError('Download limit must be positive and fit available disk space')
    with httpx.stream('GET', url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        advertised = int(response.headers.get('content-length', '0'))
        if advertised > limit:
            raise ValueError('Download exceeds the explicit size limit')
        received = 0
        checksum = hashlib.sha256()
        with temporary.open('wb') as stream:
            for block in response.iter_bytes(1024 * 1024):
                received += len(block)
                if received > limit:
                    raise ValueError('Download exceeds the explicit size limit')
                checksum.update(block)
                stream.write(block)
    actual = checksum.hexdigest()
    if expected_sha256 and actual != expected_sha256:
        raise ValueError('Downloaded checksum does not match the expected checksum')
    os.replace(temporary, destination)
    write_json(str(destination) + '.download.json', {'url': url, 'sha256': actual, 'bytes': received, 'downloaded_at': now(), 'upstream_checksum_verified': expected_sha256 is not None})


def corpus_import(source, destination, identity, max_rows=None):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError('Corpus already exists')
    if identity == 'dpr_december_2018_wikipedia_psgs_w100' and max_rows is not None:
        raise ValueError('A limited corpus must use a diagnostic identity, not the full Wikipedia identity')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.building.sqlite')
    if temporary.exists():
        raise FileExistsError('An incomplete corpus build exists; inspect it before restarting')
    connection = sqlite3.connect(temporary)
    connection.execute('CREATE TABLE passages (position INTEGER PRIMARY KEY, id TEXT UNIQUE NOT NULL, title TEXT NOT NULL, text TEXT NOT NULL)')
    csv.field_size_limit(sys.maxsize)
    opener = gzip.open if source.suffix == '.gz' else open
    checksum = hashlib.sha256()
    count = 0
    try:
        with opener(source, 'rt', encoding='utf-8', newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            if not {'id', 'text', 'title'}.issubset(reader.fieldnames or []):
                raise ValueError('Expected DPR TSV columns: id, text, title')
            for record in reader:
                if max_rows is not None and count >= max_rows:
                    break
                if not record['id'] or not record['text']:
                    raise ValueError('Empty passage ID or text')
                item = [str(record['id']), record['title'], record['text']]
                checksum.update((json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
                connection.execute('INSERT INTO passages VALUES (?,?,?,?)', [count, *item])
                count += 1
                if count % 10000 == 0:
                    connection.commit()
        if not count:
            raise ValueError('Corpus is empty')
        metadata = {'identity': identity, 'passages': count, 'source_sha256': file_digest(source), 'corpus_sha256': checksum.hexdigest(), 'created_at': now(), 'source_file': source.name, 'is_partial': max_rows is not None}
        connection.execute('CREATE TABLE metadata (value TEXT NOT NULL)')
        connection.execute('INSERT INTO metadata VALUES (?)', [json.dumps(metadata)])
        connection.commit()
    finally:
        connection.close()
    os.replace(temporary, destination)
    write_json(str(destination) + '.meta.json', metadata)
    return metadata


def corpus_metadata(filename):
    with sqlite3.connect(f'file:{Path(filename).resolve()}?mode=ro', uri=True) as connection:
        return json.loads(connection.execute('SELECT value FROM metadata').fetchone()[0])


def aliases_from(record):
    answer = record.get('answer', record.get('Answer', record.get('answers')))
    aliases = record.get('answer_aliases', [])
    if isinstance(answer, dict):
        aliases = answer.get('aliases', [])
        answer = answer.get('text', answer.get('value'))
    values = answer if isinstance(answer, list) else [answer]
    if isinstance(aliases, str):
        aliases = [aliases]
    values = list(dict.fromkeys(str(value).strip() for value in [*values, *aliases] if value is not None and str(value).strip()))
    if not values:
        raise ValueError('Missing reference answers; blind test files cannot be scored locally')
    return values


def normalize_dataset(name, source, destination, source_url, parent_mapping=None):
    if name not in DATASET_NAMES:
        raise ValueError('Unsupported benchmark name')
    source = Path(source)
    if Path(destination).exists():
        raise FileExistsError('Normalized dataset already exists')
    if source.suffix == '.jsonl':
        records = read_jsonl(source)
    elif source.suffix in {'.tsv', '.csv'}:
        with source.open(newline='') as stream:
            records = list(csv.DictReader(stream, delimiter='\t' if source.suffix == '.tsv' else ','))
    else:
        records = json.loads(source.read_text())
        if isinstance(records, dict):
            records = records.get('data', records.get('examples', records.get('questions')))
        if not isinstance(records, list):
            raise ValueError('Expected a record list, JSONL, or CSV/TSV; convert unusual source schemas explicitly')
    mapping = json.loads(Path(parent_mapping).read_text()) if parent_mapping else {}
    normalized, seen = [], set()
    for index, record in enumerate(records):
        if name == 'musique' and record.get('answerable') is False:
            continue
        question = record.get('question', record.get('Question'))
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f'Missing question at source row {index}')
        source_id = str(record.get('_id', record.get('id', record.get('question_id', digest(question)[:24]))))
        question_id = name + ':' + source_id
        if question_id in seen:
            raise ValueError('Duplicate question ID: ' + question_id)
        seen.add(question_id)
        parents = mapping.get(source_id, record.get('parent_ids', []))
        if isinstance(parents, str):
            parents = [parents]
        if not isinstance(parents, list) or any(not isinstance(parent, str) or ':' not in parent for parent in parents):
            raise ValueError('Parent identifiers must be dataset-qualified strings')
        normalized.append({'id': question_id, 'source_id': source_id, 'dataset': name, 'language': 'en', 'question': question.strip(), 'answers': aliases_from(record), 'parent_ids': parents, 'parent_mapping_reviewed': source_id in mapping, 'source_row': index})
    if not normalized:
        raise ValueError('Dataset is empty')
    write_jsonl(destination, normalized)
    write_json(str(destination) + '.meta.json', {'dataset': name, 'source_url': source_url, 'source_sha256': file_digest(source), 'normalized_sha256': file_digest(destination), 'count': len(normalized), 'created_at': now(), 'gold_contexts_imported': False, 'parent_mapping_sha256': file_digest(parent_mapping) if parent_mapping else None})
    return len(normalized)


def freeze_splits(config):
    destination = path(config['splits']['directory'])
    if (destination / 'manifest.json').exists():
        raise FileExistsError('Splits are already frozen; choose another directory rather than silently resampling')
    pools, provenance = {}, {}
    for name, spec in config['datasets'].items():
        filename = path(spec['input'])
        pool = list(read_jsonl(filename))
        if any(record['dataset'] != name for record in pool):
            raise ValueError('Dataset label mismatch')
        if spec.get('require_parent_mapping') and any(not record.get('parent_mapping_reviewed') for record in pool):
            raise ValueError('MoreHopQA needs an explicitly reviewed parent-ID mapping, including empty lists where no parent applies')
        pools[name] = pool
        provenance[name] = file_digest(filename)
    all_ids = {record['id'] for pool in pools.values() for record in pool}
    seen_questions, excluded, splits = set(), [], {'dev': [], 'test': []}
    for name, pool in pools.items():
        usable = []
        for record in sorted(pool, key=lambda item: item['id']):
            normalized_question = ' '.join(record['question'].casefold().split())
            if normalized_question in seen_questions or any(parent in all_ids for parent in record.get('parent_ids', [])):
                excluded.append({'id': record['id'], 'reason': 'duplicate_question_or_parent_in_primary_pool'})
                continue
            seen_questions.add(normalized_question)
            usable.append(record)
        random.Random(str(config['study']['selection_seed']) + ':' + name).shuffle(usable)
        spec = config['datasets'][name]
        required = spec['dev'] + spec['test']
        if len(usable) < required:
            raise ValueError(f'{name}: requested {required}, only {len(usable)} after de-duplication; amend targets explicitly instead of resampling silently')
        for split, selected in [('dev', usable[:spec['dev']]), ('test', usable[spec['dev']:required])]:
            for record in selected:
                splits[split].append({**record, 'split': split})
    dev_ids = {record['id'] for record in splits['dev']}
    test_ids = {record['id'] for record in splits['test']}
    if dev_ids & test_ids:
        raise ValueError('Development/test overlap')
    for split, records in splits.items():
        write_jsonl(destination / (split + '.jsonl'), records)
    manifest = {'created_at': now(), 'selection_seed': config['study']['selection_seed'], 'sources': provenance, 'excluded': excluded, 'counts': {split: {name: sum(record['dataset'] == name for record in records) for name in pools} for split, records in splits.items()}, 'files': {split: file_digest(destination / (split + '.jsonl')) for split in splits}, 'policy': config['splits']}
    write_json(destination / 'manifest.json', manifest)
    return manifest
