import importlib.metadata
import json
from pathlib import Path
import shutil
import sqlite3
import threading

import bm25s
import numpy as np
import Stemmer

from .common import digest, now, write_json
from .data import corpus_metadata


def tokenizer(settings):
    stemmer = Stemmer.Stemmer(settings['stemmer']) if settings.get('stemmer') else None
    return bm25s.tokenization.Tokenizer(stopwords=settings['stopwords'], stemmer=stemmer)


def build_bm25(corpus, destination, settings, allow_large=False):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Index already exists; use a new directory')
    metadata = corpus_metadata(corpus)
    if metadata['passages'] > 1_000_000 and not allow_large:
        raise ValueError('Large BM25 construction retains the tokenized corpus in RAM; inspect resources and pass --allow-large-index explicitly')
    tokens = tokenizer(settings)
    token_ids = []
    with sqlite3.connect(f'file:{Path(corpus).resolve()}?mode=ro', uri=True) as connection:
        cursor = connection.execute('SELECT title,text FROM passages ORDER BY position')
        while batch := cursor.fetchmany(10000):
            token_ids.extend(tokens.tokenize([title + '\n' + text for title, text in batch], update_vocab=True, show_progress=False))
    if len(token_ids) != metadata['passages']:
        raise ValueError('Corpus row count changed during indexing')
    retriever = bm25s.BM25(k1=settings['k1'], b=settings['b'], method=settings['method'])
    retriever.index((token_ids, tokens.get_vocab_dict()), show_progress=False)
    temporary = destination.with_name(destination.name + '.building')
    if temporary.exists():
        raise FileExistsError('Incomplete BM25 build exists')
    retriever.save(temporary, show_progress=False)
    tokens.save_vocab(temporary)
    index_metadata = {'kind': 'bm25s', 'version': importlib.metadata.version('bm25s'), 'pystemmer_version': importlib.metadata.version('PyStemmer'), 'corpus': metadata, 'settings': settings, 'created_at': now()}
    write_json(temporary / 'metadata.json', index_metadata)
    temporary.rename(destination)
    return index_metadata


def build_dense(corpus, destination, settings, allow_model_download=False, device='cpu'):
    from sentence_transformers import SentenceTransformer

    if not settings.get('revision'):
        raise ValueError('Freeze a dense encoder revision before creating an index')
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Dense index already exists')
    metadata = corpus_metadata(corpus)
    encoder = SentenceTransformer(settings['model'], revision=settings['revision'], device=device, trust_remote_code=False, local_files_only=not allow_model_download)
    dimensions = encoder.get_sentence_embedding_dimension()
    destination.parent.mkdir(parents=True, exist_ok=True)
    estimated = metadata['passages'] * dimensions * 4
    if shutil.disk_usage(destination.parent).free < estimated * 1.1:
        raise ValueError(f'Dense float32 matrix alone requires {estimated} bytes, exceeding free disk space with safety margin')
    temporary = destination.with_name(destination.name + '.building')
    temporary.mkdir()
    vectors = np.lib.format.open_memmap(temporary / 'vectors.npy', mode='w+', dtype='float32', shape=(metadata['passages'], dimensions))
    with sqlite3.connect(f'file:{Path(corpus).resolve()}?mode=ro', uri=True) as connection:
        cursor = connection.execute('SELECT position,title,text FROM passages ORDER BY position')
        while batch := cursor.fetchmany(settings['batch_size']):
            encoded = encoder.encode([title + '\n' + text for _, title, text in batch], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
            vectors[[position for position, _, _ in batch]] = encoded
    vectors.flush()
    index_metadata = {'kind': 'dense_exact_memmap', 'model': settings['model'], 'revision': settings['revision'], 'sentence_transformers_version': importlib.metadata.version('sentence-transformers'), 'dimensions': dimensions, 'corpus': metadata, 'normalized': True, 'created_at': now()}
    write_json(temporary / 'metadata.json', index_metadata)
    temporary.rename(destination)
    return index_metadata


def fuse(lexical, dense, lexical_weight, dense_weight, constant=60):
    if lexical_weight < 0 or dense_weight < 0 or lexical_weight + dense_weight <= 0:
        raise ValueError('Fusion weights must be nonnegative and not both zero')
    totals = {}
    for weight, ranking in [(lexical_weight, lexical), (dense_weight, dense)]:
        if weight == 0:
            continue
        for rank, (position, _) in enumerate(ranking, 1):
            totals[position] = totals.get(position, 0.0) + weight / (lexical_weight + dense_weight) / (constant + rank)
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))


class Retriever:
    def __init__(self, corpus, bm25_index, dense_index=None, backend='bm25', lexical_weight=10, dense_weight=0, candidate_depth=100, top_k=5, rrf_constant=60, scan_batch_size=16384, device='cpu'):
        self.corpus = Path(corpus).resolve()
        self.metadata = corpus_metadata(self.corpus)
        self.backend = backend
        self.lexical_weight = lexical_weight
        self.dense_weight = dense_weight
        self.candidate_depth = candidate_depth
        self.top_k = top_k
        self.rrf_constant = rrf_constant
        self.scan_batch_size = scan_batch_size
        self.lock = threading.Lock()
        self.index_metadata = {}
        if backend not in {'bm25', 'dense', 'hybrid'}:
            raise ValueError('Unknown retrieval backend')
        if backend == 'hybrid' and (lexical_weight < 0 or dense_weight < 0 or lexical_weight + dense_weight == 0):
            raise ValueError('Invalid retrieval weights')
        self.use_bm25 = backend == 'bm25' or (backend == 'hybrid' and lexical_weight > 0)
        self.use_dense = backend == 'dense' or (backend == 'hybrid' and dense_weight > 0)
        if self.use_bm25:
            location = Path(bm25_index)
            meta = json.loads((location / 'metadata.json').read_text())
            self._check_corpus(meta)
            if meta['version'] != importlib.metadata.version('bm25s'):
                raise ValueError('BM25s runtime version differs from index version')
            self.index_metadata['bm25'] = meta
            self.bm25 = bm25s.BM25.load(location, mmap=True, load_corpus=False, show_progress=False)
            self.tokenizer = tokenizer(meta['settings'])
            self.tokenizer.load_vocab(location)
        if self.use_dense:
            from sentence_transformers import SentenceTransformer

            location = Path(dense_index)
            meta = json.loads((location / 'metadata.json').read_text())
            self._check_corpus(meta)
            self.index_metadata['dense'] = meta
            self.vectors = np.load(location / 'vectors.npy', mmap_mode='r')
            if self.vectors.shape != (self.metadata['passages'], meta['dimensions']):
                raise ValueError('Dense matrix has an invalid shape')
            self.encoder = SentenceTransformer(meta['model'], revision=meta['revision'], device=device, trust_remote_code=False, local_files_only=True)
        self.revision = digest({'indexes': self.index_metadata, 'backend': backend, 'weights': [lexical_weight, dense_weight], 'candidate_depth': candidate_depth, 'top_k': top_k, 'rrf_constant': rrf_constant})

    def _check_corpus(self, index):
        if index['corpus']['corpus_sha256'] != self.metadata['corpus_sha256']:
            raise ValueError('Retriever index and corpus checksum differ')

    def _lexical(self, query):
        tokens = self.tokenizer.tokenize([query], update_vocab=False, show_progress=False)
        if not tokens[0]:
            return []
        positions, scores = self.bm25.retrieve(tokens, k=min(self.candidate_depth, self.metadata['passages']), show_progress=False, n_threads=1)
        return sorted([(int(position), float(score)) for position, score in zip(positions[0], scores[0]) if score > 0], key=lambda item: (-item[1], item[0]))

    def _dense(self, query):
        encoded = self.encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        best = []
        for start in range(0, len(self.vectors), self.scan_batch_size):
            scores = np.asarray(self.vectors[start:start + self.scan_batch_size]) @ encoded
            order = np.lexsort((np.arange(len(scores)), -scores))[:self.candidate_depth]
            best.extend((start + int(index), float(scores[index])) for index in order)
            best = sorted(best, key=lambda item: (-item[1], item[0]))[:self.candidate_depth]
        return best

    def search(self, query):
        with self.lock:
            lexical = self._lexical(query) if self.use_bm25 else []
            dense = self._dense(query) if self.use_dense else []
        if self.use_bm25 and self.use_dense:
            ranking = fuse(lexical, dense, self.lexical_weight, self.dense_weight, self.rrf_constant)
        else:
            ranking = lexical if self.use_bm25 else dense
        passages = []
        with sqlite3.connect(f'file:{self.corpus}?mode=ro', uri=True) as connection:
            for rank, (position, score) in enumerate(ranking[:self.top_k], 1):
                row = connection.execute('SELECT id,title,text FROM passages WHERE position=?', [position]).fetchone()
                if row is None:
                    raise ValueError('Index references a missing passage')
                passages.append({'id': row[0], 'title': row[1], 'text': row[2], 'score': score, 'rank': rank})
        return {'passages': passages, 'physical_calls': int(self.use_bm25) + int(self.use_dense), 'candidate_counts': {'bm25': len(lexical), 'dense': len(dense)}}
