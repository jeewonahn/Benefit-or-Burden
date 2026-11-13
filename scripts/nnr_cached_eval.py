"""Cached evaluation matching the explicitly selected NNR implementation.

NRMS/LSTUR/NAML cache news and user vectors. CNE-SUE caches news and history
GCN features while retaining candidate attention. NPA caches CNN token features
and personalized user vectors while retaining personalized news attention.
NPA and CNE-SUE use the recorded query and cross-view batch alignment in both
training and evaluation.
"""
import hashlib
import json
from pathlib import Path
import scheduled_cooling as cooling
import time
from contextlib import contextmanager
from functools import wraps

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from naml_cached_eval import (
    NamlCachedEvaluator, ROOT, corpus_fingerprint, model_fingerprint,
    _json_bytes, _file_hash, _save_npy, _save_json, METRIC_NAMES, metrics_from_ranks,
)
from datetime import datetime, timezone
from torch_scatter import scatter_softmax, scatter_sum
import cne_user_cache


@contextmanager
def strict_fp32_evaluation(enabled=True):
    """Temporarily disable TF32; always restore the caller's training policy."""
    if not enabled:
        yield
        return
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def _lstur_precision_entry(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        model = (args[0] if args else kwargs.get('model')) if method.__name__ == '__init__' else self.model
        targeted = (type(model.news_encoder).__name__ == 'CNN' and
                    type(model.user_encoder).__name__ == 'LSTUR')
        with strict_fp32_evaluation(targeted):
            return method(self, *args, **kwargs)
    return wrapped


class NNRCachedEvaluator(NamlCachedEvaluator):
    """Same evaluation/artifact API as NamlCachedEvaluator, with config input."""

    @_lstur_precision_entry
    def __init__(self, model, corpus, config, cache_dir, news_batch_size=256,
                 user_batch_size=256, score_batch_size=32768,
                 personalized_batch_size=512, npa_features_on_gpu=False,
                 gpu_memory_budget_gib=7.0):
        pair = (type(model.news_encoder).__name__, type(model.user_encoder).__name__)
        supported = {('NAML', 'ATT'): 'NAML', ('MHSA', 'MHSA'): 'NRMS', ('CNN', 'LSTUR'): 'LSTUR',
                     ('CNE', 'SUE'): 'CNE-SUE', ('PNE', 'PUE'): 'NPA'}
        if pair not in supported or model.click_predictor != 'dot_product':
            raise ValueError('Unsupported independent encoder pair: ' + repr(pair))
        if any(int(x) <= 0 for x in (news_batch_size, user_batch_size, score_batch_size)):
            raise ValueError('Batch sizes must be positive')
        if not 1 <= int(personalized_batch_size) <= 8192:
            raise ValueError('personalized_batch_size must be in [1, 8192]')
        if not 0 < float(gpu_memory_budget_gib) <= 64:
            raise ValueError('gpu_memory_budget_gib must be positive and at most 64')
        if next(model.parameters()).dtype != torch.float32:
            raise ValueError('This validated cache path requires a float32 model')
        self.kind = supported[pair]
        if self.kind in ('CNE-SUE', 'NPA') and not getattr(model, '_tors_batch_alignment_fixes', None):
            raise ValueError('Apply explicit batch alignment fixes before training/cache construction')
        self.model, self.corpus, self.config = model, corpus, config
        self.device = next(model.parameters()).device
        self.news_batch_size, self.user_batch_size = int(news_batch_size), int(user_batch_size)
        self.score_batch_size = int(score_batch_size)
        self.personalized_batch_size = int(personalized_batch_size)
        self.gpu_memory_budget_gib = float(gpu_memory_budget_gib)
        self.npa_features_device = None
        self.feature_residency = {'requested_on_gpu': bool(npa_features_on_gpu),
                                 'actual_on_gpu': False, 'reason': 'not_requested',
                                 'gpu_memory_budget_gib': self.gpu_memory_budget_gib}
        self.history_length = int(corpus.max_history_num)
        self.dimension = int(model.news_embedding_dim)
        self.model_hash = model_fingerprint(model)
        self.user_mapping_hash = hashlib.sha256(_json_bytes(corpus.user_ID_dict)).hexdigest()
        self.provenance = {
            'schema': 1, 'model': '-'.join(pair), 'click_predictor': 'dot_product',
            'model_state_sha256': self.model_hash,
            'corpus_sha256': corpus_fingerprint(corpus),
            'user_mapping_sha256': self.user_mapping_hash,
            'history_length': self.history_length,
            'padding_policy': 'historical_' + pair[1], 'dtype': 'float32',
            'evaluation_precision': {
                'policy': 'strict_fp32_eval_v1' if self.kind == 'LSTUR' else 'caller_policy_v1',
                'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
                'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
                'device_type': self.device.type, 'torch_version': str(torch.__version__),
            },
            'source_sha256': {name: _file_hash(ROOT / 'NNR-main' / name)
                              for name in ('model.py', 'newsEncoders.py', 'userEncoders.py', 'layers.py')},
            'cache_implementation_sha256': _file_hash(__file__),
            'base_cache_implementation_sha256': _file_hash(ROOT / 'scripts' / 'naml_cached_eval.py'),
            'alignment_implementation_sha256': _file_hash(ROOT / 'scripts' / 'nnr_batch_alignment.py'),
            'implementation_adjustments': getattr(model, '_tors_batch_alignment_fixes', []),
            'graph_config': {key: getattr(config, key) for key in
                             ('no_self_connection', 'no_adjacent_normalization', 'gcn_normalization_type')},
        }
        if self.kind == 'CNE-SUE':
            self.provenance['user_cache_storage'] = cne_user_cache.STORAGE_VERSION
            self.provenance['user_cache_implementation_sha256'] = _file_hash(cne_user_cache.__file__)
        self.identity = hashlib.sha256(_json_bytes(self.provenance)).hexdigest()
        self.cache_dir = Path(cache_dir) / self.identity
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_manifest = self.cache_dir / 'manifest.json'
        self.user_keys, self.user_lookup = [], {}
        self.user_shape = ((self.history_length, self.dimension) if self.kind == 'CNE-SUE' else (self.dimension,))
        self.users = np.zeros((0, *self.user_shape), dtype=np.float32)
        self.model.eval()
        self.timings = {'news_encoding_seconds': 0.0, 'user_encoding_seconds': 0.0}
        self.cache_loaded = False
        if self.cache_manifest.exists():
            self._load_cache()
            self.cache_loaded = True
        else:
            if any(self.cache_dir.iterdir()):
                raise ValueError('Incomplete cache directory; use a fresh cache root')
            self.news = self._encode_news()
            self._save_cache()
        self.news_device = None if self.kind == 'NPA' else torch.as_tensor(self.news, device=self.device)
        if npa_features_on_gpu:
            self._place_npa_features_on_gpu()

    def _place_npa_features_on_gpu(self):
        """Use spare device memory without changing the personalized encoder.

        Reserve workspace for attention and include driver-level usage in the
        7 GiB default budget. Fall back to the identical CPU-backed cache when
        the device cannot fit both stored features and the estimated workspace.
        """
        if self.kind != 'NPA' or self.device.type != 'cuda':
            self.feature_residency['reason'] = 'requires_NPA_on_CUDA'
            return
        torch.cuda.empty_cache()
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        feature_bytes = int(self.news.nbytes)
        length = int(self.model.news_encoder.max_sentence_length)
        width = int(self.model.news_encoder.cnn_kernel_num)
        attention_width = int(self.model.news_encoder.personalizedAttention.feature_affine.out_features)
        work_bytes = (512 * 1024 ** 2 + self.personalized_batch_size *
                      (2 * length * width + 3 * length * attention_width + 4 * self.dimension) * 4)
        used_bytes = total_bytes - free_bytes
        budget = min(int(self.gpu_memory_budget_gib * 1024 ** 3), int(total_bytes))
        self.feature_residency.update(feature_bytes=feature_bytes, estimated_workspace_bytes=int(work_bytes),
            driver_used_before_bytes=int(used_bytes), driver_free_before_bytes=int(free_bytes),
            process_allocated_before_bytes=int(torch.cuda.memory_allocated(self.device)))
        if feature_bytes + work_bytes > free_bytes or used_bytes + feature_bytes + work_bytes > budget:
            self.feature_residency['reason'] = 'insufficient_budget_with_attention_workspace'
            return
        try:
            self.npa_features_device = torch.tensor(np.asarray(self.news), device=self.device)
        except RuntimeError as exc:
            if 'out of memory' not in str(exc).lower():
                raise
            self.npa_features_device = None
            torch.cuda.empty_cache()
            self.feature_residency['reason'] = 'allocation_OOM_CPU_fallback'
            return
        self.feature_residency.update(actual_on_gpu=True, reason='within_budget',
            process_allocated_after_bytes=int(torch.cuda.memory_allocated(self.device)))

    def runtime_memory_report(self):
        report = {'personalized_batch_size': self.personalized_batch_size,
                  'npa_feature_residency': dict(self.feature_residency)}
        if self.device.type == 'cuda':
            report.update(process_current_allocated_bytes=int(torch.cuda.memory_allocated(self.device)),
                          process_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(self.device)))
        return report

    def _check_user_mapping(self):
        if hashlib.sha256(_json_bytes(self.corpus.user_ID_dict)).hexdigest() != self.user_mapping_hash:
            raise ValueError('User ID mapping changed after cache construction')
        if any(getattr(self.config, key) != value for key, value in self.provenance['graph_config'].items()):
            raise ValueError('Graph configuration changed after cache construction')
        precision = self.provenance['evaluation_precision']
        if (bool(torch.backends.cuda.matmul.allow_tf32) != precision['cuda_matmul_allow_tf32'] or
                bool(torch.backends.cudnn.allow_tf32) != precision['cudnn_allow_tf32']):
            raise ValueError('Evaluation precision changed from the cache identity')

    @_lstur_precision_entry
    def verify_against_forward(self, behavior_path, max_entries=8, max_candidates=32,
                               atol=1e-4, rtol=1e-5):
        """Compare the actual model.forward and cached path on bounded real rows.

        This evaluates the supplied checkpoint, not a training update. Candidate
        order and padding are unchanged; only the number of inspected rows and
        candidates is bounded. Raises if numerical equivalence fails.
        """
        self._check_user_mapping()
        if model_fingerprint(self.model) != self.model_hash:
            raise ValueError('Model changed after cache construction; create a new evaluator')
        if corpus_fingerprint(self.corpus) != self.provenance['corpus_sha256']:
            raise ValueError('Corpus changed after cache construction; create a new evaluator')
        if max_entries <= 0 or max_candidates <= 0:
            raise ValueError('Verification limits must be positive')
        rows, keys = [], []
        with Path(behavior_path).open('r', encoding='utf-8') as stream:
            for line in stream:
                _, user, _, raw_history, raw_candidates = line.rstrip('\r\n').split('\t')
                history = self._history(raw_history)
                candidates = [self.corpus.news_ID_dict[item.rsplit('-', 1)[0]]
                              for item in raw_candidates.split()[:max_candidates]]
                if not candidates:
                    raise ValueError('No candidates to verify')
                rows.append((user, history, candidates, len(raw_history.split())))
                keys.append(self._user_key(user, history))
                if len(rows) == max_entries:
                    break
        if not rows:
            raise ValueError('No rows to verify')
        user_indices = self._ensure_users(keys)
        feature_order = ('news_category', 'news_subCategory', 'news_title_text', 'news_title_mask',
                         'news_title_entity', 'news_abstract_text', 'news_abstract_mask', 'news_abstract_entity')
        def features(indices):
            return [torch.as_tensor(np.asarray(getattr(self.corpus, name)[indices]),
                                    device=self.device).unsqueeze(0) for name in feature_order]
        largest, compared = 0.0, 0
        self.model.eval()
        with torch.inference_mode():
            for i, (user, history, candidates, original_history_length) in enumerate(rows):
                history_mask = torch.zeros((1, self.history_length), dtype=torch.bool, device=self.device)
                history_mask[:, :min(self.history_length, original_history_length)] = True
                user_id = getattr(self.corpus, 'user_ID_dict', {}).get(user, 0)
                graph, graph_mask, graph_indices = self._history_graph(history)
                ignored_graph = torch.as_tensor(graph, device=self.device).unsqueeze(0)
                ignored_category_mask = torch.as_tensor(graph_mask, device=self.device).unsqueeze(0)
                ignored_category_indices = torch.as_tensor(graph_indices, device=self.device).unsqueeze(0)
                direct = self.model(torch.tensor([user_id], device=self.device), *features(history),
                                    history_mask, ignored_graph, ignored_category_mask,
                                    ignored_category_indices, *features(candidates)).flatten()
                cached_scores, _ = self._score_flat(np.asarray(candidates, dtype=np.int32),
                    np.full(len(candidates), user_indices[i], dtype=np.int32))
                cached = torch.as_tensor(cached_scores, device=self.device)
                difference = float((direct - cached).abs().max().item())
                largest = max(largest, difference)
                compared += len(candidates)
                if not torch.allclose(direct, cached, atol=atol, rtol=rtol):
                    raise ValueError('Cached vs direct forward mismatch: max_abs=' + str(difference))
        return {'status': 'passed', 'rows': len(rows), 'candidate_scores': compared,
                'max_absolute_error': largest, 'atol': atol, 'rtol': rtol,
                'cache_identity': self.identity, 'model_state_sha256': self.model_hash,
                'input_sha256': _file_hash(behavior_path),
                'device': str(self.device), 'evaluation_precision': self.provenance['evaluation_precision'],
                'scope': 'Bounded direct-forward/cache equivalence'}

    def _ensure_users(self, keys):
        self._check_user_mapping()
        pending = list(dict.fromkeys(key for key in keys if key not in self.user_lookup))
        if pending and self.kind == 'CNE-SUE':
            started = time.perf_counter()
            self._append_cne_users(pending)
            self.timings['user_encoding_seconds'] += time.perf_counter() - started
            return np.asarray([self.user_lookup[key] for key in keys], dtype=np.int32)
        if pending:
            started = time.perf_counter()
            output = np.empty((len(pending), *self.user_shape), dtype=np.float32)
            with torch.inference_mode():
                for start in range(0, len(pending), self.user_batch_size):
                    batch = [json.loads(key) for key in pending[start:start + self.user_batch_size]]
                    histories = torch.tensor([item[1] for item in batch], dtype=torch.long, device=self.device)
                    model_ids = torch.tensor([self.corpus.user_ID_dict.get(item[0], 0) for item in batch],
                                             dtype=torch.long, device=self.device)
                    if self.kind == 'NPA':
                        history_embeddings = self._personalized_news(
                            histories.flatten().cpu().numpy(), model_ids.repeat_interleave(self.history_length)
                        ).reshape(len(batch), self.history_length, self.dimension)
                    else:
                        history_embeddings = self.news_device[histories]
                    history_mask = histories != 0
                    encoder = self.model.user_encoder
                    if self.kind == 'NAML':
                        vectors = encoder.attention(history_embeddings)
                    elif self.kind == 'NRMS':
                        h = encoder.multiheadAttention(history_embeddings, history_embeddings,
                                                       history_embeddings, history_mask)
                        # Original F.dropout(...training=self.training) is off in eval mode.
                        vectors = encoder.attention(F.relu(encoder.affine(h), inplace=True))
                    elif self.kind == 'LSTUR':
                        users = torch.tensor([self.corpus.user_ID_dict.get(item[0], 0) for item in batch],
                                             dtype=torch.long, device=self.device)
                        long_term = self.model.user_embedding(users)
                        lengths = history_mask.sum(dim=1).long()
                        lengths_sorted, order = torch.sort(lengths, descending=True)
                        nonempty = int((lengths_sorted > 0).sum().item())
                        vectors = long_term.clone()
                        if nonempty:
                            chosen = order[:nonempty]
                            packed = pack_padded_sequence(history_embeddings[chosen],
                                                          lengths_sorted[:nonempty].cpu(), batch_first=True)
                            _, h = encoder.gru(packed, long_term[chosen].unsqueeze(0))
                            vectors[chosen] = h.squeeze(0)
                    elif self.kind == 'NPA':
                        query = F.relu(encoder.dense(self.model.user_embedding(model_ids)), inplace=True)
                        vectors = encoder.personalizedAttention(history_embeddings, query, history_mask)
                    else:
                        graphs = np.stack([self._history_graph(item[1])[0] for item in batch])
                        augmented = torch.cat([history_embeddings,
                            encoder.proxy_node_embedding.unsqueeze(0).expand(len(batch), -1, -1)], dim=1)
                        vectors = (encoder.gcn(augmented, torch.as_tensor(graphs, device=self.device)) + augmented)
                        vectors = vectors[:, :self.history_length]
                    output[start:start + len(batch)] = vectors.detach().cpu().numpy()
            if not np.isfinite(output).all():
                raise ValueError('Nonfinite user embeddings')
            offset = len(self.user_keys)
            self.user_keys.extend(pending)
            self.user_lookup.update({key: offset + i for i, key in enumerate(pending)})
            self.users = np.concatenate([self.users, output], axis=0)
            self.timings['user_encoding_seconds'] += time.perf_counter() - started
            self._save_cache()
        return np.asarray([self.user_lookup[key] for key in keys], dtype=np.int32)

    def _append_cne_users(self, pending):
        def encode_batch(keys):
            batch = [json.loads(key) for key in keys]
            with torch.inference_mode():
                histories = torch.tensor([item[1] for item in batch], dtype=torch.long, device=self.device)
                history_embeddings = self.news_device[histories]
                encoder = self.model.user_encoder
                graphs = np.stack([self._history_graph(item[1])[0] for item in batch])
                augmented = torch.cat([history_embeddings,
                    encoder.proxy_node_embedding.unsqueeze(0).expand(len(batch), -1, -1)], dim=1)
                vectors = encoder.gcn(augmented, torch.as_tensor(graphs, device=self.device)) + augmented
                return vectors[:, :self.history_length].detach().cpu().numpy()

        metadata = {'identity': self.identity, 'provenance': self.provenance,
                    'files': {'news_embeddings.npy': _file_hash(self.cache_dir / 'news_embeddings.npy')},
                    'news_count': len(self.news),
                    'cache_representation': 'news_vectors_and_history_GCN_features'}
        users, keys = cne_user_cache.append_generation(
            self.cache_dir, self.users, self.user_keys, pending, encode_batch,
            metadata, batch_rows=self.user_batch_size)
        previous = self.users
        self.users, self.user_keys = users, keys
        self.user_lookup = {key: i for i, key in enumerate(keys)}
        cne_user_cache.close_mapping(previous)

    def _save_cache(self):
        if not (self.cache_dir / 'news_embeddings.npy').exists():
            _save_npy(self.cache_dir / 'news_embeddings.npy', self.news)
        if self.kind == 'CNE-SUE':
            if not self.cache_manifest.exists():
                self._append_cne_users([])
            return
        _save_npy(self.cache_dir / 'user_embeddings.npy', self.users)
        _save_json(self.cache_dir / 'user_keys.json', self.user_keys)
        names = ('news_embeddings.npy', 'user_embeddings.npy', 'user_keys.json')
        _save_json(self.cache_manifest, {'identity': self.identity, 'provenance': self.provenance,
            'files': {name: _file_hash(self.cache_dir / name) for name in names},
            'news_count': len(self.news), 'user_history_count': len(self.users),
            'cache_representation': ('CNN_token_features_and_personalized_user_vectors' if self.kind == 'NPA'
                                     else 'news_vectors_and_history_GCN_features' if self.kind == 'CNE-SUE'
                                     else 'news_vectors_and_user_vectors')})

    def _load_cache(self):
        metadata = json.loads(self.cache_manifest.read_text(encoding='utf-8'))
        if metadata['identity'] != self.identity or metadata['provenance'] != self.provenance:
            raise ValueError('Cache identity/provenance mismatch')
        for name, expected in metadata['files'].items():
            if _file_hash(self.cache_dir / name) != expected:
                raise ValueError('Cache integrity mismatch: ' + name)
        self.news = np.load(self.cache_dir / 'news_embeddings.npy', allow_pickle=False,
                            mmap_mode='r' if self.kind == 'NPA' else None)
        if self.kind == 'CNE-SUE':
            self.users, self.user_keys = cne_user_cache.load_generation(self.cache_dir, metadata, verify_hashes=False)
        else:
            self.users = np.load(self.cache_dir / 'user_embeddings.npy', allow_pickle=False)
            self.user_keys = json.loads((self.cache_dir / 'user_keys.json').read_text(encoding='utf-8'))
        self.user_lookup = {key: i for i, key in enumerate(self.user_keys)}
        if len(self.user_lookup) != len(self.user_keys):
            raise ValueError('Duplicate user-history cache keys')
        news_shape = ((len(self.corpus.news_title_text), self.model.news_encoder.max_sentence_length,
                       self.model.news_encoder.cnn_kernel_num) if self.kind == 'NPA'
                      else (len(self.corpus.news_title_text), self.dimension))
        if self.news.shape != news_shape or self.users.shape != (len(self.user_keys), *self.user_shape):
            raise ValueError('Cache shape mismatch')
        if self.news.dtype != np.float32 or self.users.dtype != np.float32:
            raise ValueError('Cache dtype mismatch')

    def _encode_news(self):
        if self.kind != 'NPA':
            return super()._encode_news()
        started = time.perf_counter()
        encoder = self.model.news_encoder
        result = np.lib.format.open_memmap(self.cache_dir / 'news_embeddings.npy', mode='w+', dtype=np.float32,
            shape=(len(self.corpus.news_title_text), encoder.max_sentence_length, encoder.cnn_kernel_num))
        with torch.inference_mode():
            for start in range(0, len(result), self.news_batch_size):
                end = min(start + self.news_batch_size, len(result))
                title = torch.as_tensor(self.corpus.news_title_text[start:end], device=self.device)
                tokens = encoder.word_embedding(title)
                features = encoder.conv(tokens.permute(0, 2, 1)).permute(0, 2, 1)
                if not bool(torch.isfinite(features).all()):
                    raise ValueError('Nonfinite CNN token features')
                result[start:end] = features.cpu().numpy()
        result.flush()
        self.timings['news_encoding_seconds'] = time.perf_counter() - started
        return result

    def _personalized_news(self, news_ids, model_user_ids):
        encoder = self.model.news_encoder
        output = []
        with torch.inference_mode():
            for start in range(0, len(news_ids), self.personalized_batch_size):
                ids = np.asarray(news_ids[start:start + self.personalized_batch_size])
                users = model_user_ids[start:start + len(ids)]
                if self.npa_features_device is None:
                    features = torch.as_tensor(np.asarray(self.news[ids]), device=self.device)
                else:
                    index = torch.as_tensor(ids, dtype=torch.long, device=self.device)
                    features = self.npa_features_device.index_select(0, index)
                mask = torch.as_tensor(self.corpus.news_title_mask[ids], device=self.device)
                query = F.relu(encoder.dense(self.model.user_embedding(users)), inplace=True)
                vector = encoder.personalizedAttention(features, query, mask)
                category = torch.as_tensor(self.corpus.news_category[ids], device=self.device)
                subcategory = torch.as_tensor(self.corpus.news_subCategory[ids], device=self.device)
                # Exact feature_fusion in evaluation mode: dropout is inactive.
                output.append(torch.cat([vector, encoder.category_embedding(category),
                                         encoder.subCategory_embedding(subcategory)], dim=1))
        return torch.cat(output, dim=0)

    def _history_graph(self, padded_history):
        count = int(self.config.category_num)
        size = self.history_length + count
        graph = (np.zeros((size, size), dtype=np.float32) if self.config.no_self_connection
                 else np.eye(size, dtype=np.float32))
        categories = np.full(self.history_length, count, dtype=np.int64)
        mask = np.zeros(count + 1, dtype=bool)
        nonempty = sum(int(x != 0) for x in padded_history)
        for i in range(nonempty):
            category = int(self.corpus.news_category[padded_history[i]])
            categories[i], mask[category] = category, True
            graph[i, self.history_length + category] = graph[self.history_length + category, i] = 1
            for j in range(i + 1, nonempty):
                other = int(self.corpus.news_category[padded_history[j]])
                if category == other:
                    graph[i, j] = graph[j, i] = 1
                else:
                    graph[self.history_length + category, self.history_length + other] = 1
                    graph[self.history_length + other, self.history_length + category] = 1
        if nonempty and not self.config.no_adjacent_normalization:
            if self.config.gcn_normalization_type == 'asymmetric':
                inverse = np.diag(1 / graph.sum(axis=1))
                graph = inverse @ graph
            else:
                inverse = np.diag(np.sqrt(1 / graph.sum(axis=1)))
                graph = inverse @ graph @ inverse
        return graph, mask, categories

    def _score_flat(self, news_ids, user_indices, on_scored=None):
        started = time.perf_counter()
        result = np.empty(len(news_ids), dtype=np.float32)
        with torch.inference_mode():
            if self.kind not in ('CNE-SUE', 'NPA'):
                user_vectors = torch.as_tensor(self.users, device=self.device)
                for start in range(0, len(news_ids), self.score_batch_size):
                    end = min(start + self.score_batch_size, len(news_ids))
                    ni = torch.as_tensor(news_ids[start:end], dtype=torch.long, device=self.device)
                    ui = torch.as_tensor(user_indices[start:end], dtype=torch.long, device=self.device)
                    result[start:end] = (self.news_device[ni] * user_vectors[ui]).sum(dim=1).cpu().numpy()
                    cooling.safe_boundary()
                    if on_scored is not None:
                        on_scored(end, len(news_ids))
            elif self.kind == 'NPA':
                user_vectors = torch.as_tensor(self.users, device=self.device)
                model_ids = torch.tensor([self.corpus.user_ID_dict.get(json.loads(key)[0], 0)
                                          for key in self.user_keys], device=self.device)
                for start in range(0, len(news_ids), self.personalized_batch_size):
                    end = min(start + self.personalized_batch_size, len(news_ids))
                    ui = torch.as_tensor(user_indices[start:end], dtype=torch.long, device=self.device)
                    vectors = self._personalized_news(news_ids[start:end], model_ids[ui])
                    result[start:end] = (vectors * user_vectors[ui]).sum(dim=1).cpu().numpy()
                    cooling.safe_boundary()
                    if on_scored is not None:
                        on_scored(end, len(news_ids))
            else:
                encoder = self.model.user_encoder
                boundaries = np.r_[0, np.flatnonzero(np.diff(user_indices)) + 1, len(user_indices)]
                for a, b in zip(boundaries[:-1], boundaries[1:]):
                    index = int(user_indices[a])
                    history = json.loads(self.user_keys[index])[1]
                    _, category_mask, category_indices = self._history_graph(history)
                    category_mask[-1] = True
                    mask = torch.as_tensor(category_mask, device=self.device)
                    indices = torch.as_tensor(category_indices, device=self.device)
                    # One bounded history copy avoids a writable tensor view of
                    # the immutable memory-mapped cache (50 x dimension only).
                    feature = torch.tensor(self.users[index], device=self.device)
                    key = encoder.intraCluster_K(feature)
                    for start in range(int(a), int(b), self.personalized_batch_size):
                        end = min(start + self.personalized_batch_size, int(b)); n = end - start
                        news = self.news_device[torch.as_tensor(news_ids[start:end], dtype=torch.long, device=self.device)]
                        query = encoder.intraCluster_Q(news)
                        attention = torch.bmm(key.unsqueeze(0).expand(n, -1, -1), query.unsqueeze(2)).squeeze(2)
                        attention = (attention / encoder.attention_scalar).unsqueeze(1)
                        expanded_indices = indices.view(1, 1, -1).expand(n, 1, -1)
                        weights = scatter_softmax(attention, expanded_indices, 2).unsqueeze(3)
                        intra = scatter_sum(weights * feature.view(1, 1, self.history_length, self.dimension),
                                            expanded_indices, dim=2, dim_size=encoder.category_num)
                        intra = F.relu(encoder.clusterFeatureAffine(intra), inplace=True) + intra
                        user = encoder.interClusterAttention(intra[:, 0], news, mask=mask.unsqueeze(0).expand(n, -1))
                        result[start:end] = (user * news).sum(dim=1).cpu().numpy()
                        cooling.safe_boundary()
                        if on_scored is not None:
                            on_scored(end, len(news_ids))
        return result, time.perf_counter() - started

    @_lstur_precision_entry
    def evaluate_file(self, behavior_path, output_prefix, *, progress=None):
        """Evaluate a TSV without adding millions of candidates to MIND_Corpus.

        Returns aggregate metrics and writes flat candidate arrays + row offsets,
        per-entry metrics, stable ranks, official-format truth and row metadata.
        No outputs are overwritten. Unknown news and invalid labels fail closed.
        """
        cooling_start = cooling.marker()
        started = time.perf_counter()
        self._check_user_mapping()
        if model_fingerprint(self.model) != self.model_hash:
            raise ValueError('Model changed after cache construction; create a new evaluator')
        if corpus_fingerprint(self.corpus) != self.provenance['corpus_sha256']:
            raise ValueError('Corpus changed after cache construction; create a new evaluator')
        self.model.eval()
        behavior_path, prefix = Path(behavior_path), Path(output_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        suffixes = ('metrics.json', 'scores.npy', 'labels.npy', 'offsets.npy', 'news_indices.npy',
                    'user_indices.npy', 'entry_metrics.npy', 'records.jsonl', 'predictions.txt', 'truth.txt')
        paths = {name: Path(str(prefix) + '.' + name) for name in suffixes}
        if any(path.exists() for path in paths.values()):
            raise ValueError('Output prefix already has artifacts; choose a new prefix')
        records, keys, news_chunks, label_chunks, offsets = [], [], [], [], [0]
        with behavior_path.open('r', encoding='utf-8') as stream:
            for row_index, line in enumerate(stream):
                parts = line.rstrip('\r\n').split('\t')
                if len(parts) != 5:
                    raise ValueError('Expected five behavior columns at row ' + str(row_index + 1))
                impression, user, timestamp, raw_history, raw_candidates = parts
                history = self._history(raw_history)
                tokens = [item.rsplit('-', 1) for item in raw_candidates.split()]
                if not tokens or any(len(item) != 2 or item[1] not in ('0', '1') for item in tokens):
                    raise ValueError('Invalid labeled candidates at row ' + str(row_index + 1))
                try:
                    news = np.asarray([self.corpus.news_ID_dict[item[0]] for item in tokens], dtype=np.int32)
                except KeyError as exc:
                    raise ValueError('Candidate news missing from the trained corpus mapping') from exc
                labels = np.asarray([int(item[1]) for item in tokens], dtype=np.int8)
                if labels.sum() == 0 or labels.sum() == len(labels):
                    raise ValueError('Every evaluated row needs positive and negative labels')
                if len(set(news.tolist())) != len(news):
                    raise ValueError('Duplicate candidate news in a behavior row')
                key = self._user_key(user, history)
                keys.append(key)
                records.append({'impression_id': impression, 'user_id': user, 'time': timestamp,
                                'history_sha256': hashlib.sha256(raw_history.encode('utf-8')).hexdigest(),
                                'positive_news': [item[0] for item in tokens if item[1] == '1'],
                                'candidate_count': len(news)})
                news_chunks.append(news)
                label_chunks.append(labels)
                offsets.append(offsets[-1] + len(news))
        if not records:
            raise ValueError('Empty evaluation behavior file')
        if progress is not None:
            progress(0, len(records))
        users = self._ensure_users(keys)
        news = np.concatenate(news_chunks)
        labels = np.concatenate(label_chunks)
        del news_chunks, label_chunks, keys
        offsets = np.asarray(offsets, dtype=np.int64)
        flat_users = np.repeat(users, np.diff(offsets))
        if progress is None:
            scores, dot_seconds = self._score_flat(news, flat_users)
        else:
            def on_scored(done, total):
                progress(0, len(records), scored=done, score_total=total)
            scores, dot_seconds = self._score_flat(news, flat_users, on_scored=on_scored)
        if not np.isfinite(scores).all():
            raise ValueError('Nonfinite cached logits')
        del flat_users
        entry_metrics = np.empty((len(records), len(METRIC_NAMES)), dtype=np.float64)
        with paths['predictions.txt'].open('x', encoding='utf-8', newline='\n') as prediction, \
             paths['truth.txt'].open('x', encoding='utf-8', newline='\n') as truth, \
             paths['records.jsonl'].open('x', encoding='utf-8', newline='\n') as row_file:
            for i, record in enumerate(records):
                start, end = int(offsets[i]), int(offsets[i + 1])
                order = np.argsort(-scores[start:end], kind='stable')
                ranks = np.empty(end - start, dtype=np.int32)
                ranks[order] = np.arange(1, end - start + 1)
                entry_metrics[i] = metrics_from_ranks(labels[start:end], ranks)
                prediction.write(str(i + 1) + ' ' + json.dumps(ranks.tolist(), separators=(',', ':')) + '\n')
                truth.write(str(i + 1) + ' ' + json.dumps(labels[start:end].tolist(), separators=(',', ':')) + '\n')
                row_file.write(json.dumps(record, ensure_ascii=False) + '\n')
                cooling.safe_boundary()
                if progress is not None and i + 1 < len(records):
                    progress(i + 1, len(records))
        for name, value in (('scores', scores), ('labels', labels), ('offsets', offsets),
                            ('news_indices', news), ('user_indices', users), ('entry_metrics', entry_metrics)):
            _save_npy(paths[name + '.npy'], value)
        report = {
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'metrics': dict(zip(METRIC_NAMES, entry_metrics.mean(axis=0).tolist())),
            'rows': len(records), 'candidates': len(news),
            'input': {'path': str(behavior_path.resolve()), 'sha256': _file_hash(behavior_path)},
            'implementation_adjustments': getattr(self.model, '_tors_batch_alignment_fixes', []),
            'evaluation_precision': self.provenance['evaluation_precision'],
            'cache_identity': self.identity, 'cache_dir': str(self.cache_dir.resolve()),
            'cache_loaded_from_disk': self.cache_loaded,
            'cached_news_count': len(self.news), 'cached_user_history_count': len(self.users),
            'elapsed_seconds': time.perf_counter() - started,
            'cached_scoring_seconds': dot_seconds,
            'dot_product_seconds': dot_seconds if self.kind not in ('CNE-SUE', 'NPA') else None,
            'cache_timings_cumulative': dict(self.timings),
            'runtime_memory': self.runtime_memory_report(),
            'metric_order': list(METRIC_NAMES), 'tie_policy': 'stable_original_candidate_order',
            'files': {name: str(path.resolve()) for name, path in paths.items()},
            'scope': 'Evaluation of this checkpoint/input only; not historical paper reproduction',
        }
        report.update(cooling.duration_fields(report['elapsed_seconds'], cooling_start))
        _save_json(paths['metrics.json'], report)
        return report

