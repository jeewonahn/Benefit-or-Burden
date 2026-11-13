"""Build portable shared DIGAT news/word/SAG assets from registered local inputs.

This adapter leaves vendor source unchanged. It deliberately builds only the
six shared artifacts: user-history graphs belong to individual training runs.
No model library is imported in --check-only; execution never downloads assets.
"""
import argparse
from collections import OrderedDict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import tempfile
import sys
import time
import traceback

from prepare_model_assets import (
    GLOVE_MANIFEST, SEMANTIC_MANIFEST, fingerprint, now, read, save,
    verify_asset_manifest, verify_semantic_snapshot,
)

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = (
    'news_ID-MIND-small.json', 'category-MIND-small.json',
    'subCategory-MIND-small.json', 'vocabulary-3-32-MIND-small.json',
    'word_embedding-3-300-32-MIND-small.pkl', 'news_graph-2-5-MIND-small.pkl',
)
METADATA_FILES = REQUIRED_FILES[:-1] + ('user_ID-MIND-small.json',)


def after_completed_work(function, boundary):
    """Keep the original result and rest only after successful completed work."""
    def completed(*positional, **keywords):
        result = function(*positional, **keywords)
        boundary()
        return result
    return completed


def _legacy_helpers():
    # These helpers use the standard library at import time. Heavy vendor/model
    # imports remain isolated in execute_backend below.
    from digat_small_assets import config_for, identity, shared_spec
    return config_for, identity, shared_spec


def validate_glove_vocabulary(vectors):
    """Accept 400k ordinary tokens plus a certified optional final <unk>."""
    rows = vectors.get('rows')
    ordinary = vectors.get('ordinary_rows', rows)
    unknown = vectors.get('unknown_rows', 0)
    if (vectors.get('dimensions') != 300 or ordinary != 400000
            or unknown not in (0, 1) or rows != ordinary + unknown):
        raise ValueError('Register 400000 ordinary GloVe 6B 300d vectors, with at most one certified final <unk>')

def build_spec(args):
    """Bind news order, local pretrained bytes, seed, source and builder options."""
    for split in ('train', 'dev', 'test'):
        if not (args.data_root / split / 'news.tsv').is_file():
            raise ValueError('Missing news.tsv for ' + split)
    if not (args.data_root / 'train' / 'behaviors.tsv').is_file():
        raise ValueError('IMP master train/behaviors.tsv is required for DIGAT metadata')
    glove = verify_asset_manifest(args.embedding_root, GLOVE_MANIFEST, 'glove.6B.300d')
    validate_glove_vocabulary(glove.get('vectors', {}))
    semantic = verify_semantic_snapshot(args.sag_model)
    _, _, shared_spec = _legacy_helpers()
    spec = shared_spec(args.data_root, args.embedding_root, args.sag_model, args.seed)
    # Metadata user IDs are not shared by a training run, but still bind all
    # files read by this preparation process, including the master behavior file.
    spec['metadata_user_reference'] = fingerprint(args.data_root / 'train' / 'behaviors.tsv')
    spec['registered_glove_manifest'] = fingerprint(args.embedding_root / GLOVE_MANIFEST)
    spec['registered_semantic_manifest'] = fingerprint(args.sag_model / SEMANTIC_MANIFEST)
    spec['semantic_revision'] = semantic['revision']
    spec['builder'] = {
        'source': fingerprint(Path(__file__)),
        'asset_registration_source': fingerprint(Path(__file__).with_name('prepare_model_assets.py')),
        'legacy_helper_source': fingerprint(Path(__file__).with_name('digat_small_assets.py')),
        'semantic_batch_size': args.semantic_batch_size,
        'semantic_device': 'cuda',
        'shared_only': True,
        'scheduled_long_rest': bool(getattr(args, 'cooling', False)),
    }
    return spec


def verify_prepared(output, manifest, expected_identity):
    if manifest.get('schema') != 1 or manifest.get('status') != 'complete':
        raise ValueError('Shared DIGAT preparation is not complete')
    _, identity, _ = _legacy_helpers()
    if manifest.get('identity') != expected_identity or identity(manifest['spec']) != expected_identity:
        raise ValueError('Shared DIGAT preparation input/source identity differs')
    for name in REQUIRED_FILES:
        actual = fingerprint(output / name)
        expected = manifest.get('files', {}).get(name, {})
        if any(actual[key] != expected.get(key) for key in ('bytes', 'sha256')):
            raise ValueError('Shared artifact differs from its manifest: ' + name)
    return manifest


def add_self_loops_and_save(graph_parts, output_file, graph_size, *, numpy_module):
    """Retain the upstream MIND_Corpus.preprocess SAG finalization exactly."""
    node_ids, graph, mask = graph_parts
    for index in range(graph.shape[0]):
        graph[index] += numpy_module.identity(graph_size, dtype=bool)
    temporary = output_file.with_name(output_file.name + '.part')
    with temporary.open('wb') as stream:
        pickle.dump({'news_node_ID': node_ids, 'news_graph': graph,
                     'news_graph_mask': mask}, stream, protocol=4)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output_file)


def execute_backend(args, manifest, record):
    """Run metadata plus the unmodified vendor SAG function; no user graph."""
    import numpy as np
    import torch
    import scheduled_cooling as cooling

    sys.path.insert(0, str(ROOT / 'third_party' / 'DIGAT'))
    import MIND_corpus
    import construct_SAG

    if not torch.cuda.is_available():
        raise RuntimeError('Upstream DIGAT SAG construction requires a compatible CUDA GPU')
    if getattr(args, 'cooling', False):
        cooling.install(args.output / 'cooling_events.jsonl', torch.cuda.synchronize)
        cooling.set_phase('shared_asset_preparation')
    config_for, _, _ = _legacy_helpers()
    cfg = config_for(args.data_root, args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(4)
    original_glove = MIND_corpus.GloVe
    private_glove_cache = None
    original_construct = MIND_corpus.construct_SAG
    original_encoder = construct_SAG.SentenceTransformer
    original_model_name = construct_SAG.pretrain_model_name
    original_similarities = getattr(construct_SAG, 'generate_cos_similarities', None)
    original_directory = Path.cwd()
    environment = {name: os.environ.get(name) for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')}

    class SAGDeferred(Exception):
        pass

    def defer_sag(*positional, **keywords):
        raise SAGDeferred()

    def local_glove(*positional, **keywords):
        if private_glove_cache is None:
            raise RuntimeError('Private verified GloVe parsing cache was not initialized')
        keywords.update(name='6B', dim=300, cache=private_glove_cache.name)
        return original_glove(*positional, **keywords)

    try:
        os.chdir(args.output)
        MIND_corpus.GloVe = local_glove
        MIND_corpus.construct_SAG = defer_sag
        if not all((args.output / name).is_file() for name in METADATA_FILES):
            if (args.output / REQUIRED_FILES[-1]).exists():
                raise ValueError('Partial SAG exists without complete metadata; choose a fresh asset output')
            # Never trust an unrelated torchtext .pt cache beside the registered
            # text. Parse in an empty per-attempt directory using that exact text.
            private_glove_cache = tempfile.TemporaryDirectory(prefix='glove-parse-', dir=str(args.output))
            vector_source = args.embedding_root / 'glove.6B.300d.txt'
            vector_target = Path(private_glove_cache.name) / vector_source.name
            try:
                os.link(vector_source, vector_target)
                manifest['glove_staging'] = 'hardlink to verified text; fresh parsed cache'
            except OSError:
                shutil.copyfile(vector_source, vector_target)
                manifest['glove_staging'] = 'verified text copy; fresh parsed cache'
            if fingerprint(vector_target) != fingerprint(vector_source):
                raise ValueError('Private GloVe text staging differs from the registered source')
            try:
                MIND_corpus.MIND_Corpus.preprocess(cfg)
            except SAGDeferred:
                pass
            if not all((args.output / name).is_file() for name in METADATA_FILES):
                raise ValueError('Upstream metadata stage did not produce all expected files')
        manifest['phase'] = 'metadata_complete'
        manifest['metadata_files'] = {name: fingerprint(args.output / name) for name in METADATA_FILES}
        record()
        for name in environment:
            os.environ[name] = '1'
        base = original_encoder(str(args.sag_model), device='cuda')
        encoded = OrderedDict()

        class ExactTextCache:
            def encode(self, sentences):
                key = hashlib.sha256(json.dumps(sentences, ensure_ascii=False,
                    separators=(',', ':')).encode('utf-8')).hexdigest()
                if key in encoded:
                    encoded.move_to_end(key)
                    manifest['exact_text_cache_hits'] += 1
                    record()
                    cooling.safe_boundary()
                    return encoded[key]
                started = time.monotonic()
                value = base.encode(sentences, batch_size=args.semantic_batch_size,
                                    show_progress_bar=True)
                encoding_seconds = time.monotonic() - started
                cooling.safe_boundary()
                encoded[key] = value
                while len(encoded) > 2:
                    encoded.popitem(last=False)
                event = {'texts': len(sentences), 'text_list_sha256': key,
                         'seconds': encoding_seconds}
                manifest['semantic_encodings'].append(event)
                record()
                print(json.dumps({'stage': 'semantic_encoded', **event}), flush=True)
                return value

        encoder = ExactTextCache()
        construct_SAG.SentenceTransformer = lambda name: encoder
        construct_SAG.pretrain_model_name = str(args.sag_model)
        if callable(original_similarities):
            construct_SAG.generate_cos_similarities = after_completed_work(
                original_similarities, cooling.safe_boundary)
        news_ids = read(args.output / REQUIRED_FILES[0])
        manifest['phase'] = 'semantic_graph'
        record()
        parts = original_construct(cfg.dataset, cfg.train_root, cfg.dev_root,
                                   cfg.test_root, cfg.SAG_neighbors, cfg.SAG_hops,
                                   cfg.news_graph_size, news_ids)
        if len(news_ids) != parts[1].shape[0]:
            raise ValueError('SAG news count differs from the metadata dictionary')
        add_self_loops_and_save(parts, args.output / REQUIRED_FILES[-1],
                               cfg.news_graph_size, numpy_module=np)
        manifest['news_count'] = len(news_ids) - 1
        manifest['runtime_versions'] = {}
        for name in ('torch', 'torchtext', 'numpy', 'sentence-transformers', 'transformers', 'huggingface-hub'):
            try:
                manifest['runtime_versions'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                manifest['runtime_versions'][name] = None
    finally:
        MIND_corpus.GloVe = original_glove
        MIND_corpus.construct_SAG = original_construct
        construct_SAG.SentenceTransformer = original_encoder
        construct_SAG.pretrain_model_name = original_model_name
        if original_similarities is not None:
            construct_SAG.generate_cos_similarities = original_similarities
        os.chdir(original_directory)
        if private_glove_cache is not None:
            private_glove_cache.cleanup()
        for name, value in environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def prepare(args, *, backend=None, spec_builder=None):
    for name in ('data_root', 'embedding_root', 'sag_model', 'output'):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if not 0 <= args.seed < 2 ** 32 or args.semantic_batch_size < 1:
        raise ValueError('Seed must fit uint32 and semantic batch size must be positive')
    for source in (args.data_root, args.embedding_root, args.sag_model):
        if source == args.output or source in args.output.parents or args.output in source.parents:
            raise ValueError('Use a separate DIGAT output, outside the input folders')
    spec = (spec_builder or build_spec)(args)
    _, identity, _ = _legacy_helpers()
    signature = identity(spec)
    manifest_file = args.output / 'asset_prep.json'
    previous = read(manifest_file) if manifest_file.is_file() else None
    if previous:
        if previous.get('identity') != signature:
            raise ValueError('Existing DIGAT output has different inputs, source or settings')
        if previous.get('status') == 'complete':
            return verify_prepared(args.output, previous, signature)
    if args.check_only:
        return {'status': 'inputs_verified', 'identity': signature,
                'prepared_assets_complete': False, 'execution_requested': False,
                'requires_cuda_for_preparation': True, 'files_required': list(REQUIRED_FILES)}
    if previous and args.resume:
        for name, expected in previous.get('metadata_files', {}).items():
            if fingerprint(args.output / name) != expected:
                raise ValueError('Partial metadata changed before resume: ' + name)
    if previous and not args.resume:
        raise ValueError('Incomplete matching output exists; use --resume or a new output')
    if args.output.exists() and not previous:
        raise ValueError('Unregistered DIGAT output is preserved; choose a new output')
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema': 1, 'status': 'preparing', 'identity': signature, 'spec': spec,
        'created_utc': previous.get('created_utc', now()) if previous else now(),
        'historical_R2_reproduction': False, 'phase': 'metadata',
        'preparation_scope': 'shared news/word/SAG only; user-history graphs are run-local',
        'semantic_encodings': [], 'exact_text_cache_hits': 0,
        'resumed_from_status': previous.get('status') if previous else None,
        'execution_segments': list(previous.get('execution_segments', [])) if previous else [],
    }
    import scheduled_cooling as cooling
    started = time.monotonic()
    segment = {'start_unix': time.time(), 'elapsed_seconds': 0.0}
    manifest['execution_segments'].append(segment)

    def record():
        segment['elapsed_seconds'] = time.monotonic() - started
        manifest['elapsed_seconds'] = sum(row['elapsed_seconds'] for row in manifest['execution_segments'])
        cooling.record_manifest(manifest, segment)
        save(manifest_file, manifest)

    record()
    try:
        (backend or execute_backend)(args, manifest, record)
        manifest['files'] = {name: fingerprint(args.output / name) for name in REQUIRED_FILES}
        manifest['status'] = 'complete'
        manifest['phase'] = 'shared_assets_complete'
        record()
        return verify_prepared(args.output, manifest, signature)
    except BaseException:
        manifest['status'] = 'failed'
        manifest['error'] = traceback.format_exc()
        record()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--embedding-root', type=Path, required=True)
    parser.add_argument('--sag-model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--semantic-batch-size', type=int, default=32)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cooling', action='store_true', help='Long scheduled rests after completed shared-asset work')
    args = parser.parse_args()
    result = prepare(args)
    print(json.dumps({'status': result['status'], 'manifest': str(args.output / 'asset_prep.json'),
                      'check_only': args.check_only}), flush=True)


if __name__ == '__main__':
    main()


