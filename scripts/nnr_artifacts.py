"""Read-only, standard-library checks for explicitly selected TORS inputs."""
import hashlib
from pathlib import Path
from context_feature_policy import context_feature_spec

EVAL_TYPES = ('imp', 'tt', 'rs30', 'rs50', 'rs60', 'rs90', 'rs120', 'rs150')
MODELS = {'NPA': ('PNE', 'PUE'), 'NAML': ('NAML', 'ATT'),
          'NRMS': ('MHSA', 'MHSA'), 'LSTUR': ('CNN', 'LSTUR'),
          'CNE-SUE': ('CNE', 'SUE')}

def behavior_name(kind):
    if kind not in EVAL_TYPES + ('rs50',):
        raise ValueError('Unknown behavior protocol: ' + kind)
    return 'behaviors.tsv' if kind == 'imp' else 'behaviors_' + kind + '.tsv'

def fingerprint(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}

def check_inputs(data_root, train_type, dev_type, test_type):
    """Validate all rows, without repairing or resampling any source data.

    The same all-split news union as the historical corpus loader is used.
    A shared user/time/history fingerprint across dev/test requires review.
    This check cannot establish the historical provenance of a split.
    """
    root = Path(data_root).resolve()
    kinds = {'train': train_type, 'dev': dev_type, 'test': test_type}
    selected = {split: root / split / behavior_name(kind) for split, kind in kinds.items()}
    context = context_feature_spec(root)
    assets = ('news.tsv', 'entity_embedding.vec') + (('context_embedding.vec',) if context['required'] else ())
    required = list(selected.values()) + [root / split / name for split in kinds for name in assets]
    if not context['required']:
        required.append(root / 'manifest.json')
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError('Missing explicit inputs; no automatic preparation: ' + ', '.join(missing))
    news_ids = set()
    for split in kinds:
        with (root / split / 'news.tsv').open(encoding='utf-8') as stream:
            for number, row in enumerate(stream, 1):
                fields = row.rstrip('\r\n').split('\t')
                if len(fields) != 8 or not fields[0]:
                    raise ValueError(f'{split}/news.tsv:{number}: expected 8 columns')
                news_ids.add(fields[0])
    result = {'files': [fingerprint(path) for path in required], 'splits': {}}
    if not context['required']:
        result['context_feature_policy'] = context
    sessions = {}
    for split, path in selected.items():
        keys = set()
        count = candidates = positives = 0
        with path.open(encoding='utf-8') as stream:
            for number, row in enumerate(stream, 1):
                fields = row.rstrip('\r\n').split('\t')
                if len(fields) != 5:
                    raise ValueError(f'{split}:{number}: expected 5 behavior columns')
                impression, user, timestamp, history, candidate_text = fields
                if not impression or not user or not timestamp:
                    raise ValueError(f'{split}:{number}: missing impression/user/time')
                labels = []
                refs = history.split()
                candidate_ids = []
                for candidate in candidate_text.split():
                    parts = candidate.rsplit('-', 1)
                    if len(parts) != 2 or parts[1] not in ('0', '1'):
                        raise ValueError(f'{split}:{number}: unlabeled or invalid candidate')
                    candidate_ids.append(parts[0]); labels.append(int(parts[1]))
                if len(candidate_ids) != len(set(candidate_ids)):
                    raise ValueError(f'{split}:{number}: duplicate candidate IDs')
                if set(labels) != {0, 1}:
                    raise ValueError(f'{split}:{number}: both positive and negative candidates are required')
                if any(item not in news_ids for item in refs + candidate_ids):
                    raise ValueError(f'{split}:{number}: news reference missing from the all-split news union')
                keys.add(hashlib.sha256(('\t'.join((user, timestamp, history))).encode()).digest())
                count += 1; candidates += len(labels); positives += sum(labels)
        if not count:
            raise ValueError(split + ': no behavior rows')
        result['splits'][split] = {'protocol': kinds[split], 'rows': count, 'candidates': candidates, 'positive_labels': positives}
        sessions[split] = keys
    overlap = len(sessions['dev'] & sessions['test'])
    if overlap:
        raise ValueError(f'Dev/test share {overlap} user/time/history fingerprints; split provenance must be resolved before running')
    result['dev_test_session_overlap'] = overlap
    return result
