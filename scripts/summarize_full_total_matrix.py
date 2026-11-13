"""Full-Total score-artifact validation and matrix-report utilities."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

METRICS = ('AUC', 'MRR', 'nDCG@5', 'nDCG@10')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def independent_metrics(labels, scores):
    """Original scorer uses unique integer ranks, resolving logit ties by input order."""
    order = np.argsort(-scores, kind='stable')
    ordered = labels[order]
    positive = np.flatnonzero(ordered == 1) + 1
    n_pos, n_neg = len(positive), len(labels)-len(positive)
    if not n_pos or not n_neg:
        raise ValueError('Degenerate sampled evaluation row')
    negatives_before = positive - np.arange(1, n_pos+1)
    auc = float((n_neg-negatives_before).sum())/(n_pos*n_neg)
    result = [auc, float(np.reciprocal(positive.astype(float)).mean())]
    for k in (5, 10):
        weights = 1.0/np.log2(np.arange(2, min(k, len(ordered))+2))
        dcg = float(np.dot(ordered[:k], weights))
        ideal = float((1.0/np.log2(np.arange(2, min(k, n_pos)+2))).sum())
        result.append(dcg/ideal)
    return np.asarray(result)


def validate_metrics(prefix, expected_input):
    prefix = Path(prefix)
    report_path = Path(str(prefix)+'.metrics.json')
    report = read(report_path)
    if report.get('input', {}).get('sha256') != expected_input:
        raise ValueError('Evaluation input identity mismatch: '+str(prefix))
    entries = np.load(str(prefix)+'.entry_metrics.npy', mmap_mode='r', allow_pickle=False)
    if entries.shape != (report['rows'], 4) or not np.isfinite(entries).all():
        raise ValueError('Invalid per-entry metrics: '+str(prefix))
    if ((entries < 0) | (entries > 1)).any():
        raise ValueError('Metric out of range')
    expected = np.asarray([report['metrics'][key] for key in METRICS])
    if not np.allclose(entries.mean(axis=0), expected, rtol=0, atol=1e-10):
        raise ValueError('Per-entry mean mismatch')
    offsets = np.load(str(prefix)+'.offsets.npy', mmap_mode='r', allow_pickle=False)
    scores = np.load(str(prefix)+'.scores.npy', mmap_mode='r', allow_pickle=False)
    labels = np.load(str(prefix)+'.labels.npy', mmap_mode='r', allow_pickle=False)
    if offsets.shape != (report['rows']+1,) or offsets[0] != 0 or offsets[-1] != report['candidates']:
        raise ValueError('Invalid offsets')
    if (np.diff(offsets) < 2).any() or len(scores) != offsets[-1] or len(labels) != len(scores):
        raise ValueError('Score/label count mismatch')
    chosen = sorted(set(np.linspace(0, len(entries)-1, min(32,len(entries)), dtype=int).tolist()))
    for i in chosen:
        a, b = int(offsets[i]), int(offsets[i+1])
        if not np.isfinite(scores[a:b]).all() or not np.isin(labels[a:b], (0,1)).all():
            raise ValueError('Invalid sampled raw score/label')
        if not np.allclose(independent_metrics(labels[a:b], scores[a:b]), entries[i], rtol=0, atol=1e-10):
            raise ValueError('Independent raw-score metric mismatch: %s row %d' % (prefix, i))
    return report, {'metrics_sha256': digest(report_path),
                    'entry_metrics_sha256': digest(str(prefix)+'.entry_metrics.npy'),
                    'raw_score_rows_checked': chosen,
                    'scope': 'all entry metrics and aggregate means; 32 deterministic raw-score rows per combination'}


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)




