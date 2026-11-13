"""Resolve model training batch defaults from the preserved implementation.

Cached evaluation has separate resource settings. A training override is explicit
and is never silently reduced when GPU memory is insufficient.
"""
import ast
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS = {
    'NAML': 'NNR-main/config.py',
    'NRMS': 'NNR-main/config.py',
    'LSTUR': 'NNR-main/config.py',
    'NPA': 'NNR-main/config.py',
    'CNE-SUE': 'NNR-main/config.py',
    'DIGAT': 'third_party/DIGAT/config.py',
}


def resolve_training_batch(model, override=None, root=ROOT):
    if model not in MODEL_CONFIGS:
        raise ValueError('Unknown model: ' + str(model))
    if override is not None and (type(override) is not int or override < 1):
        raise ValueError('Training batch override must be a positive integer')
    relative = MODEL_CONFIGS[model]
    raw = (Path(root) / relative).read_bytes()
    tree = ast.parse(raw.decode('utf-8-sig'))
    matches = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'):
            continue
        if not any(isinstance(arg, ast.Constant) and arg.value == '--batch_size'
                   for arg in node.args):
            continue
        defaults = [kw.value for kw in node.keywords if kw.arg == 'default']
        if len(defaults) != 1:
            raise ValueError('Expected one explicit batch_size default in ' + relative)
        matches.append(ast.literal_eval(defaults[0]))
    if len(matches) != 1 or type(matches[0]) is not int or matches[0] < 1:
        raise ValueError('Ambiguous or invalid batch_size default in ' + relative)
    default = matches[0]
    return {'value': default if override is None else override,
            'source': relative, 'source_sha256': hashlib.sha256(raw).hexdigest(),
            'default_value': default, 'overridden': override is not None}
