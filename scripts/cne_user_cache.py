"""Bounded-memory CNE user features with immutable, manifest-selected generations.

One evaluator writes a cache at a time. A generation is committed only by the
final atomic manifest replacement. Previous generations and failed partial files
are retained as evidence; no mapped data file is ever replaced or removed.
"""
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

STORAGE_VERSION = 'cne-user-generations-v1'
COPY_CHUNK_BYTES = 16 * 1024 * 1024


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def close_mapping(value):
    mapping = getattr(value, '_mmap', None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def _selected_path(directory, name):
    if not isinstance(name, str) or Path(name).name != name:
        raise ValueError('Invalid CNE generation filename')
    return Path(directory) / name


def load_generation(directory, metadata, verify_hashes=True):
    if metadata.get('user_storage') != STORAGE_VERSION:
        raise ValueError('Unsupported CNE user cache storage')
    data_name, keys_name = metadata['user_data_file'], metadata['user_keys_file']
    for name in (data_name, keys_name):
        path = _selected_path(directory, name)
        if name not in metadata['files']:
            raise ValueError('CNE generation is absent from hash manifest')
        if verify_hashes and file_hash(path) != metadata['files'][name]:
            raise ValueError('CNE generation integrity mismatch: ' + name)
    keys = json.loads(_selected_path(directory, keys_name).read_text(encoding='utf-8'))
    if len(keys) != metadata['user_history_count'] or len(set(keys)) != len(keys):
        raise ValueError('Invalid CNE generation keys')
    # NumPy cannot map an empty data extent on every supported platform.
    mode = 'r' if keys else None
    users = np.load(_selected_path(directory, data_name), allow_pickle=False, mmap_mode=mode)
    if users.dtype != np.float32 or len(users) != len(keys):
        close_mapping(users)
        raise ValueError('Invalid CNE generation shape or dtype')
    return users, keys


def append_generation(directory, old_users, old_keys, pending, encode_batch,
                      metadata, batch_rows=256, copy_chunk_bytes=COPY_CHUNK_BYTES):
    """Copy old rows and encode new rows without a full-array heap allocation.

    encode_batch receives a slice of pending keys and must return fp32 features.
    Neither caller keys/users nor the committed manifest change on write failure.
    The returned read-only map owns its handle; callers release their old map only
    after swapping to the committed result. Concurrent writers are unsupported.
    """
    directory = Path(directory)
    if batch_rows < 1 or copy_chunk_bytes < 1:
        raise ValueError('Positive bounded chunk sizes required')
    if old_users.dtype != np.float32 or len(old_users) != len(old_keys):
        raise ValueError('Old CNE user cache shape or dtype mismatch')
    keys = list(old_keys) + list(pending)
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate CNE generation keys')
    token = uuid.uuid4().hex
    data_name, keys_name = 'users-' + token + '.npy', 'user-keys-' + token + '.json'
    data_path, keys_path = directory / data_name, directory / keys_name
    partial_data = directory / (data_name + '.incomplete')
    partial_keys = directory / (keys_name + '.incomplete')
    partial_manifest = directory / ('manifest-' + token + '.json.incomplete')
    shape = (len(keys), *old_users.shape[1:])
    result = None
    try:
        if keys:
            result = np.lib.format.open_memmap(partial_data, mode='w+', dtype=np.float32, shape=shape)
            row_bytes = max(1, int(np.prod(shape[1:])) * np.dtype(np.float32).itemsize)
            copy_rows = max(1, int(copy_chunk_bytes) // row_bytes)
            for start in range(0, len(old_keys), copy_rows):
                block = old_users[start:start + copy_rows]
                if not np.isfinite(block).all():
                    raise ValueError('Nonfinite prior CNE user features')
                result[start:start + len(block)] = block
            for start in range(0, len(pending), batch_rows):
                selected = pending[start:start + batch_rows]
                block = encode_batch(selected)
                expected = (len(selected), *shape[1:])
                if block.dtype != np.float32 or block.shape != expected:
                    raise ValueError('CNE encoded user feature shape or dtype mismatch')
                if not np.isfinite(block).all():
                    raise ValueError('Nonfinite CNE user features')
                offset = len(old_keys) + start
                result[offset:offset + len(selected)] = block
            result.flush()
        else:
            with partial_data.open('xb') as stream:
                np.save(stream, old_users, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        # Windows disallows replacing/renaming the data file while it is mapped.
        close_mapping(result)
    with partial_data.open('r+b') as stream:
        os.fsync(stream.fileno())
    os.replace(str(partial_data), str(data_path))
    _write_json(partial_keys, keys)
    os.replace(str(partial_keys), str(keys_path))
    committed = dict(metadata)
    committed.update(user_storage=STORAGE_VERSION, user_data_file=data_name,
                     user_keys_file=keys_name, user_history_count=len(keys))
    committed['files'] = dict(metadata['files'])
    committed['files'].update({data_name: file_hash(data_path), keys_name: file_hash(keys_path)})
    _write_json(partial_manifest, committed)
    # Manifest is the sole commit point for indices, keys, hashes and data.
    os.replace(str(partial_manifest), str(directory / 'manifest.json'))
    return load_generation(directory, committed, verify_hashes=False)
