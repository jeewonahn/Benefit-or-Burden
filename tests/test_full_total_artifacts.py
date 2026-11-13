"""Full-Total artifact ordering, weighted aggregation and resume-integrity checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evaluate_existing_digat import (ARTIFACTS, fingerprint, merge_chunks,
                                    shard_input, validated_chunk, validate_candidate_manifest)


class FullTotalArtifactsTests(unittest.TestCase):
    def test_shard_input_preserves_rows_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'input.tsv'
            source.write_bytes(b'a\r\nb\r\nc\r\n')
            records = list(shard_input(source, root / 'chunks', 2))
            self.assertEqual([r['rows'] for r in records], [2, 1])
            self.assertEqual([r['start_row'] for r in records], [0, 2])
            self.assertEqual(Path(records[0]['input']['path']).read_bytes(), b'a\nb\n')
            self.assertEqual(records, list(shard_input(source, root / 'chunks', 2)))
            Path(records[0]['input']['path']).write_bytes(b'changed\n')
            with self.assertRaisesRegex(ValueError, 'hash/size'):
                list(shard_input(source, root / 'chunks', 2))

    def test_candidate_manifest_rejects_total5_label(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'manifest.json'
            target.write_text(json.dumps({'status': 'complete', 'protocol': 'total5'}))
            with self.assertRaisesRegex(ValueError, 'FULL Total'):
                validate_candidate_manifest(target, Path(temp) / 'input.tsv', {})

    def _record(self, root, index, values, candidate_counts):
        directory = root / ('chunk%d' % index)
        directory.mkdir()
        prefix = directory / 'test_total'
        rows, count = len(values), sum(candidate_counts)
        input_path = directory / 'input.tsv'
        input_path.write_text('synthetic\n' * rows)
        files = {name: str(prefix) + '.' + name for name in ARTIFACTS + ('metrics.json',)}
        arrays = {'scores': np.arange(count, dtype=np.float32),
                  'labels': np.array([1] + [0] * (count - 1), dtype=np.int8),
                  'offsets': np.concatenate(([0], np.cumsum(candidate_counts))).astype(np.int64),
                  'news_indices': np.arange(count, dtype=np.int32),
                  'user_indices': np.arange(rows, dtype=np.int64),
                  'entry_metrics': np.repeat(np.array(values, dtype=np.float64)[:, None], 4, axis=1)}
        for name, array in arrays.items():
            np.save(files[name + '.npy'], array)
        for name in ('predictions.txt', 'truth.txt'):
            Path(files[name]).write_text(''.join('%d [1,2]\n' % (i + 1) for i in range(rows)))
        Path(files['records.jsonl']).write_text(''.join(json.dumps({'impression_id': str(i)}) + '\n' for i in range(rows)))
        report = {'rows': rows, 'candidates': count, 'input': fingerprint(input_path),
                  'cache_identity': 'same-checkpoint', 'elapsed_seconds': 2.0, 'files': files}
        Path(files['metrics.json']).write_text(json.dumps(report))
        return {'index': index, 'rows': rows, 'input': fingerprint(input_path),
                'report': fingerprint(files['metrics.json']),
                'artifacts': {name: fingerprint(files[name]) for name in ARTIFACTS}}

    def test_merge_uses_entry_weights_global_offsets_and_global_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            records = [self._record(root, 0, [0.0, 0.0], [2, 3]),
                       self._record(root, 1, [1.0], [4])]
            candidate = {'statistics': {'rows': 3, 'candidates': 9},
                         'output': {'path': 'full.tsv', 'sha256': 'source', 'bytes': 0}}
            report = merge_chunks(records, 'full.tsv', root / 'merged' / 'test_total', candidate, 'same-checkpoint')
            self.assertAlmostEqual(report['metrics']['MRR'], 1 / 3)
            self.assertEqual(np.load(report['files']['offsets.npy']).tolist(), [0, 2, 5, 9])
            self.assertEqual(Path(report['files']['predictions.txt']).read_text().splitlines(),
                             ['1 [1,2]', '2 [1,2]', '3 [1,2]'])
            self.assertEqual(report['rows'], 3)
            self.assertEqual(report['elapsed_seconds'], 4.0)
            with self.assertRaisesRegex(ValueError, 'Merged output exists'):
                merge_chunks(records, 'full.tsv', root / 'merged' / 'test_total', candidate, 'same-checkpoint')

    def test_completed_chunk_tamper_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            record = self._record(Path(temp), 0, [0.5], [2])
            validated_chunk(record)
            Path(record['artifacts']['scores.npy']['path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash/size'):
                validated_chunk(record)


if __name__ == '__main__':
    unittest.main()
