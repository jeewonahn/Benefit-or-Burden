"""Check the sampled cohort, full-source publication proxy and raw request lineage."""
import hashlib
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import prepare_mind_masters as masters
from test_prepare_mind_masters import arguments, fixture, replace_member


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class LargeCohortTests(unittest.TestCase):
    def test_samples_exactly_two_hundred_thousand_unique_users_and_can_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive_members = {}
            for role, ids in [('official_train', range(160_000)), ('official_dev', range(155_000, 202_000))]:
                path = root / role
                with path.open('w', encoding='utf-8') as out:
                    for user in ids:
                        out.write(f'i\tU{user}\tnot read by user sampler\n')
                archive_members[role] = {'behaviors.tsv': {'path': str(path)}}
            first, receipt = masters.sample_large_users(archive_members, 291, root / 'first')
            repeat, _ = masters.sample_large_users(dict(reversed(list(archive_members.items()))), 291, root / 'repeat')
            other, _ = masters.sample_large_users(archive_members, 384, root / 'other')
            self.assertEqual(first, repeat)
            self.assertNotEqual(first, other)
            self.assertEqual(len(first), 200_000)
            self.assertEqual(receipt['population_count'], 202_000)
            self.assertEqual(read(Path(receipt['user_ids']['path'])), sorted(first))
            self.assertTrue(any(int(user[1:]) >= 160_000 for user in first))

    def test_preserves_user_rows_and_full_large_impression_minima(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train, dev, _, *_ = fixture(root)
            seed = 99
            population = ['U0', 'U1', 'U2', 'U3']
            sampled = sorted(random.Random(f'{seed}:MIND-200K:user-cohort:v1').sample(population, 2))
            training_user, test_only = sampled
            outside = sorted(set(population) - set(sampled))
            # Earlier impressions of N3 and all impressions of N4 come only from unsampled users.
            news = ''.join(f'N{i}\tcat\tsub\ttitle\tabstract\turl\t[]\t[]\n' for i in range(1, 6)).encode()
            rows = [f'outside\t{outside[0]}\t11/09/2019 01:00:00 AM\tN5\tN4-1 N3-0\r\n']
            for day in range(9, 15):
                rows.append(f'i{day}\t{training_user}\t11/{day}/2019 03:00:00 PM\tN5\tN1-1 N2-0\r\n')
            rows.append(f'duplicate\t{training_user}\t11/14/2019 03:00:00 PM\tN5\tN2-1 N1-0\r\n')
            final = [f't0\t{outside[1]}\t11/15/2019 01:00:00 AM\t\tN1-1 N2-0\n',
                     f't1\t{test_only}\t11/15/2019 02:00:00 AM\t\tN3-1 N2-0\n']
            for path, values in [(train, rows), (dev, final)]:
                replace_member(path, 'behaviors.tsv', ''.join(values).encode())
                replace_member(path, 'news.tsv', news)
            out = root / 'masters'
            args = arguments(train, dev, None, out, dataset='MIND-200K', seed=seed, split_policy='chronological',
                             max_member_bytes=None, max_total_bytes=None,
                             context_feature_mode=masters.CONTEXT_NOT_USED_BY_PAPER_MODELS, selected_models=['NAML'])
            with patch.object(masters, 'MIND_200K_USERS', 2):
                masters.prepare(args)
            manifest = read(out / 'manifest.json')
            self.assertEqual(manifest['dataset'], 'MIND-200K')
            self.assertEqual(manifest['user_sample']['seed'], seed)
            self.assertEqual(manifest['user_sample']['population_count'], 4)
            self.assertEqual(read(Path(manifest['user_sample']['user_ids']['path'])), sampled)
            self.assertEqual((out / 'train/behaviors.tsv').read_bytes(), ''.join(rows[1:6]).encode())
            self.assertEqual((out / 'dev/behaviors.tsv').read_bytes(), ''.join(rows[6:]).encode())
            self.assertEqual((out / 'test/behaviors.tsv').read_bytes(), final[1].encode())
            source_rows = [json.loads(line)['source_row'] for line in (out/'request_mapping.jsonl').read_text().splitlines()]
            self.assertEqual(source_rows, [2,3,4,5,6,7,8,2])
            self.assertEqual(manifest['source_statistics']['official_train']['rows'], 8)
            self.assertEqual(manifest['cohort_statistics']['official_train']['rows'], 7)
            pin = manifest['full_source_minimum_map']
            path = Path(pin['path'])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), pin['sha256'])
            data = read(path)
            self.assertEqual(data['all_impressions_min']['N3'], '2019-11-09T01:00:00')
            self.assertEqual(data['all_impressions_min']['N4'], '2019-11-09T01:00:00')
            self.assertNotIn('N5', data['all_impressions_min'])
            for role in ('official_train', 'official_dev'):
                source = manifest['source_archives'][role]['extracted_members']['behaviors.tsv']
                self.assertEqual(data['input_files'][role], {k:source[k] for k in ('path','sha256','bytes')})


if __name__ == '__main__':
    unittest.main()
