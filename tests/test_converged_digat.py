"""DIGAT seed isolation, upstream schedule and owned-checkpoint safety."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_converged_digat import (learning_rate_epoch, remove_owned_checkpoint, signature, validate_assets,
                                offload_optimizer_state, restore_optimizer_devices,
                                configure_development, corpus_behavior_overrides, verify_resume_development, fingerprint)


class ConvergedDigatTests(unittest.TestCase):
    def test_development_protocol_and_path_follow_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            for train, filename in (("imp", "behaviors.tsv"), ("rs50", "behaviors_rs50.tsv")):
                args = SimpleNamespace(data_root=root, train_type=train, dev_protocol="auto", dev_behaviors=None)
                self.assertEqual(configure_development(args), root / "dev" / filename)
                self.assertEqual(args.dev_protocol, train)
            conflict = SimpleNamespace(data_root=root, train_type="rs50", dev_protocol="imp", dev_behaviors=None)
            with self.assertRaisesRegex(ValueError, "matching"):
                configure_development(conflict)

    def test_development_cannot_be_redirected_to_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            args = SimpleNamespace(data_root=root, train_type="rs50", dev_protocol="auto",
                                   dev_behaviors=root / "test/behaviors_rs50.tsv")
            with self.assertRaisesRegex(ValueError, "held-out"):
                configure_development(args)

    def test_corpus_reads_the_selected_dev_candidates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            mapping = corpus_behavior_overrides(root, "rs50")
            self.assertEqual(mapping[root / "train/behaviors.tsv"], root / "train/behaviors_rs50.tsv")
            self.assertEqual(mapping[root / "dev/behaviors.tsv"], root / "dev/behaviors_rs50.tsv")
            self.assertNotIn(root / "test/behaviors.tsv", mapping)
            self.assertNotIn(root / "dev/news.tsv", mapping)

    def test_old_randS_imp_dev_resume_is_rejected_before_snapshot_dispatch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / "experiment.json").write_text(json.dumps({"selection_protocol": {"candidates": "imp"}}))
            args = SimpleNamespace(run_dir=root, resume=True, dev_protocol="rs50",
                                   dev_behaviors=root / "absent-randS-dev.tsv")
            with self.assertRaisesRegex(ValueError, "protocol differs"):
                verify_resume_development(args)

    def test_matching_resume_rejects_modified_dev_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            dev = root / "behaviors_rs50.tsv";dev.write_bytes(b"synthetic held-out candidates")
            (root / "experiment.json").write_text(json.dumps({
                "selection_protocol": {"candidates": "rs50", "file": fingerprint(dev)}}))
            args = SimpleNamespace(run_dir=root, resume=True, dev_protocol="rs50", dev_behaviors=dev)
            verify_resume_development(args)
            dev.write_bytes(b"modified candidates")
            with self.assertRaisesRegex(ValueError, "input changed"):
                verify_resume_development(args)


    def test_upstream_final_ten_percent_lr_boundary(self):
        self.assertEqual(learning_rate_epoch(50), 46)
        self.assertEqual(learning_rate_epoch(30), 28)
        self.assertEqual(learning_rate_epoch(5), 5)

    def test_other_seed_assets_rejected_before_loading_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = {'seed': 42}
            (root / 'asset_prep.json').write_text(json.dumps(
                {'status': 'complete', 'spec': spec, 'identity': signature(spec)}))
            with self.assertRaisesRegex(ValueError, 'match this training seed'):
                validate_assets(root, root / 'data', root / 'glove', 43)

    def test_checkpoint_pruning_is_limited_to_generated_files_in_new_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run = root / 'newrun'; run.mkdir()
            external = root / 'best_epoch_0001.pt'; external.write_bytes(b'preserve')
            with self.assertRaisesRegex(ValueError, 'outside the generated run scope'):
                remove_owned_checkpoint(run, external)
            checkpoint = run / 'latest_training_state.pt'; checkpoint.write_bytes(b'preserve')
            with self.assertRaisesRegex(ValueError, 'outside the generated run scope'):
                remove_owned_checkpoint(run, checkpoint)
            best = run / 'best_epoch_0001.pt'; best.write_bytes(b'generated')
            remove_owned_checkpoint(run, best)
            self.assertFalse(best.exists())
            self.assertEqual(external.read_bytes(), b'preserve')
            self.assertEqual(checkpoint.read_bytes(), b'preserve')

    def test_adam_offload_roundtrip_preserves_following_update(self):
        import copy
        import torch
        parameter = torch.nn.Parameter(torch.tensor([0.3, -0.7]))
        optimizer = torch.optim.Adam([parameter], lr=0.01)
        for _ in range(2):
            optimizer.zero_grad()
            parameter.square().sum().backward()
            optimizer.step()
        peer = torch.nn.Parameter(parameter.detach().clone())
        peer_optimizer = torch.optim.Adam([peer], lr=0.01)
        peer_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        before = copy.deepcopy(optimizer.state_dict())
        locations = offload_optimizer_state(optimizer)
        restore_optimizer_devices(optimizer, locations)
        self.assertEqual(id(optimizer.param_groups[0]['params'][0]), id(parameter))
        for key in before['state'][0]:
            self.assertTrue(torch.equal(before['state'][0][key], optimizer.state_dict()['state'][0][key]))
        for value, opt in ((parameter, optimizer), (peer, peer_optimizer)):
            opt.zero_grad()
            (value.square().sum() + value.sum()).backward()
            opt.step()
        self.assertTrue(torch.equal(parameter, peer))
        for key in optimizer.state_dict()['state'][0]:
            self.assertTrue(torch.equal(optimizer.state_dict()['state'][0][key], peer_optimizer.state_dict()['state'][0][key]))


if __name__ == '__main__':
    unittest.main()
