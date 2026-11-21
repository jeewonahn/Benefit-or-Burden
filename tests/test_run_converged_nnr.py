"""Restart transaction checks for the multi-epoch NNR wrapper."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from convergence_control import EarlyStopping
import run_converged_nnr as runner
from run_converged_nnr import (restore_committed_training, configure_development,
                               nnr_runtime_arguments)


class CommittedNNRRestart(unittest.TestCase):
    def test_last_checkpoint_is_authority_over_uncommitted_json(self):
        control=EarlyStopping();control.update(1,0.4)
        identity={'seed':43,'training':'imp'}
        checkpoint={'schema':'converged_training_v1','identity':identity,'epoch':1,
                    'early_stopping':control.state_dict(),'epoch_records':[{'epoch':1}]}
        manifest={'epoch_records':[{'epoch':1},{'epoch':2}], 'best_epoch':2}
        restored=restore_committed_training(manifest,checkpoint,identity)
        self.assertEqual(manifest['epoch_records'],[{'epoch':1}])
        self.assertEqual(manifest['best_epoch'],1)
        self.assertEqual(restored.last_epoch,1)

    def test_changed_seed_or_inconsistent_committed_history_is_rejected(self):
        control=EarlyStopping();control.update(1,0.4)
        checkpoint={'schema':'converged_training_v1','identity':{'seed':43},'epoch':1,
                    'early_stopping':control.state_dict(),'epoch_records':[{'epoch':1}]}
        with self.assertRaises(ValueError):
            restore_committed_training({},checkpoint,{'seed':42})
        broken=copy.deepcopy(checkpoint);broken['epoch_records']=[]
        with self.assertRaises(ValueError):
            restore_committed_training({},broken,{'seed':43})



class MatchedNNRDevelopment(unittest.TestCase):
    def arguments(self, folder, training, requested="auto", behaviors=None):
        return SimpleNamespace(data_root=Path(folder), train_type=training,
                               dev_protocol=requested, dev_behaviors=behaviors)

    def test_imp_and_rands_select_their_matching_heldout_files(self):
        with tempfile.TemporaryDirectory() as folder:
            for training, filename in (("imp", "behaviors.tsv"), ("rs50", "behaviors_rs50.tsv")):
                args = configure_development(self.arguments(folder, training))
                self.assertEqual(args.dev_protocol, training)
                self.assertEqual(args.dev_behaviors, (Path(folder)/"dev"/filename).resolve())

    def test_conflicting_protocol_or_test_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "requires matching"):
                configure_development(self.arguments(folder, "rs50", "imp"))
            with self.assertRaisesRegex(ValueError, "held-out"):
                configure_development(self.arguments(
                    folder, "rs50", behaviors=Path(folder)/"test/behaviors_rs50.tsv"))

    def test_matching_protocol_reaches_the_legacy_corpus_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            args = configure_development(self.arguments(folder, "rs50"))
            args.model="NAML"; args.embedding_root=Path(folder)/"glove"
            args.seed=42; args.max_epochs=50
            argv=nnr_runtime_arguments(args, {"value":64})
            self.assertEqual(argv[argv.index("--dev_type")+1],"rs50")
            self.assertEqual(argv[argv.index("--train_type")+1],"rs50")
            self.assertEqual(argv[argv.index("--test_type")+1],"imp")
            self.assertEqual(argv[argv.index("--batch_size")+1],"64")
            self.assertEqual(argv[argv.index("--dataset")+1],"small")
            args.matched_candidate_profile = {"dataset": "MIND-200K"}
            argv = nnr_runtime_arguments(args, {"value": 64})
            self.assertEqual(argv[argv.index("--dataset")+1],"200k")

    def test_changed_dev_objective_rejects_old_rands_checkpoint(self):
        control=EarlyStopping(); control.update(1,0.4)
        old={"train_type":"rs50","dev_protocol":"imp","dev_sha256":"old"}
        checkpoint={"schema":"converged_training_v1","identity":old,"epoch":1,
                    "early_stopping":control.state_dict(),"epoch_records":[{"epoch":1}]}
        new={"train_type":"rs50","dev_protocol":"rs50","dev_sha256":"new"}
        with self.assertRaisesRegex(ValueError,"exact restart"):
            restore_committed_training({},checkpoint,new)

    def test_shared_dev_policy_is_snapshotted_and_pinned_for_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"repo"; (root/"scripts").mkdir(parents=True)
            (root/"NNR-main").mkdir()
            controls=("run_converged_nnr.py","convergence_control.py","nnr_artifacts.py",
                      "model_batch_defaults.py","training_validation.py","prepare_development_candidates.py",
                      "scheduled_cooling.py","candidate_policy.py","consistent_candidate_profile.py",
                      "nnr_consistent_preprocessing.py","context_feature_policy.py")
            for name in controls:
                (root/"scripts"/name).write_text("# original control\n",encoding="utf-8")
            output=Path(folder)/"run"; output.mkdir()
            with patch.object(runner,"ROOT",root):
                inventory=runner.source_inventory()
                self.assertIn("scripts/training_validation.py",inventory)
                self.assertIn("scripts/scheduled_cooling.py",inventory)
                runner.snapshot_source(output,inventory)
                runner.check_snapshot(output,inventory)
                for name in ("training_validation.py","candidate_policy.py","consistent_candidate_profile.py",
                             "nnr_consistent_preprocessing.py","context_feature_policy.py"):
                    (root/"scripts"/name).write_text("# changed policy\n",encoding="utf-8")
                    with self.assertRaisesRegex(ValueError,"Restart control source changed"):
                        runner.check_snapshot(output,inventory)
                    (root/"scripts"/name).write_text("# original control\n",encoding="utf-8")




    def test_certificate_is_part_of_the_converged_restart_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            args=configure_development(self.arguments(folder,"rs50"))
            args.model="NAML"; args.embedding_root=Path(folder)/"glove"
            args.seed=42; args.max_epochs=50; args.min_epochs=5
            args.patience=5; args.min_delta=.0001; args.selection_metric="avg"
            args.chunk_rows=5000; args.personalized_batch_size=512; args.npa_features_on_gpu=False
            args.candidate_manifest=Path(folder)/"total_manifest.json"
            args.candidate_manifest.write_text("{}",encoding="utf-8")
            args.matched_candidate_profile={"sha256":"synthetic-certificate","profile":"request_time_history_and_current_positives_v1"}
            identity=runner.config_identity(args,{"files":[{"sha256":"synthetic-dev"}]},
                                            {"output":{"sha256":"synthetic-total"}},{"value":64})
            self.assertEqual(identity["dev_protocol"],"rs50")
            self.assertEqual(identity["dev_behaviors"],str(args.dev_behaviors))
            self.assertEqual(identity["matched_candidate_profile"],args.matched_candidate_profile)


if __name__=='__main__':
    unittest.main()
