"""Meaningful CPU-only convergence and exact optimizer/RNG restart checks."""
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from convergence_control import (EarlyStopping,selection_value,capture_rng_state,
    restore_rng_state,atomic_torch_save,prepare_epoch_cache,cleanup_epoch_cache)


class ConvergenceControls(unittest.TestCase):
    def test_weighted_average_matches_existing_nnr_definition(self):
        values={'AUC':0.8,'MRR':0.2,'nDCG@5':0.1,'nDCG@10':0.3}
        self.assertAlmostEqual(selection_value(values),0.4)
        self.assertNotAlmostEqual(selection_value(values),sum(values.values())/4)

    def test_tiny_improvements_update_actual_best_but_not_patience(self):
        control=EarlyStopping(max_epochs=10,min_epochs=2,patience=3,min_delta=0.1)
        results=[control.update(i,value) for i,value in enumerate([0.5,0.55,0.55,0.56],1)]
        self.assertEqual(control.best_epoch,4)
        self.assertEqual(control.best_score,0.56)
        self.assertFalse(results[2]['is_best'])
        self.assertTrue(results[3]['is_best'])
        self.assertFalse(results[3]['significant_improvement'])
        self.assertEqual(results[3]['stop_reason'],'patience')

    def test_minimum_epochs_and_maximum_cap_are_distinct(self):
        control=EarlyStopping(max_epochs=8,min_epochs=5,patience=2)
        for epoch in range(1,5):
            self.assertFalse(control.update(epoch,0.5)['should_stop'])
        self.assertEqual(control.update(5,0.5)['stop_reason'],'patience')
        control=EarlyStopping(max_epochs=4,min_epochs=2,patience=3)
        for epoch in range(1,5):
            result=control.update(epoch,epoch/10)
        self.assertEqual(result['stop_reason'],'max_epochs')

    def test_control_restart_preserves_decisions_and_rejects_replayed_epoch(self):
        original=EarlyStopping(max_epochs=10,min_epochs=3,patience=3)
        original.update(1,0.4);original.update(2,0.6)
        resumed=EarlyStopping.from_state_dict(json.loads(json.dumps(original.state_dict())))
        with self.assertRaises(ValueError):
            resumed.update(2,0.7)
        for epoch,score in [(3,0.59),(4,0.58),(5,0.57)]:
            self.assertEqual(original.update(epoch,score),resumed.update(epoch,score))

    def test_cache_cleanup_is_limited_to_owned_epoch_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            run=Path(temporary)/'new_run';run.mkdir()
            cache=prepare_epoch_cache(run,1)
            (cache/'large.npy').write_bytes(b'cache')
            with self.assertRaises(ValueError):
                cleanup_epoch_cache(run,run)
            outside=Path(temporary)/'preserve';outside.mkdir()
            with self.assertRaises(ValueError):
                cleanup_epoch_cache(run,outside)
            cleanup_epoch_cache(run,cache)
            self.assertFalse(cache.exists())
            self.assertTrue(outside.exists())

    def test_cpu_adam_dropout_and_all_rng_resume_is_bitwise_identical(self):
        import numpy as np
        import torch
        torch.set_num_threads(1)
        def initialize():
            random.seed(19);np.random.seed(19);torch.manual_seed(19)
            model=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(0.25),torch.nn.Linear(4,1))
            return model,torch.optim.Adam(model.parameters(),lr=0.01)
        def epoch(model,optimizer):
            model.train()
            for _ in range(3):
                values=torch.randn(5,3)+float(np.random.normal())*0.01
                target=torch.randn(5,1)+random.random()*0.01
                optimizer.zero_grad()
                ((model(values)-target)**2).mean().backward();optimizer.step()
            # Validation-side RNG consumption must be captured too.
            random.random();np.random.random();torch.rand(2)
        with patch('torch.cuda.is_available',return_value=False):
            model,optimizer=initialize()
            epoch(model,optimizer)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'last.pt'
                atomic_torch_save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                                  'rng':capture_rng_state()},path)
                epoch(model,optimizer)
                expected_model=copy.deepcopy(model.state_dict())
                expected_optimizer=copy.deepcopy(optimizer.state_dict())
                expected_random=(random.random(),float(np.random.random()),torch.rand(4))
                resumed,resumed_optimizer=initialize()
                checkpoint=torch.load(path,map_location='cpu')
                resumed.load_state_dict(checkpoint['model'])
                resumed_optimizer.load_state_dict(checkpoint['optimizer'])
                restore_rng_state(checkpoint['rng'])
                epoch(resumed,resumed_optimizer)
                for name,value in resumed.state_dict().items():
                    self.assertTrue(torch.equal(value,expected_model[name]),name)
                for key,state in resumed_optimizer.state_dict()['state'].items():
                    for name,value in state.items():
                        expected=expected_optimizer['state'][key][name]
                        self.assertTrue(torch.equal(value,expected) if torch.is_tensor(value) else value==expected)
                self.assertEqual(random.random(),expected_random[0])
                self.assertEqual(float(np.random.random()),expected_random[1])
                self.assertTrue(torch.equal(torch.rand(4),expected_random[2]))
                self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':
    unittest.main()
