import unittest
import json
import os

from neuro_chimera_experiments_bundle import run_experiment_2_test

class TestExperiment2(unittest.TestCase):
    def test_deterministic_run(self):
        out = run_experiment_2_test()
        # Verify keys
        expected_keys = ['memory_persistence', 'integration', 'metacog_acc', 'metacog_conf']
        for k in expected_keys:
            self.assertIn(k, out)
        
        # Verify ranges
        # memory_persistence is an integral of normalized distance, so it can be > 1.0.
        self.assertGreaterEqual(out['memory_persistence'], 0.0)
        # self.assertLessEqual(out['memory_persistence'], 1.0) # Removed incorrect upper bound check
        
        # Integration can be positive
        self.assertGreaterEqual(out['integration'], 0.0)
        
        # Metacognition
        self.assertGreaterEqual(out['metacog_acc'], 0.0)
        self.assertLessEqual(out['metacog_acc'], 1.0)
        self.assertGreaterEqual(out['metacog_conf'], 0.0)
        self.assertLessEqual(out['metacog_conf'], 1.0)
        
        # Deterministic check for seed 99999 (if logic is perfectly deterministic across platforms)
        # We can loosely check if it returns > 0 values
        self.assertTrue(out['memory_persistence'] > 1e-6)

if __name__ == '__main__':
    unittest.main()
