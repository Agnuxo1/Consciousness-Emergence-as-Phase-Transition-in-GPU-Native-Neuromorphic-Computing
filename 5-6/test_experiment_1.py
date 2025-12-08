import unittest
import json
import os

from neuro_chimera_experiments_bundle import run_experiment_1_test

class TestExperiment1(unittest.TestCase):
    def test_deterministic_run(self):
        out = run_experiment_1_test()
        # Ensure expected keys exist
        self.assertIn('metrics', out)
        self.assertIn('t', out['metrics'])
        self.assertIn('sync', out['metrics'])
        self.assertIn('entropy', out['metrics'])
        # Ensure tc is present (could be None)
        self.assertIn('tc', out)
        # Check that sync series is not empty
        self.assertTrue(len(out['metrics']['sync']) > 0)
        # Optionally write results for later use
        with open('exp1_test_results.json', 'w') as f:
            json.dump(out, f)

if __name__ == '__main__':
    unittest.main()
