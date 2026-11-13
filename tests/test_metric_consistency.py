"""Independent rank-metric checks for the shared result validator."""
from pathlib import Path
import sys
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import summarize_full_total_matrix as report

class MetricConsistencyTests(unittest.TestCase):
    def test_rank_metrics_known_order_and_ties(self):
            np.testing.assert_allclose(report.independent_metrics(np.array([1,0,0]),np.array([3.,2.,1.])),[1,1,1,1])
            got=report.independent_metrics(np.array([0,1,0]),np.array([1.,1.,1.]))
            np.testing.assert_allclose(got,[.5,.5,1/np.log2(3),1/np.log2(3)])
            got=report.independent_metrics(np.array([1,0,1,0]),np.array([4.,3.,2.,1.]))
            expected_dcg=(1+1/np.log2(4))/(1+1/np.log2(3))
            np.testing.assert_allclose(got,[.75,(1+1/3)/2,expected_dcg,expected_dcg])
