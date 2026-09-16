import unittest
import numpy as np

from scripts.build_features_and_splits import FEATURE_COLUMNS
from scripts.run_tabpfn_experiment import feature_array, sample_context


class TabPFNExperimentTests(unittest.TestCase):
    def test_sampling_is_order_and_target_independent(self):
        rows = [{"origin_event_id":str(i),"target_gap_days":str(i+1)} for i in range(20)]
        ids = lambda rs: [r["origin_event_id"] for r in rs]
        self.assertEqual(ids(sample_context(rows,5)),ids(sample_context(list(reversed(rows)),5)))
        changed = [{**r,"target_gap_days":"1000"} for r in rows]
        self.assertEqual(ids(sample_context(rows,5)),ids(sample_context(changed,5)))

    def test_only_origin_features_enter_matrix_and_unknown_product_is_missing(self):
        row = {k:"1" for k in FEATURE_COLUMNS}
        row.update(product_id="p",target_gap_days="20",customer_id="private")
        schema = {"products":{"p":0}}
        original = feature_array([row],schema)
        np.testing.assert_array_equal(original,feature_array([{**row,"target_gap_days":"999","customer_id":"other"}],schema))
        self.assertEqual(original.shape,(1,21))
        self.assertTrue(np.isnan(feature_array([{**row,"product_id":"new"}],schema)[0,0]))


if __name__ == "__main__":
    unittest.main()
