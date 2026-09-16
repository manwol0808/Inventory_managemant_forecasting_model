import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.build_features_and_splits import FEATURE_COLUMNS, next_midnight, read_csv
from scripts.build_purchase_events import write_csv
from scripts.evaluate_final_test import evaluate
from scripts.run_baseline import digest, write_json


class FakeBooster:
    def __init__(self, *args):
        pass

    def load_model(self, path):
        self.quantity = path.name.startswith("quantity")

    def predict(self, dm):
        return np.array([1.4 if self.quantity else 8.2], dtype=np.float32)


class FinalTestTests(unittest.TestCase):
    def fixture(self, root, target="2026-08-30"):
        split, run, base = [root/name for name in ("split","candidate","baseline")]
        for path in (split,run,base):
            path.mkdir()
        row = {k:"1" for k in FEATURE_COLUMNS}
        row.update(product_id="p",origin_event_id="e",customer_id="c",origin_on="2026-08-20",target_on=target,
                   prediction_asof=next_midnight("2026-08-20").isoformat(),target_quantity="1",target_gap_days="10",
                   fit_role="final_test",outer_split="test",label_state="observed",gap_count="0",gap_median_days="")
        write_csv(split/"final_test.csv",list(row),[row])
        manifest={"split_config":{"train_end":"2026-07-13","validation_end":"2026-08-14","test_end":"2026-09-15"},
                  "files":{"final_test.csv":digest(split/"final_test.csv")},"exploratory_file_rows":{"final_test":1},
                  "origins_by_outer_split":{"test":2},"role_outcomes":{"final_test":{"observed":1,"right_censored":1}}}
        write_json(split/"report.json",manifest)
        write_json(base/"model.json",{"products":{},"product_min_intervals":5,"global_gap_median":12})
        write_json(base/"report.json",{"input_report_sha256":digest(split/"report.json"),"files":{"model.json":digest(base/"model.json")}})
        write_json(run/"feature-schema.json",{})
        for target in ("quantity","gap"):
            write_json(run/(target+"-model.json"),{})
        write_json(run/"report.json",{"baseline_report_sha256":digest(base/"report.json"),
                   "files":{p.name:digest(p) for p in run.iterdir()},"limitations":["test fixture"]})
        return split,run,base,root/"result"

    def test_choice_frozen_before_holdout_read_and_labels_not_predicted(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.fixture(Path(temp))
            def read_checked(path):
                self.assertTrue((args[-1]/"selection-frozen.json").exists())
                return read_csv(path)
            def features_checked(rows,schema):
                self.assertEqual(set(rows[0]),set(FEATURE_COLUMNS))
                self.assertNotIn("target_quantity",rows[0])
                return rows
            with patch("scripts.evaluate_final_test.xgb.Booster",FakeBooster), patch("scripts.evaluate_final_test.read_csv",read_checked), patch("scripts.evaluate_final_test.matrix",features_checked):
                result=evaluate(*args)
            self.assertEqual(result["cart_metrics"]["quantity_mae"],0)
            self.assertEqual(result["metrics"]["date_mae_days"],2)
            self.assertTrue(result["final_test_consumed"])
            self.assertFalse(result["operational_approval"])
            with self.assertRaises(FileExistsError):
                evaluate(*args)

    def test_future_test_target_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.fixture(Path(temp),"2026-09-16")
            with patch("scripts.evaluate_final_test.xgb.Booster",FakeBooster):
                with self.assertRaisesRegex(ValueError,"contract violation"):
                    evaluate(*args)
