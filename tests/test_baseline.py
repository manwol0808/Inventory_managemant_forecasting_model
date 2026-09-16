import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.run_baseline import INPUT_FEATURES, RunLog, digest, fit, metrics, predict, run
from scripts.build_features_and_splits import next_midnight
from scripts.metrics_exporter import render

CONFIG = json.loads(Path("config/baseline-v1.json").read_text())


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.train = [{"outer_split": "train", "target_on": "2026-07-01", "target_gap_days": str(g), "product_id": p}
                      for p, g in [("p", 2), ("p", 4), ("p", 6), ("p", 8), ("p", 10), ("sparse", 30)]]
        self.model = fit(self.train, CONFIG, "2026-07-13")

    def test_gap_sources_rounding_and_serialization(self):
        feature = {"product_id": "p", "qty_last": "3", "gap_count": "0", "gap_median_days": ""}
        self.assertEqual(predict(self.model, feature), {"predicted_quantity": 3, "predicted_gap_days": 6, "gap_source": "product"})
        for product in ("sparse", "unknown"):
            feature["product_id"] = product
            self.assertEqual(predict(self.model, feature)["predicted_gap_days"], 7)
            self.assertEqual(predict(self.model, feature)["gap_source"], "global")
        feature.update(gap_count="2", gap_median_days="2.5")
        self.assertEqual(predict(self.model, feature)["predicted_gap_days"], 3)
        self.assertEqual(predict(self.model, feature), predict(json.loads(json.dumps(self.model)), feature))

    def test_no_target_in_prediction_and_future_fit_rejected(self):
        row = {"product_id": "p", "qty_last": "3", "gap_count": "1", "gap_median_days": "9", "target_quantity": "1"}
        before = predict(self.model, {k: row[k] for k in INPUT_FEATURES})
        row["target_quantity"] = "999999"
        self.assertEqual(before, predict(self.model, {k: row[k] for k in INPUT_FEATURES}))
        with self.assertRaises(ValueError):
            predict(self.model, row)
        for field, value in (("outer_split", "validation"), ("target_on", "2026-07-14")):
            bad = copy.deepcopy(self.train)
            bad[0][field] = value
            with self.assertRaises(ValueError):
                fit(bad, CONFIG, "2026-07-13")

    def test_metrics_against_hand_example(self):
        rows = [{"predicted_quantity": 2, "target_quantity": 4, "predicted_gap_days": 10, "target_gap_days": 20},
                {"predicted_quantity": 5, "target_quantity": 4, "predicted_gap_days": 12, "target_gap_days": 10}]
        actual = metrics(rows, 7)
        self.assertEqual(actual["quantity_mae"], 1.5)
        self.assertEqual(actual["quantity_rmse"], math.sqrt(2.5))
        self.assertEqual(actual["quantity_wape"], 3 / 8)
        self.assertEqual(actual["quantity_bias_mean"], -0.5)
        self.assertEqual(actual["date_mae_days"], 6)
        self.assertEqual(actual["date_within_tolerance_rate"], .5)
        self.assertNotIn("quantity_mae", metrics([], 7))

    def test_slot_reset_failure_and_no_pii_or_final_test_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder in ("one", "two"):
                (root / folder).mkdir()
            first = RunLog(root / "one", root / "state", CONFIG)
            first.event("run_finished", status="completed", metrics={"quantity_mae": 2}, last_success_timestamp_seconds=123)
            self.assertIn('metric="quantity_mae"', render(first.state))
            second = RunLog(root / "two", root / "state", CONFIG)
            self.assertNotIn('metric="quantity_mae"', render(second.state))
            self.assertEqual(second.state["last_success_timestamp_seconds"], 123)
            second.event("run_failed", status="failed")
            output = render(second.state)
            self.assertIn("forecasting_run_failed 1.0", output)
            for forbidden in ("customer", "product_id", "run_id", "final_test", "guardrail_pass"):
                self.assertNotIn(forbidden, output)
            self.assertEqual(render(None), "forecasting_state_present 0.0\n")

    def test_hash_failure_logged_and_existing_run_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                run(root / "missing", "config/baseline-v1.json", root / "run", root / "state")
            state = json.loads((root / "state/latest.json").read_text())
            self.assertEqual(state["status"], "failed")
            with self.assertRaises(FileExistsError):
                run(root / "missing", "config/baseline-v1.json", root / "run", root / "state")

    def test_end_to_end_without_test_file_and_mutated_validation_labels(self):
        def row(identity, origin, target, split, quantity):
            from datetime import date
            return {"origin_event_id": identity, "customer_id": identity, "product_id": "p",
                    "origin_on": origin, "prediction_asof": next_midnight(origin).isoformat(),
                    "target_on": target, "target_gap_days": (date.fromisoformat(target) - date.fromisoformat(origin)).days,
                    "target_quantity": quantity, "outer_split": split, "label_state": "observed",
                    "qty_last": 3, "episode_purchase_count": 1, "gap_count": 0, "gap_median_days": ""}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "input"
            source.mkdir()
            train = [row("a", "2026-07-01", "2026-07-10", "train", 2)]
            valid = [row("b", "2026-07-14", "2026-07-24", "validation", 4)]
            manifest = {"version": "features-splits-v1", "mode": "retrospective_exploration",
                        "split_config": json.loads(Path("config/time-split-v1.json").read_text()),
                        "feature_columns": list(INPUT_FEATURES), "exploratory_file_rows": {"train_pool": 1, "validation": 1},
                        "origins_by_outer_split": {"validation": 2}, "role_outcomes": {"validation": {"observed": 1, "right_censored": 1}}, "files": {}}
            outputs = []
            for iteration in range(2):
                valid[0]["target_quantity"] = 4 if iteration == 0 else 100
                for filename, rows in (("train_pool.csv", train), ("validation.csv", valid)):
                    with (source / filename).open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                        writer.writeheader()
                        writer.writerows(rows)
                    manifest["files"][filename] = digest(source / filename)
                (source / "report.json").write_text(json.dumps(manifest))
                result = run(source, "config/baseline-v1.json", root / str(iteration), root / "state")
                with (root / str(iteration) / "validation-predictions.csv").open() as stream:
                    outputs.append(next(csv.DictReader(stream)))
                self.assertFalse(result["final_test_metrics_computed"])
                self.assertEqual(result["coverage"]["evaluated_fraction"], .5)
            for key in ("predicted_on", "predicted_quantity", "predicted_gap_days", "gap_source"):
                self.assertEqual(outputs[0][key], outputs[1][key])
            self.assertEqual(outputs[0]["predicted_on"], "2026-07-23")
            self.assertEqual((root / "0/model.json").read_bytes(), (root / "1/model.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
