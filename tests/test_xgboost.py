import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xgboost as xgb

from scripts.build_features_and_splits import FEATURE_COLUMNS
from scripts.train_xgboost import Monitor, align_baseline, fit_schema, matrix, postprocess, train_target
from scripts.run_baseline import digest
from scripts.metrics_exporter import render_xgboost


def example(product, qty):
    row = {k: "0" for k in FEATURE_COLUMNS}
    row.update(product_id=product, qty_last=str(qty), target_quantity=str(qty), target_gap_days=str(3 * qty),
               gap_median_days="", gap_mean_days="", gap_last_days="")
    return row


class XGBoostTests(unittest.TestCase):
    def test_live_metrics_hide_stale_losses_and_continuous_exact_match(self):
        state = {"status": "completed", "updated_at": 100, "guardrail_status": "unassessed",
                 "training": {"target": "quantity", "round": 80},
                 "metrics": {"quantity_mae": .7, "quantity_exact_rate": 0},
                 "cart_metrics": {"quantity_exact_rate": .6}}
        result = render_xgboost(state)
        self.assertNotIn('prediction="continuous",metric="quantity_exact_rate"', result)
        self.assertIn('prediction="cart",metric="quantity_exact_rate"', result)
        self.assertNotIn("training_round", result)
        state["status"] = "failed"
        self.assertNotIn("validation_metric", render_xgboost(state))

    def test_category_identity_unknown_missing_and_target_independence(self):
        rows = [example("200", 2), example("100", 1)]
        schema = fit_schema(rows)
        self.assertEqual(schema, fit_schema(list(reversed(rows))))
        self.assertEqual(schema["products"], {"100": 0, "200": 1})
        a = matrix(rows, schema)
        rows[0]["target_quantity"] = "99999"
        b = matrix(rows, schema)
        np.testing.assert_array_equal(a.get_data().toarray(), b.get_data().toarray())
        self.assertEqual(a.feature_types[0], "c")
        unknown = matrix([example("new", 1)], schema).get_data()
        self.assertNotIn(0, unknown.indices)
        self.assertNotIn(FEATURE_COLUMNS.index("gap_median_days"), unknown.indices)

    def test_postprocessing_separates_continuous_and_cart(self):
        q, cart, gap = postprocess([-.2, 1.49, 1.5, 2.8], [-3, 1.49, 1.5, 2.8])
        np.testing.assert_allclose(q, [1, 1.49, 1.5, 2.8])
        np.testing.assert_array_equal(cart, [1, 1, 2, 3])
        np.testing.assert_array_equal(gap, [1, 1, 2, 3])
        with self.assertRaises(ValueError):
            postprocess([float("nan")], [2])

    def test_early_stopping_refit_reload_and_row_reordering(self):
        config = json.loads(Path("config/xgboost-v1.json").read_text())
        config.update(max_rounds=12, early_stopping_rounds=3)
        core = [example(str(i % 3), (i % 3) + 1) for i in range(30)]
        early = [example(str(i % 4), (i % 3) + 1) for i in range(8)]
        pool = core + early
        sc, sp = fit_schema(core), fit_schema(pool)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            monitor = Monitor(out, out / "state/latest.json")
            for target in ("quantity", "gap"):
                model, info = train_target(core, early, pool, sc, sp, target, config, out, monitor)
                scores = monitor.curves[target]["scores"]["early_stop"][info["eval_metric"]]
                self.assertEqual(info["selected_rounds"], int(np.argmin(scores)) + 1)
                self.assertEqual(model.num_boosted_rounds(), info["selected_rounds"])
                matrix1 = matrix(early, sp)
                p = model.predict(matrix1)
                np.testing.assert_array_equal(p, model.predict(matrix(list(reversed(early)), sp))[::-1])
                loaded = xgb.Booster()
                loaded.load_model(out / f"{target}-model.json")
                np.testing.assert_array_equal(p, loaded.predict(matrix1))
                self.assertTrue(np.isfinite(loaded.predict(matrix([example("brand-new", 2)], sp))).all())

    def test_baseline_target_mismatch_is_rejected(self):
        row = {"origin_event_id": "one", "customer_id": "c", "product_id": "p", "origin_on": "2026-07-14",
               "prediction_asof": "2026-07-15", "target_on": "2026-07-20", "target_quantity": "2", "target_gap_days": "6"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root / "validation-predictions.csv"
            with file.open("w", newline="") as stream:
                w = csv.DictWriter(stream, fieldnames=list(row))
                w.writeheader()
                w.writerow(row)
            (root / "report.json").write_text(json.dumps({"files": {file.name: digest(file)}, "input_report_sha256": "same"}))
            align_baseline(root, [row], "same")
            with self.assertRaises(ValueError):
                align_baseline(root, [{**row, "target_quantity": "3"}], "same")
