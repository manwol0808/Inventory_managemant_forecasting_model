"""Compare saved candidates on identical observed outcomes; never open final_test."""
import csv
import json
import math
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xgboost as xgb

from scripts.build_features_and_splits import read_csv
from scripts.run_baseline import digest, metrics, write_json
from scripts.train_xgboost import matrix


RUNS = {
    "short": ("xgboost-short-v2", "features-splits-short-v2"),
    "full": ("xgboost-full-v1", "features-splits-full-v1"),
    "recency180": ("xgboost-full-recency180-v1", "features-splits-full-v1"),
}


def load_verified(run_name, split_name):
    run, split = Path("artifacts") / run_name, Path("data") / split_name
    report = json.loads((run / "report.json").read_text())
    source = read_csv(split / "validation.csv")
    rows = read_csv(run / "validation-predictions.csv")
    if digest(run / "validation-predictions.csv") != report["files"]["validation-predictions.csv"]:
        raise ValueError("Prediction hash changed")
    if digest(split / "validation.csv") != report["inputs_read"]["validation.csv"]:
        raise ValueError("Source hash changed")
    schema = json.loads((run / "feature-schema.json").read_text())
    if digest(run / "feature-schema.json") != report["files"]["feature-schema.json"]:
        raise ValueError("Schema hash changed")
    dm = matrix(source, schema)
    for target, field in (("quantity", "raw_quantity"), ("gap", "raw_gap_days")):
        path = run / (target + "-model.json")
        if digest(path) != report["files"][path.name]:
            raise ValueError("Model hash changed")
        model = xgb.Booster({"nthread": 4})
        model.load_model(path)
        if model.num_boosted_rounds() != report["training"][target]["selected_rounds"]:
            raise ValueError("Model rounds changed")
        np.testing.assert_array_equal(model.predict(dm), np.array([float(r[field]) for r in rows], dtype=np.float32))
    if len(rows) != len(source):
        raise ValueError("Source/prediction row count mismatch")
    for row, src in zip(rows, source):
        for field in ("origin_event_id", "customer_id", "product_id", "origin_on", "target_on", "target_quantity", "target_gap_days"):
            if row[field] != src[field]:
                raise ValueError("Prediction/source truth mismatch")
        qty = max(float(row["raw_quantity"]), 1)
        gap = math.floor(max(float(row["raw_gap_days"]), 1) + 0.5)
        if (qty != float(row["predicted_quantity"]) or math.floor(qty + 0.5) != int(row["cart_quantity"])
                or gap != int(row["predicted_gap_days"])
                or (date.fromisoformat(row["origin_on"]) + timedelta(days=gap)).isoformat() != row["predicted_on"]):
            raise ValueError("Postprocessing mismatch")
    # Independent arithmetic, separate from the production metrics helper.
    for key, qty_key in (("metrics", "predicted_quantity"), ("cart_metrics", "cart_quantity")):
        error = [float(r[qty_key]) - int(r["target_quantity"]) for r in rows]
        check = {"quantity_rmse": math.sqrt(sum(e*e for e in error)/len(error)),
                 "quantity_mae": sum(abs(e) for e in error)/len(error),
                 "date_mae_days": sum(abs(int(r["predicted_gap_days"])-int(r["target_gap_days"])) for r in rows)/len(rows)}
        for field, value in check.items():
            if not math.isclose(value, report[key][field], abs_tol=1e-12):
                raise ValueError("Independent metric mismatch")
    return rows, report


def build():
    loaded = {name: load_verified(*args) for name, args in RUNS.items()}
    ref = {r["origin_event_id"]: r for r in loaded["short"][0]}
    indexed = {name: {r["origin_event_id"]: r for r in rows} for name, (rows, _) in loaded.items()}
    for name, rows in indexed.items():
        if rows.keys() != ref.keys():
            raise ValueError("Comparison populations differ")
        for key, row in rows.items():
            for field in ("customer_id", "product_id", "origin_on", "target_on", "target_quantity", "target_gap_days"):
                if row[field] != ref[key][field]:
                    raise ValueError("Comparison truths differ")
    groups = {}
    # Keep membership fixed to the short-window cohort; full history changes counts.
    for group in ("1", "2", "3-5", "6+"):
        keys = [k for k,r in ref.items() if r["history_group"] == group]
        groups[group] = {name: metrics([rows[k] for k in keys], 7) for name,rows in indexed.items()}
    bands = {}
    for low, high, label in ((1,1,"1"),(2,2,"2"),(3,5,"3-5"),(6,1000000,"6+")):
        keys = [k for k,r in ref.items() if low <= int(r["target_quantity"]) <= high]
        bands[label] = {name: metrics([rows[k] for k in keys], 7) for name,rows in indexed.items()}
    weeks = {}
    for row in ref.values():
        day = date.fromisoformat(row["origin_on"])
        monday = (day-timedelta(days=day.weekday())).isoformat()
        weeks.setdefault(monday, []).append(row["origin_event_id"])
    result = {
        "version": "history-comparison-v1", "validation_rows": len(ref), "truths_identical": True,
        "runs": {name: {"run": RUNS[name][0], "report_sha256": digest(Path("artifacts")/RUNS[name][0]/"report.json"),
                        "metrics": report["metrics"], "cart_metrics": report["cart_metrics"],
                        "training": report["training"], "split_config": report["split_config"]}
                 for name, (_,report) in loaded.items()},
        "fixed_short_history_groups": groups, "actual_quantity_bands": bands,
        "weekly_origins": {week:{name:metrics([rows[k] for k in keys],7) for name,rows in indexed.items()} for week,keys in weeks.items()},
        "diagnostic_only": "Actual-quantity bands use outcomes for error analysis, never routing at prediction time.",
        "verification": {"independent_metric_recomputations": True, "all_saved_model_reload_predictions_equal": True,
                         "verified_prediction_rows": sum(len(rows) for rows,_ in loaded.values()),
                         "final_test_read": False},
        "decision": "Keep short candidate for further validation; reject full and recency180 as replacements on this development split.",
        "operational_approval": False,
    }
    write_json(Path("artifacts/history-comparison-v1.json"),result)
    return result


if __name__ == "__main__":
    result = build()
    print(json.dumps({"rows":result["validation_rows"],"verification":result["verification"],"decision":result["decision"]},ensure_ascii=False,indent=2))
