"""One frozen XGBoost experiment. Tune inside train, refit, then score validation."""
import argparse
import csv
import json
import math
import os
import platform
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb

from scripts.build_features_and_splits import FEATURE_COLUMNS, read_csv, validate_config
from scripts.run_baseline import digest, history_group, metrics, validate_rows, write_json


def fit_schema(rows):
    return {"features": FEATURE_COLUMNS, "types": ["c"] + ["q"] * (len(FEATURE_COLUMNS) - 1),
            "products": {p: i for i, p in enumerate(sorted({r["product_id"] for r in rows if r["product_id"]}))},
            "unknown_product": "missing"}


def matrix(rows, schema, label=None):
    if schema["features"] != FEATURE_COLUMNS or schema["types"] != ["c"] + ["q"] * 20:
        raise ValueError("Unexpected feature schema")
    data = []
    for row in rows:
        values = []
        for field in schema["features"]:
            value = row[field]
            if field == "product_id":
                values.append(schema["products"].get(value, np.nan))
            elif value == "":
                values.append(np.nan)
            else:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError("Nonfinite feature")
                values.append(value)
        data.append(values)
    labels = np.asarray([float(r[label]) for r in rows], dtype=np.float32) if label else None
    return xgb.DMatrix(np.asarray(data, dtype=np.float32), label=labels, feature_names=schema["features"],
                       feature_types=schema["types"], enable_categorical=True, nthread=4)


def postprocess(qty, gap):
    qty, gap = np.asarray(qty, dtype=float), np.asarray(gap, dtype=float)
    if not np.isfinite(qty).all() or not np.isfinite(gap).all():
        raise ValueError("Nonfinite prediction")
    continuous = np.maximum(qty, 1)
    return continuous, np.floor(continuous + 0.5).astype(int), np.floor(np.maximum(gap, 1) + 0.5).astype(int)


def training_weights(rows, cutoff, half_life_days):
    if not math.isfinite(half_life_days) or half_life_days <= 0:
        raise ValueError("Positive finite weight half-life required")
    ages = [(date.fromisoformat(cutoff) - date.fromisoformat(r["target_on"])).days for r in rows]
    if any(age < 0 for age in ages):
        raise ValueError("Weight target is after training cutoff")
    return np.asarray([2 ** (-age / half_life_days) for age in ages], dtype=np.float32)


class Monitor:
    def __init__(self, output, state_path):
        self.output, self.state_path = output, state_path
        self.start = time.monotonic()
        self.curves = {}
        self.state = {"run_id": output.name, "model_family": "xgboost", "status": "running", "stage": "loading",
                      "metrics": {}, "cart_metrics": {}, "baseline_metrics": {}, "training": {},
                      "guardrail_status": "unassessed", "final_test_metrics_computed": False}
        self.event("run_started")

    def event(self, event, **kwargs):
        self.state.update(kwargs)
        self.state.update(updated_at=time.time(), duration_seconds=time.monotonic() - self.start)
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "event_type": event,
                                    **self.state}, ensure_ascii=False, allow_nan=False) + "\n")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        write_json(tmp, self.state)
        os.replace(tmp, self.state_path)


class Progress(xgb.callback.TrainingCallback):
    def __init__(self, monitor, target, phase):
        self.monitor, self.target, self.phase = monitor, target, phase

    def after_iteration(self, model, epoch, evals_log):
        if epoch % 10 == 0:
            losses = {split: {name: float(values[-1]) for name, values in scores.items()}
                      for split, scores in evals_log.items()}
            self.monitor.event("training_metric", stage=self.phase, training={"target": self.target, "round": epoch + 1, "losses": losses})
        return False


def train_target(core, early, pool, schema_core, schema_pool, target, config, output, monitor):
    spec = config["targets"][target]
    params = {**config["params"], "objective": spec["objective"], "eval_metric": spec["eval_metric"]}
    core_dm, early_dm = matrix(core, schema_core, spec["label"]), matrix(early, schema_core, spec["label"])
    half_life = config.get("sample_weight_half_life_days")
    if half_life is not None:
        core_dm.set_weight(training_weights(core, core[0]["role_cutoff"], half_life))
    curve = {}
    probe = xgb.train(params, core_dm, config["max_rounds"], evals=[(core_dm, "train_core"), (early_dm, "early_stop")],
                      evals_result=curve, early_stopping_rounds=config["early_stopping_rounds"], verbose_eval=False,
                      callbacks=[Progress(monitor, target, "early_stopping")])
    rounds = probe.best_iteration + 1
    best = probe[:rounds]
    best.save_model(output / f"{target}-early-best.json")
    np.testing.assert_array_equal(best.predict(early_dm), probe.predict(early_dm, iteration_range=(0, rounds)))
    monitor.curves[target] = {"metric": spec["eval_metric"], "selected_rounds": rounds, "scores": curve}
    write_json(output / "learning-curves.json", monitor.curves)
    monitor.event("iterations_selected", stage="refit", target=target, selected_rounds=rounds)
    pool_dm = matrix(pool, schema_pool, spec["label"])
    if half_life is not None:
        pool_cutoff = max(row["role_cutoff"] for row in pool)
        pool_dm.set_weight(training_weights(pool, pool_cutoff, half_life))
    model = xgb.train(params, pool_dm, rounds,
                      callbacks=[Progress(monitor, target, "refit")], verbose_eval=False)
    if model.num_boosted_rounds() != rounds:
        raise AssertionError("Refit iteration mismatch")
    model.save_model(output / f"{target}-model.json")
    loaded = xgb.Booster({"nthread": 4})
    loaded.load_model(output / f"{target}-model.json")
    probe_dm = matrix(pool[:128], schema_pool)
    np.testing.assert_array_equal(loaded.predict(probe_dm), model.predict(probe_dm))
    return loaded, {"selected_rounds": rounds, "probe_rounds": probe.num_boosted_rounds(),
                    "best_inner_score": float(probe.best_score), "objective": spec["objective"], "eval_metric": spec["eval_metric"]}


def load_inputs(input_dir):
    manifest = json.loads((input_dir / "report.json").read_text())
    validate_config(manifest["split_config"])
    if manifest["version"] != "features-splits-v1" or manifest["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Unexpected input contract")
    data = {}
    for name in ("train_core", "early_stop", "train_pool", "validation"):
        path = input_dir / f"{name}.csv"
        if digest(path) != manifest["files"][path.name]:
            raise ValueError("Input hash mismatch")
        rows = read_csv(path)
        if len(rows) != manifest["exploratory_file_rows"][name] or not rows:
            raise ValueError("Unexpected input count")
        if name in ("train_pool", "validation"):
            validate_rows(rows, "train" if name == "train_pool" else "validation", manifest)
        else:
            cfg = manifest["split_config"]
            cutoff = cfg["train_core_end"] if name == "train_core" else cfg["train_end"]
            for r in rows:
                if r["fit_role"] != name or r["outer_split"] != "train" or r["label_state"] != "observed" or not cfg["start"] <= r["origin_on"] < r["target_on"] <= cutoff:
                    raise ValueError("Inner split/label boundary violation")
                if name == "early_stop" and r["origin_on"] <= cfg["train_core_end"]:
                    raise ValueError("Inner split origin violation")
        data[name] = rows
    pool = {r["origin_event_id"]: r for r in data["train_pool"]}
    for name in ("train_core", "early_stop"):
        if len({r["origin_event_id"] for r in data[name]}) != len(data[name]):
            raise ValueError("Duplicate inner origin")
        for r in data[name]:
            if pool.get(r["origin_event_id"]) != r:
                raise ValueError("Inner row absent/different in pool")
    if set(pool) & {r["origin_event_id"] for r in data["validation"]}:
        raise ValueError("Train and validation overlap")
    return manifest, data


def align_baseline(baseline_dir, rows, input_report_sha256):
    report = json.loads((baseline_dir / "report.json").read_text())
    path = baseline_dir / "validation-predictions.csv"
    if digest(path) != report["files"][path.name] or report["input_report_sha256"] != input_report_sha256:
        raise ValueError("Baseline provenance mismatch")
    predictions = read_csv(path)
    indexed = {r["origin_event_id"]: r for r in predictions}
    if len(indexed) != len(predictions) or set(indexed) != {r["origin_event_id"] for r in rows}:
        raise ValueError("Baseline evaluation population mismatch")
    for row in rows:
        base = indexed[row["origin_event_id"]]
        for key in ("customer_id", "product_id", "origin_on", "prediction_asof", "target_on", "target_quantity", "target_gap_days"):
            if row[key] != base[key]:
                raise ValueError("Baseline truth mismatch")
    return report, indexed


def comparison(candidate, baseline):
    result = {}
    if not candidate["n"]:
        return result
    for metric in ("quantity_rmse", "quantity_mae", "date_mae_days"):
        b, c = baseline[metric], candidate[metric]
        result[metric] = {"baseline": b, "xgboost": c, "delta": c - b,
                          "improvement_fraction": (b - c) / b if b else None}
    return result


def run(input_dir, baseline_dir, config_path, output, state_path):
    input_dir, baseline_dir, config_path, output, state_path = map(Path, (input_dir, baseline_dir, config_path, output, state_path))
    config = json.loads(config_path.read_text())
    if config["version"] != "xgboost-v1" or config["mode"] != "retrospective_exploration" or not config["final_test_locked"]:
        raise ValueError("Unsupported experiment")
    if xgb.__version__ != config["xgboost_version"] or np.__version__ != config["numpy_version"]:
        raise ValueError("Dependency versions differ from frozen experiment")
    output.mkdir(parents=True, exist_ok=False)
    monitor = Monitor(output, state_path)
    try:
        manifest, data = load_inputs(input_dir)
        baseline, indexed = align_baseline(baseline_dir, data["validation"], digest(input_dir / "report.json"))
        core_schema, pool_schema = fit_schema(data["train_core"]), fit_schema(data["train_pool"])
        write_json(output / "core-schema.json", core_schema)
        write_json(output / "feature-schema.json", pool_schema)
        write_json(output / "config.json", config)
        monitor.event("data_validated", rows={k: len(v) for k, v in data.items()})
        models, training = {}, {}
        for target in ("quantity", "gap"):
            models[target], training[target] = train_target(data["train_core"], data["early_stop"], data["train_pool"],
                core_schema, pool_schema, target, config, output, monitor)
        # Validation features are constructed only after both models/iteration counts are frozen.
        valid_dm = matrix(data["validation"], pool_schema)
        raw_qty, raw_gap = models["quantity"].predict(valid_dm), models["gap"].predict(valid_dm)
        qty, cart, gap = postprocess(raw_qty, raw_gap)
        predictions, carts = [], []
        for i, row in enumerate(data["validation"]):
            p = {k: row[k] for k in ("origin_event_id", "customer_id", "product_id", "origin_on", "prediction_asof", "target_on", "target_quantity", "target_gap_days")}
            p.update(raw_quantity=float(raw_qty[i]), raw_gap_days=float(raw_gap[i]), predicted_quantity=float(qty[i]),
                     cart_quantity=int(cart[i]), predicted_gap_days=int(gap[i]),
                     predicted_on=(date.fromisoformat(row["origin_on"]) + timedelta(days=int(gap[i]))).isoformat(),
                     history_group=history_group(row["episode_purchase_count"]),
                     product_group="seen" if row["product_id"] in pool_schema["products"] else "unseen")
            predictions.append(p)
            carts.append({**p, "predicted_quantity": int(cart[i])})
        with (output / "validation-predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        overall, cart_metrics = metrics(predictions, 7), metrics(carts, 7)
        groups = {}
        for dimension, names in (("history_group", ("1", "2", "3-5", "6+")), ("product_group", ("seen", "unseen"))):
            groups[dimension] = {}
            for name in names:
                selected = [p for p in predictions if p[dimension] == name]
                bm = metrics([indexed[p["origin_event_id"]] for p in selected], 7)
                cm = metrics(selected, 7)
                groups[dimension][name] = {"metrics": cm, "baseline_metrics": bm, "comparison": comparison(cm, bm)}
        input_files = ("report.json", "train_core.csv", "early_stop.csv", "train_pool.csv", "validation.csv")
        report = {"version": "xgboost-result-v1", "mode": config["mode"], "config": config,
                  "config_sha256": digest(config_path), "code_sha256": digest(__file__),
                  "environment": {"python": platform.python_version(), "xgboost": xgb.__version__, "numpy": np.__version__},
                  "split_config": manifest["split_config"], "inputs_read": {f: digest(input_dir / f) for f in input_files},
                  "baseline_report_sha256": digest(baseline_dir / "report.json"), "training": training,
                  "metrics": overall, "cart_metrics": cart_metrics, "baseline_metrics": baseline["metrics"],
                  "comparison": comparison(overall, baseline["metrics"]), "cart_comparison": comparison(cart_metrics, baseline["metrics"]),
                  "groups": groups, "coverage": baseline["coverage"], "final_test_metrics_computed": False,
                  "strict_training_ready": False, "guardrail_status": "unassessed", "operational_approval": False,
                  "selection_status": "exploratory_candidate_only", "limitations": baseline["limitations"],
                  "files": {p.name: digest(p) for p in output.iterdir() if p.name != "events.jsonl"}}
        write_json(output / "report.json", report)
        monitor.event("run_finished", stage="finished", status="completed", metrics=overall, cart_metrics=cart_metrics,
                      baseline_metrics=baseline["metrics"], comparison=report["comparison"], selected_iterations=training,
                      coverage=baseline["coverage"], last_success_timestamp_seconds=time.time())
        return report
    except Exception as error:
        monitor.event("run_failed", status="failed", stage="failed", metrics={}, cart_metrics={}, reason=type(error).__name__)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config/xgboost-v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("artifacts/monitoring/xgboost-latest.json"))
    args = parser.parse_args()
    report = run(args.input_dir, args.baseline, args.config, args.output, args.state)
    print(json.dumps({"training": report["training"], "comparison": report["comparison"], "cart_metrics": report["cart_metrics"]}, indent=2))
