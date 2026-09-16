"""Fit a frozen fallback baseline and score validation only, never final_test."""
import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median

from scripts.build_features_and_splits import next_midnight, read_csv, validate_config

INPUT_FEATURES = ("product_id", "qty_last", "gap_count", "gap_median_days")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError("Expected a positive finite number")
    return number


def positive_integer(value):
    number = positive(value)
    if not number.is_integer():
        raise ValueError("Expected positive catalog-unit integer")
    return int(number)


def fit(rows, config, train_end):
    """Only train_pool labels may supply population gap medians."""
    gaps, product_gaps = [], defaultdict(list)
    for row in rows:
        if row["outer_split"] != "train" or row["target_on"] > train_end:
            raise ValueError("Fallback fit received non-training/future target")
        gap = positive_integer(row["target_gap_days"])
        gaps.append(gap)
        product_gaps[row["product_id"]].append(gap)
    if not gaps:
        raise ValueError("Cannot fit empty training data")
    return {"version": config["version"], "train_end": train_end,
            "quantity_rule": config["quantity_rule"], "gap_rule": config["gap_rule"],
            "gap_rounding": config["gap_rounding"],
            "product_min_intervals": config["product_min_intervals"],
            "global_gap_median": median(gaps), "global_interval_count": len(gaps),
            "products": {p: {"gap_median": median(gs), "interval_count": len(gs)}
                         for p, gs in sorted(product_gaps.items())}}


def predict(model, features):
    """No dates, identities, split fields or targets enter the predictor."""
    if set(features) != set(INPUT_FEATURES):
        raise ValueError("Predictor accepts only its feature allowlist")
    qty = positive_integer(features["qty_last"])
    count = int(features["gap_count"])
    if count < 0:
        raise ValueError("Negative gap count")
    if count:
        gap, source = positive(features["gap_median_days"]), "pair"
    else:
        if features["gap_median_days"] not in ("", None):
            raise ValueError("Gap median present without observed intervals")
        product = model["products"].get(features["product_id"])
        if product and product["interval_count"] >= model["product_min_intervals"]:
            gap, source = product["gap_median"], "product"
        else:
            gap, source = model["global_gap_median"], "global"
    return {"predicted_quantity": qty, "predicted_gap_days": max(1, math.floor(gap + 0.5)),
            "gap_source": source}


def history_group(count):
    count = positive_integer(count)
    return str(count) if count <= 2 else "3-5" if count <= 5 else "6+"


def metrics(rows, tolerance):
    if not rows:
        return {"n": 0, "status": "no_observations"}
    errors = [float(r["predicted_quantity"]) - float(r["target_quantity"]) for r in rows]
    days = [int(r["predicted_gap_days"]) - int(r["target_gap_days"]) for r in rows]
    total = sum(float(r["target_quantity"]) for r in rows)
    absolute = sum(abs(e) for e in errors)
    n = len(rows)
    return {"n": n, "status": "exploratory", "quantity_rmse": math.sqrt(sum(e * e for e in errors) / n),
            "quantity_mae": absolute / n, "quantity_wape": absolute / total if total else None,
            "actual_quantity_sum": total, "predicted_quantity_sum": total + sum(errors),
            "quantity_bias_mean": sum(errors) / n, "quantity_exact_rate": sum(e == 0 for e in errors) / n,
            "quantity_under_rate": sum(e < 0 for e in errors) / n,
            "quantity_over_rate": sum(e > 0 for e in errors) / n,
            "date_mae_days": sum(abs(d) for d in days) / n,
            "date_bias_days": sum(days) / n,
            "date_within_tolerance_rate": sum(abs(d) <= tolerance for d in days) / n}


class RunLog:
    """One local run slot; persistent JSONL plus atomic latest state for the exporter."""
    def __init__(self, output, state_dir, config):
        self.output, self.state_dir = output, state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        previous = {}
        if (state_dir / "latest.json").exists():
            previous = json.loads((state_dir / "latest.json").read_text())
        self.state = {"run_id": output.name, "model_family": "baseline", "mode": config["mode"],
                      "evaluation_version": config["version"], "status": "running", "stage": "start",
                      "started_at": time.time(), "metrics": {}, "groups": {}, "rows": {},
                      "guardrail_status": "unassessed", "final_test_metrics_computed": False,
                      "last_success_timestamp_seconds": previous.get("last_success_timestamp_seconds")}
        self.event("run_started")

    def event(self, kind, **values):
        self.state.update(values)
        self.state["updated_at"] = time.time()
        self.state["duration_seconds"] = time.monotonic() - self.started
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                                    "event_type": kind, **self.state}, ensure_ascii=False, allow_nan=False) + "\n")
        temporary = self.state_dir / "latest.tmp"
        write_json(temporary, self.state)
        os.replace(temporary, self.state_dir / "latest.json")


def validate_rows(rows, split, manifest):
    config = manifest["split_config"]
    seen = set()
    for row in rows:
        origin, target = date.fromisoformat(row["origin_on"]), date.fromisoformat(row["target_on"])
        lower = date.fromisoformat(config["start"]) if split == "train" else date.fromisoformat(config["train_end"]) + timedelta(days=1)
        upper = date.fromisoformat(config["train_end"] if split == "train" else config["validation_end"])
        if not lower <= origin < target <= upper or row["outer_split"] != split or row["label_state"] != "observed":
            raise ValueError("Origin/target violates frozen split")
        if row["prediction_asof"] != next_midnight(row["origin_on"]).isoformat():
            raise ValueError("Wrong prediction timestamp")
        if positive_integer(row["target_gap_days"]) != (target - origin).days:
            raise ValueError("Target gap/date mismatch")
        positive_integer(row["target_quantity"])
        positive_integer(row["qty_last"])
        if row["origin_event_id"] in seen or not row["product_id"] or not row["customer_id"]:
            raise ValueError("Duplicate/missing identity")
        seen.add(row["origin_event_id"])
        if int(row["gap_count"]) != positive_integer(row["episode_purchase_count"]) - 1:
            raise ValueError("Episode/gap count mismatch")
    filename = "train_pool" if split == "train" else "validation"
    if len(rows) != manifest["exploratory_file_rows"][filename] or not rows:
        raise ValueError("Unexpected/empty row count")


def run(input_dir, config_path, output_dir, state_dir):
    input_dir, config_path, output_dir, state_dir = map(Path, (input_dir, config_path, output_dir, state_dir))
    config = json.loads(config_path.read_text())
    expected = {"version": "baseline-v1", "mode": "retrospective_exploration", "quantity_rule": "last_quantity",
                "gap_rule": "pair_median_then_frozen_product_then_global", "product_min_intervals": 5,
                "gap_rounding": "half_up_minimum_one", "primary_metric": "quantity_rmse",
                "descriptive_date_tolerance_days": 7, "history_groups": ["1", "2", "3-5", "6+"],
                "guardrail_status": "unassessed", "final_test_locked": True}
    if config != expected:
        raise ValueError("Changed evaluation policy requires a new implementation/version")
    output_dir.mkdir(parents=True, exist_ok=False)
    log = RunLog(output_dir, state_dir, config)
    try:
        manifest = json.loads((input_dir / "report.json").read_text())
        validate_config(manifest["split_config"])
        if manifest["version"] != "features-splits-v1" or manifest["mode"] != config["mode"]:
            raise ValueError("Unsupported feature manifest")
        for filename in ("train_pool.csv", "validation.csv"):
            if digest(input_dir / filename) != manifest["files"][filename]:
                raise ValueError("Input hash mismatch")
        # Intentionally never open final_test.csv or all-origins-audit.csv.
        train, valid = read_csv(input_dir / "train_pool.csv"), read_csv(input_dir / "validation.csv")
        validate_rows(train, "train", manifest)
        validate_rows(valid, "validation", manifest)
        if not set(INPUT_FEATURES) <= set(manifest["feature_columns"]):
            raise ValueError("Baseline inputs outside feature allowlist")
        if {r["origin_event_id"] for r in train} & {r["origin_event_id"] for r in valid}:
            raise ValueError("Training/validation identities overlap")
        log.event("data_validated", stage="fit", rows={"train": len(train), "validation": len(valid)})
        model = fit(train, config, manifest["split_config"]["train_end"])
        write_json(output_dir / "model.json", model)
        # Predict using the serialized model that will be handed off.
        model = json.loads((output_dir / "model.json").read_text())
        log.event("stage_finished", stage="validation")
        predictions = []
        for row in valid:
            prediction = predict(model, {k: row[k] for k in INPUT_FEATURES})
            predictions.append({k: row[k] for k in ("origin_event_id", "customer_id", "product_id", "origin_on", "prediction_asof", "target_on", "target_quantity", "target_gap_days")})
            predictions[-1].update(prediction)
            predictions[-1]["predicted_on"] = (date.fromisoformat(row["origin_on"]) + timedelta(days=prediction["predicted_gap_days"])).isoformat()
            predictions[-1]["history_group"] = history_group(row["episode_purchase_count"])
        with (output_dir / "validation-predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        tolerance = config["descriptive_date_tolerance_days"]
        overall = metrics(predictions, tolerance)
        groups = {name: metrics([r for r in predictions if r["history_group"] == name], tolerance) for name in config["history_groups"]}
        sources = {name: metrics([r for r in predictions if r["gap_source"] == name], tolerance) for name in ("pair", "product", "global")}
        coverage = {"validation_origins": manifest["origins_by_outer_split"]["validation"],
                    "outcomes": manifest["role_outcomes"]["validation"],
                    "evaluated_rows": len(valid),
                    "evaluated_fraction": len(valid) / manifest["origins_by_outer_split"]["validation"]}
        report = {"version": "baseline-result-v1", "config": config, "split_config": manifest["split_config"],
                  "input_report_sha256": digest(input_dir / "report.json"), "config_sha256": digest(config_path),
                  "code_sha256": digest(__file__),
                  "inputs_read": {f: digest(input_dir / f) for f in ("report.json", "train_pool.csv", "validation.csv")},
                  "fit_rows": len(train), "metrics": overall, "history_groups": groups, "gap_sources": sources,
                  "coverage": coverage, "strict_training_ready": False, "final_test_metrics_computed": False,
                  "guardrail_status": "unassessed", "operational_approval": False,
                  "limitations": ["Latest-state snapshot: historical availability is unverified.",
                                  "Only returns observed by validation cutoff are scored; short-gap selection bias.",
                                  "Post-purchase forecasts only, no no-purchase probability or daily rescheduling.",
                                  "Timing tolerance is descriptive, not a business approval threshold."],
                  "files": {f: digest(output_dir / f) for f in ("model.json", "validation-predictions.csv")}}
        write_json(output_dir / "report.json", report)
        log.event("fold_evaluated", metrics=overall, groups=groups, coverage=coverage)
        log.event("run_finished", status="completed", stage="finished", last_success_timestamp_seconds=time.time())
        return report
    except Exception as error:
        log.event("run_failed", status="failed", stage="failed", metrics={}, groups={}, reason=type(error).__name__)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/baseline-v1.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, default=Path("artifacts/monitoring"))
    args = parser.parse_args()
    result = run(args.input_dir, args.config, args.output, args.state_dir)
    print(json.dumps({"output": str(args.output), "metrics": result["metrics"], "coverage": result["coverage"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
