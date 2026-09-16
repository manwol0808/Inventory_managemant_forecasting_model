"""Past-only features and purged calendar splits; snapshot exploration, not operational validation."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from statistics import mean, median, pstdev
from zoneinfo import ZoneInfo

from scripts.build_purchase_events import digest, timestamp, write_csv

FEATURE_COLUMNS = [
    "product_id", "origin_weekday", "origin_month", "episode_purchase_count",
    "episode_age_days", "qty_last", "qty_mean", "qty_mean_last3", "qty_median_last3",
    "qty_std_last3", "qty_sum_7d", "qty_sum_28d", "gap_count", "gap_last_days",
    "gap_mean_days", "gap_median_days", "customer_purchase_days", "customer_product_count",
    "product_purchase_events", "product_customer_count", "product_mean_quantity",
]
NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c != "product_id"]


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def next_midnight(day):
    return datetime.combine(date.fromisoformat(day) + timedelta(days=1), time(), ZoneInfo("Asia/Seoul"))


def validate_config(config):
    if (config.get("version") != "time-split-v1" or config.get("mode") != "retrospective_exploration" or
            config.get("timezone") != "Asia/Seoul" or config.get("split_anchor") != "origin_purchase_day" or
            config.get("prediction_asof") != "next_local_midnight_after_origin_purchase_day" or
            config.get("historical_availability_verified") is not False or
            config.get("last_day_completeness_verified") is not False or config.get("final_test_locked") is not True):
        raise ValueError("Unsupported split contract")
    start, core, train, valid, end = (date.fromisoformat(config[k]) for k in
                                     ["start", "train_core_end", "train_end", "validation_end", "test_end"])
    if not start <= core < train < valid < end:
        raise ValueError("Invalid calendar boundaries")


def split_for(origin, config):
    if origin < config["start"] or origin > config["test_end"]:
        return "out_of_window", "", ""
    if origin <= config["train_end"]:
        if origin <= config["train_core_end"]:
            return "train", "train_core", config["train_core_end"]
        return "train", "early_stop", config["train_end"]
    if origin <= config["validation_end"]:
        return "validation", "validation", config["validation_end"]
    return "test", "final_test", config["test_end"]


def feature_rows(events, labels, config):
    """Uses label values only after features are built; all current-day events enter as one batch."""
    validate_config(config)
    label_by_id = {r["origin_event_id"]: r for r in labels}
    event_by_id = {e["event_id"]: e for e in events}
    if len(label_by_id) != len(labels) or len(event_by_id) != len(events):
        raise ValueError("Duplicate event/label identity")
    complete = {e["event_id"] for e in events if e["event_state"] == "completed"}
    if complete != set(label_by_id):
        raise ValueError("Every completed day must have one label record")
    dates = defaultdict(list)
    for event in events:
        if event["event_state"] not in {"completed", "held", "cancelled_only", "excluded_only"}:
            raise ValueError("Unknown event state")
        date.fromisoformat(event["ordered_on"])
        if event["event_state"] in {"completed", "held"}:
            dates[event["ordered_on"]].append(event)
    histories = defaultdict(list)
    customer_days, customer_products, product_customers = defaultdict(set), defaultdict(set), defaultdict(set)
    product_counts, product_quantities = Counter(), Counter()
    known_snapshot = None
    results = []
    for day in sorted(dates):
        batch = sorted(dates[day], key=lambda e: e["event_id"])
        # Batch update first: predictions occur after the entire local purchase day has ended.
        for event in batch:
            customer, product = event["customer_id"], event["product_id"]
            observed = timestamp(event["snapshot_observed_at"])
            known_snapshot = max(known_snapshot, observed) if known_snapshot else observed
            key = (customer, product)
            if event["event_state"] == "held":
                histories[key] = []
                continue
            qty = int(event["quantity"])
            if qty <= 0:
                raise ValueError("Nonpositive event quantity")
            histories[key].append((date.fromisoformat(day), qty))
            customer_days[customer].add(day)
            customer_products[customer].add(product)
            product_customers[product].add(customer)
            product_counts[product] += 1
            product_quantities[product] += qty
        for event in batch:
            if event["event_state"] != "completed":
                continue
            customer, product = event["customer_id"], event["product_id"]
            history = histories[(customer, product)]
            today = date.fromisoformat(day)
            quantities = [qty for _, qty in history]
            gaps = [(b[0] - a[0]).days for a, b in zip(history, history[1:])]
            if any(g <= 0 for g in gaps):
                raise ValueError("Duplicate/unsorted customer-product day")
            features = {
                "product_id": product, "origin_weekday": today.weekday(), "origin_month": today.month,
                "episode_purchase_count": len(history), "episode_age_days": (today - history[0][0]).days,
                "qty_last": quantities[-1], "qty_mean": mean(quantities),
                "qty_mean_last3": mean(quantities[-3:]), "qty_median_last3": median(quantities[-3:]),
                "qty_std_last3": pstdev(quantities[-3:]),
                "qty_sum_7d": sum(q for d, q in history if (today - d).days < 7),
                "qty_sum_28d": sum(q for d, q in history if (today - d).days < 28),
                "gap_count": len(gaps), "gap_last_days": gaps[-1] if gaps else "",
                "gap_mean_days": mean(gaps) if gaps else "", "gap_median_days": median(gaps) if gaps else "",
                "customer_purchase_days": len(customer_days[customer]),
                "customer_product_count": len(customer_products[customer]),
                "product_purchase_events": product_counts[product],
                "product_customer_count": len(product_customers[product]),
                "product_mean_quantity": product_quantities[product] / product_counts[product],
            }
            label = label_by_id[event["event_id"]]
            if (label["customer_id"], label["product_id"], label["origin_on"]) != (customer, product, day):
                raise ValueError("Label/event identity mismatch")
            if label["label_state"] == "observed":
                target = event_by_id.get(label["target_event_id"])
                if (not target or target["event_state"] != "completed" or
                        (target["customer_id"], target["product_id"], target["ordered_on"], target["quantity"]) !=
                        (customer, product, label["target_on"], label["target_quantity"]) or
                        target["episode_id"] != event["episode_id"] or
                        int(target["episode_purchase_index"]) != int(event["episode_purchase_index"]) + 1 or
                        int(label["target_gap_days"]) != (date.fromisoformat(label["target_on"]) - today).days or
                        int(label["target_gap_days"]) <= 0):
                    raise ValueError("Invalid next-purchase target")
            elif label["label_state"] not in {"blocked_by_hold", "right_censored"}:
                raise ValueError("Unknown label state")
            outer, role, cutoff = split_for(day, config)
            outer_cutoff = config["train_end"] if outer == "train" else cutoff
            outcome = label["label_state"]
            if outer == "out_of_window":
                outcome = "out_of_window"
            elif outcome == "observed" and label["target_on"] > cutoff:
                outcome = "crosses_role_boundary"
            eligible = outcome == "observed"
            train_pool = outer == "train" and label["label_state"] == "observed" and label["target_on"] <= outer_cutoff
            asof = next_midnight(day)
            feature_available = known_snapshot <= asof
            # Scope/state availability remains unverified even if timestamps happen to pass.
            availability = eligible and feature_available and timestamp(label["label_snapshot_observed_at"]) < next_midnight(cutoff)
            results.append({"origin_event_id": event["event_id"], "customer_id": customer,
                            "origin_on": day, "prediction_asof": asof.isoformat(),
                            "feature_snapshot_observed_at": known_snapshot.isoformat(),
                            "outer_split": outer, "fit_role": role, "role_cutoff": cutoff,
                            **features, "label_state": label["label_state"], "target_on": label["target_on"],
                            "target_quantity": label["target_quantity"], "target_gap_days": label["target_gap_days"],
                            "label_snapshot_observed_at": label["label_snapshot_observed_at"],
                            "split_outcome": outcome, "exploratory_role_eligible": str(eligible).lower(),
                            "exploratory_train_pool_eligible": str(train_pool).lower(),
                            "feature_timestamp_available": str(feature_available).lower(),
                            "timestamp_availability_pass": str(bool(availability)).lower(),
                            "strict_training_ready": "false"})
    return results


def build(event_dir, config_path, output_dir):
    event_dir, config_path, output_dir = map(Path, (event_dir, config_path, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest = json.loads((event_dir / "report.json").read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    for filename in ["daily-events.csv", "next-purchase-labels.csv"]:
        if digest(event_dir / filename) != manifest["files"][filename]:
            raise ValueError("Event/label file hash mismatch")
    if config["start"] != manifest["first_ordered_on"] or config["test_end"] != manifest["last_ordered_on"]:
        raise ValueError("Config range differs from frozen snapshot range")
    rows = feature_rows(read_csv(event_dir / "daily-events.csv"), read_csv(event_dir / "next-purchase-labels.csv"), config)
    if not rows:
        raise ValueError("No completed purchase origins")
    output_dir.mkdir(parents=True, exist_ok=False)
    fields = list(rows[0])
    write_csv(output_dir / "all-origins-audit.csv", fields, rows)
    sizes = {}
    for role in ["train_core", "early_stop", "validation", "final_test"]:
        selected = [r for r in rows if r["fit_role"] == role and r["exploratory_role_eligible"] == "true"]
        write_csv(output_dir / (role + ".csv"), fields, selected)
        sizes[role] = len(selected)
    pool = [r for r in rows if r["exploratory_train_pool_eligible"] == "true"]
    write_csv(output_dir / "train_pool.csv", fields, pool)
    sizes["train_pool"] = len(pool)
    train, valid, end, start = (date.fromisoformat(config[k]) for k in ["train_end", "validation_end", "test_end", "start"])
    summary = {
        "version": "features-splits-v1", "mode": config["mode"], "split_config": config,
        "split_config_sha256": digest(config_path), "events_report_sha256": digest(event_dir / "report.json"),
        "source_event_files": {k: manifest["files"][k] for k in ["daily-events.csv", "next-purchase-labels.csv"]},
        "feature_columns": FEATURE_COLUMNS, "numeric_features": NUMERIC_FEATURES, "categorical_features": ["product_id"],
        "all_origins": len(rows), "calendar_days": {"train": (train - start).days + 1,
                                                      "validation": (valid - train).days, "test": (end - valid).days},
        "origins_by_outer_split": dict(Counter(r["outer_split"] for r in rows)),
        "role_outcomes": {role: dict(Counter(r["split_outcome"] for r in rows if r["fit_role"] == role))
                          for role in ["train_core", "early_stop", "validation", "final_test"]},
        "exploratory_file_rows": sizes,
        "feature_timestamp_available_origins": sum(r["feature_timestamp_available"] == "true" for r in rows),
        "timestamp_availability_pass_rows": sum(r["timestamp_availability_pass"] == "true" for r in rows),
        "strict_training_ready_rows": 0, "strict_training_ready": False, "final_test_metrics_computed": False,
        "note": "Calendar/order-time leakage checks do not repair retrospectively selected final states. All files are exploratory only. Fit using feature_columns allowlist, never target/metadata columns.",
        "files": {p.name: digest(p) for p in sorted(output_dir.glob("*.csv"))},
    }
    (output_dir / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_dir", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.event_dir, args.config, args.output_dir), ensure_ascii=False, indent=2))
