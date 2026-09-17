"""Describe early cart schedules on saved repurchases; not a fresh stockout test."""
import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path

from scripts.predict_replenishment import cart_schedule
from scripts.build_features_and_splits import next_midnight
from scripts.run_baseline import digest


def evaluate(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    policies = []
    for lead in (0, 7, 10):
        margins = []
        clipped = 0
        for row in rows:
            # Final-test export omits prediction_asof; its evaluator checks this contract.
            available = next_midnight(row["origin_on"]).date()
            if row.get("prediction_asof"):
                assert datetime.fromisoformat(row["prediction_asof"]) == next_midnight(row["origin_on"])
            predicted = date.fromisoformat(row["predicted_on"])
            scheduled = date.fromisoformat(cart_schedule(predicted.isoformat(), available.isoformat(), lead))
            margins.append((date.fromisoformat(row["target_on"]) - scheduled).days)
            clipped += (predicted - available).days < lead
        policies.append({
            "lead_days": lead, "n": len(rows),
            "before_repurchase": sum(m > 0 for m in margins),
            "same_day": sum(m == 0 for m in margins),
            "late": sum(m < 0 for m in margins),
            "at_least_7_days_before": sum(m >= 7 for m in margins),
            "at_least_10_days_before": sum(m >= 10 for m in margins),
            "over_30_days_before": sum(m > 30 for m in margins),
            "clipped_to_feature_availability": clipped,
        })
    return {"source": str(path), "source_sha256": digest(Path(path)), "policies": policies}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps({
        "contract": "Descriptive reuse of consumed cohorts, observed repurchases only. Assumes daily execution from feature availability; no measured stockout or delivery labels. Lead choices fixed at 0, 7, 10; no test tuning.",
        "cohorts": [evaluate(path) for path in args.predictions],
    }, ensure_ascii=False, indent=2))
