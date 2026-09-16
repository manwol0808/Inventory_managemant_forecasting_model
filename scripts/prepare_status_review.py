"""Apply the approved cancellation exclusion; keep other decisions pending.

Outputs retain every source row and its identifiers. This is not training data.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def status_decision(status):
    if status == "CANCEL_COMPLETE":
        return "exclude_cancelled"
    if status in {"PURCHASE_CONFIRMATION", "SHIPPING_COMPLETE"}:
        return "candidate_completed"
    if status in {"PRODUCT_PREPARATION", "SHIPPING_READY", "SHIPPING"}:
        return "review_in_progress"
    if status in {"RETURN_COMPLETE", "EXCHANGE_PRODUCT_PREPARATION", "EXCHANGE_PURCHASE_CONFIRMATION"}:
        return "review_return_exchange"
    return "review_unknown_status"


def prepare(source, output):
    source, output = Path(source), Path(output)
    if output.exists() or source.resolve() == output.resolve():
        raise ValueError("Output must be a new path")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames
        required = {"order_id", "order_item_code", "order_section_item_no", "section_status"}
        if not required.issubset(header or []):
            raise ValueError("Missing v2 tracking fields")
        rows = list(reader)
    extra = ["status_decision", "repeated_item_code", "training_ready"]
    if set(extra) & set(header):
        raise ValueError("Input is already annotated")
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("Malformed CSV row")
    keys = Counter((r["order_id"], r["order_item_code"]) for r in rows)
    decisions = Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header + extra)
        writer.writeheader()
        for row in rows:
            decision = status_decision(row["section_status"])
            decisions[decision] += 1
            writer.writerow({**row, "status_decision": decision,
                             "repeated_item_code": str(keys[(row["order_id"], row["order_item_code"])] > 1).lower(),
                             "training_ready": "false"})
    return {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output": str(output), "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "rows_preserved": len(rows), "decisions": dict(sorted(decisions.items())),
            "training_ready": False,
            "rule": "Only cancellation exclusion approved; completed rows are candidates, other statuses pending. No quantities aggregated."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), ensure_ascii=False, indent=2))
