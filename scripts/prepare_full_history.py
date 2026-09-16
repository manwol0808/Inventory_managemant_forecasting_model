"""Validate a complete bq CSV snapshot; quarantine unidentifiable customer prefixes.

The raw export is immutable. Never infer a product number from its name/SKU.
After a customer's final unidentified product day, restart observable history.
This deliberately sacrifices identifiable earlier rows to avoid jumping over a
potential purchase of the same product. It is retrospective exploration only.
"""
import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_purchase_events import digest, write_csv
from scripts.merge_order_exports import merge


def partition(rows, train_cutoff):
    cutoffs = {}
    for row in rows:
        if not row["product_id"]:
            customer = row["customer_id"]
            cutoffs[customer] = max(cutoffs.get(customer, ""), row["ordered_on"])
    # Do not let a validation/test-time missing identity prune training history.
    if any(day > train_cutoff for day in cutoffs.values()):
        raise ValueError("Unidentified product beyond training cutoff requires causal barrier handling")
    kept, quarantine = [], []
    for row in rows:
        if row["ordered_on"] <= cutoffs.get(row["customer_id"], ""):
            reason = "missing_product_id" if not row["product_id"] else "prefix_before_unknown_product"
            quarantine.append({**row, "quarantine_reason": reason})
        else:
            kept.append(row)
    return kept, quarantine, cutoffs


def prepare(source, table_metadata, output, train_cutoff):
    source, table_metadata, output = map(Path, (source, table_metadata, output))
    if output.exists():
        raise FileExistsError(output)
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    table = json.loads(table_metadata.read_text())
    if not rows or len(rows) != int(table["numRows"]):
        raise ValueError("Server/local result row count mismatch")
    fields = list(rows[0])
    seen = set()
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("Malformed CSV")
        key = (row["order_id"], row["order_section_item_no"])
        if key in seen:
            raise ValueError("Duplicate detail identity")
        seen.add(key)
        # bq's CSV TIMESTAMP rendering is UTC without a suffix and at second precision.
        # Preserve the original export; normalize only this derived input.
        for field in ("ordered_at", "source_updated_at", "extracted_at"):
            value = datetime.fromisoformat(row[field])
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            row[field] = value.astimezone(timezone.utc).isoformat()
        if int(row["quantity"]) <= 0 or float(row["quantity_raw"]) != int(row["quantity"]):
            raise ValueError("Invalid quantity")
    kept, quarantine, cutoffs = partition(rows, train_cutoff)
    output.mkdir(parents=True)
    write_csv(output / "identified-export.csv", fields, kept)
    write_csv(output / "quarantine.csv", fields + ["quarantine_reason"], quarantine)
    audit = merge([output / "identified-export.csv"], output / "orders.csv", len(kept))
    if audit["latest_timestamp_tie_orders"]:
        raise ValueError("Latest-version tie requires review")
    report = {
        "version": "full-history-ingestion-v1", "raw_sha256": digest(source),
        "server_table_metadata_sha256": digest(table_metadata), "server_rows": int(table["numRows"]),
        "raw_rows": len(rows), "kept_rows": len(kept), "quarantine_rows": len(quarantine),
        "quarantine_reasons": dict(Counter(r["quarantine_reason"] for r in quarantine)),
        "affected_customers": len(cutoffs), "latest_missing_product_day": max(cutoffs.values(), default=None),
        "raw_date_min": min(r["ordered_on"] for r in rows), "raw_date_max": max(r["ordered_on"] for r in rows),
        "orders": audit["orders"], "customers": audit["customers"],
        "files": {p.name: digest(p) for p in output.glob("*.csv")},
        "identification_policy": "No inferred IDs. Quarantine all customer rows through last unknown-product date; retain later history.",
        "timestamp_precision": "bq CSV UTC seconds; raw export and job metadata retained",
        "strict_training_ready": False,
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--table-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-cutoff", default="2026-07-13")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.table_metadata, args.output, args.train_cutoff), ensure_ascii=False, indent=2))
