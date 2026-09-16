"""Merge explicitly selected v2 export shards without deleting any source rows.

Reject duplicate files, overlapping orders, invalid quantities and row-count
mismatches. This creates a tracking snapshot, not cleaned training demand.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path


def merge(sources, output, expected_rows):
    output = Path(output)
    if output.exists():
        raise ValueError("Choose a new output path; overwriting is disabled")
    rows, manifests, hashes = [], [], set()
    header = None
    prior_orders, record_keys, business_keys = set(), set(), set()
    required = {"customer_id", "order_id", "ordered_on", "order_section_item_no",
                "order_item_code", "section_position", "item_position", "product_id",
                "quantity", "quantity_raw", "latest_timestamp_row_count",
                "extracted_at", "source_updated_at", "section_status", "item_json_sha256"}
    for source in map(Path, sources):
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest in hashes:
            raise ValueError("Duplicate input file")
        hashes.add(digest)
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError("Required v2 columns missing")
            if header is not None and reader.fieldnames != header:
                raise ValueError("Shard schemas differ")
            header = reader.fieldnames
            shard = list(reader)
        orders = {r["order_id"] for r in shard}
        if prior_orders & orders:
            raise ValueError("Orders overlap between shards")
        prior_orders.update(orders)
        for r in shard:
            if None in r or any(v is None for v in r.values()):
                raise ValueError("Malformed CSV record")
            for field in required - {"quantity", "quantity_raw"}:
                if not r[field].strip():
                    raise ValueError(f"Missing required value: {field}")
            date.fromisoformat(r["ordered_on"])
            raw_quantity = Decimal(r["quantity_raw"])
            if not raw_quantity.is_finite() or raw_quantity <= 0 or raw_quantity != int(r["quantity"]):
                raise ValueError("Invalid or inconsistent quantity")
            key = (r["order_id"], int(r["section_position"]), int(r["item_position"]))
            business_key = (r["order_id"], r["order_section_item_no"])
            if key in record_keys or business_key in business_keys:
                raise ValueError("Repeated source position or section item identity")
            record_keys.add(key)
            business_keys.add(business_key)
        manifests.append({"file": source.name, "sha256": digest, "rows": len(shard),
                          "extracted_at": sorted({r["extracted_at"] for r in shard})})
        rows.extend(shard)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, received {len(rows)}")
    rows.sort(key=lambda r: (r["ordered_on"], r["order_id"], int(r["section_position"]), int(r["item_position"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    item_codes = Counter((r["order_id"], r["order_item_code"]) for r in rows)
    return {
        "inputs": manifests, "output": str(output),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rows": len(rows), "orders": len(prior_orders),
        "customers": len({r["customer_id"] for r in rows}),
        "unique_source_positions": len(record_keys), "unique_section_item_keys": len(business_keys),
        "repeated_order_item_codes": sum(n > 1 for n in item_codes.values()),
        "latest_timestamp_tie_orders": len({r["order_id"] for r in rows if int(r["latest_timestamp_row_count"]) > 1}),
        "statuses": dict(sorted(Counter(r["section_status"] for r in rows).items())),
        "date_min": min(r["ordered_on"] for r in rows),
        "date_max": max(r["ordered_on"] for r in rows),
        "training_ready": False,
        "scope": "All statuses retained; separate live shard snapshots; quantities not summed or deduplicated.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(merge(args.sources, args.output, args.expected_rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
