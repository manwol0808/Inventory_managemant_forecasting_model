"""Link latest-snapshot section items without inferring event history or demand."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.prepare_status_review import status_decision

COMPLETED = {"PURCHASE_CONFIRMATION", "SHIPPING_COMPLETE"}


def classify_group(rows):
    statuses = {r["section_status"] for r in rows}
    if len({r["product_id"] for r in rows}) != 1:
        return "review_product_identity"
    if any(status_decision(s) == "review_unknown_status" for s in statuses):
        return "review_unknown_status"
    if any(status_decision(s) == "review_return_exchange" for s in statuses):
        return "review_return_exchange_linked"
    if any(status_decision(s) == "review_in_progress" for s in statuses):
        return "review_in_progress_linked"
    if statuses == {"CANCEL_COMPLETE"}:
        return "excluded_cancelled_group"
    if "CANCEL_COMPLETE" in statuses:
        return "review_partial_cancel_linked"
    if statuses <= COMPLETED:
        return "completed_group_candidate"
    raise ValueError("Unexpected status group")


def audit(source, output_dir):
    source, output_dir = Path(source), Path(output_dir)
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"customer_id", "order_id", "order_item_code", "order_section_item_no",
                    "order_section_code", "product_id", "section_status", "quantity"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Missing source tracking columns")
        rows = list(reader)
    groups = defaultdict(list)
    seen = set()
    for number, row in enumerate(rows, 1):
        if any(not row.get(k) for k in required):
            raise ValueError("Missing source value")
        key = (row["order_id"], row["order_section_item_no"])
        if key in seen or int(row["quantity"]) <= 0:
            raise ValueError("Duplicate section identity or invalid quantity")
        seen.add(key)
        groups[(row["order_id"], row["order_item_code"])].append((number, row))
    group_records, links = [], []
    decisions, patterns = Counter(), Counter()
    for (order_id, code), members in sorted(groups.items()):
        group_rows = [r for _, r in members]
        decision = classify_group(group_rows)
        quantities = Counter()
        for number, row in members:
            quantities[row["section_status"]] += int(row["quantity"])
            links.append({"source_record_number": number, "order_id": order_id,
                          "order_item_code": code, "order_section_item_no": row["order_section_item_no"],
                          "row_decision": status_decision(row["section_status"]),
                          "linked_group_decision": decision, "training_ready": "false"})
        decisions[decision] += 1
        if len(members) > 1:
            patterns[" | ".join(sorted(r["section_status"] for r in group_rows))] += 1
        group_records.append({"order_id": order_id, "order_item_code": code,
                              "source_rows": len(members),
                              "quantity_by_current_status": json.dumps(dict(sorted(quantities.items()))),
                              "linked_group_decision": decision, "training_ready": "false"})
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, records in [("item-groups.csv", group_records), ("source-links.csv", links)]:
        if not records:
            raise ValueError("Empty source")
        with (output_dir / filename).open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    report = {"source": str(source), "source_sha256": source_hash, "source_rows": len(rows),
              "linked_rows": len(links), "item_groups": len(groups),
              "group_decisions": dict(sorted(decisions.items())),
              "repeated_group_status_patterns": dict(sorted(patterns.items())),
              "training_ready": False,
              "scope": "Snapshot links by order_id + order_item_code; state subtotals are audit values, not demand labels or event history."}
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.output_dir), ensure_ascii=False, indent=2))
