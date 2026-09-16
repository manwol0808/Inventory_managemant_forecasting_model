"""Apply v1 snapshot eligibility, preserving all rows; not a training dataset."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from scripts.audit_item_lifecycle import classify_group


def validate_product_exclusions(manifest):
    reasons = {"product-exclusions-v1": {"exclude_prepaid_credit"},
               "product-exclusions-v2": {"exclude_prepaid_credit", "exclude_sample", "exclude_gift"},
               "product-exclusions-v3": {"exclude_prepaid_credit", "exclude_sample", "exclude_gift",
                                         "exclude_test_product", "hold_non_catalog_payment"}}
    allowed = reasons.get(manifest.get("version"), set())
    products = manifest.get("products")
    if (not allowed or not isinstance(products, dict) or not products or
            any(not key or reason not in allowed for key, reason in products.items())):
        raise ValueError("Invalid product exclusion configuration")
    return products


def apply_policy(source, output_dir, product_exclusions=None, customer_exclusions=None):
    source, output_dir = Path(source), Path(output_dir)
    exclusions, exclusion_manifest = {}, None
    if product_exclusions is not None:
        product_exclusions = Path(product_exclusions)
        exclusion_manifest = json.loads(product_exclusions.read_text(encoding="utf-8"))
        exclusions = validate_product_exclusions(exclusion_manifest)
    policy_version = "purchase-policy-v2" if exclusion_manifest else "purchase-policy-v1"
    excluded_customers, customer_manifest = {}, None
    if customer_exclusions is not None:
        customer_exclusions = Path(customer_exclusions)
        customer_manifest = json.loads(customer_exclusions.read_text(encoding="utf-8"))
        excluded_customers = customer_manifest["customers"]
        if (customer_manifest.get("version") != "test-account-exclusions-v1" or
                not isinstance(excluded_customers, dict) or not excluded_customers or
                any(not key or reason != "exclude_internal_test" for key, reason in excluded_customers.items())):
            raise ValueError("Invalid customer exclusion configuration")
        policy_version = "purchase-policy-v3"
    if exclusion_manifest and exclusion_manifest["version"] == "product-exclusions-v2":
        policy_version = "purchase-policy-v4"
    if exclusion_manifest and exclusion_manifest["version"] == "product-exclusions-v3":
        policy_version = "purchase-policy-v5"
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        header = reader.fieldnames or []
        rows = list(reader)
    required = {"customer_id", "order_id", "product_id", "order_item_code",
                "order_section_item_no", "ordered_at", "ordered_on", "quantity",
                "source_updated_at", "extracted_at", "section_status"}
    extra = ["source_record_number", "policy_version", "purchase_decision",
             "eligible_quantity", "sequence_barrier", "training_ready"]
    if not rows or not required.issubset(header) or set(extra) & set(header):
        raise ValueError("Expected unannotated v2 source")
    groups, item_products, order_identity = defaultdict(list), {}, {}
    seen = set()
    for number, row in enumerate(rows, 1):
        if None in row or any(v is None for v in row.values()) or any(not row[k] for k in required):
            raise ValueError("Missing or malformed source values")
        detail = (row["order_id"], row["order_section_item_no"])
        if detail in seen or int(row["quantity"]) <= 0:
            raise ValueError("Duplicate detail or invalid quantity")
        seen.add(detail)
        item = (row["order_id"], row["order_item_code"])
        if item_products.setdefault(item, row["product_id"]) != row["product_id"]:
            raise ValueError("Conflicting item product identity")
        identity = (row["customer_id"], row["ordered_at"], row["ordered_on"])
        if order_identity.setdefault(row["order_id"], identity) != identity:
            raise ValueError("Conflicting order identity")
        groups[(row["order_id"], row["product_id"])].append((number, row))
    annotations, group_counts, row_counts, quantity_counts = {}, Counter(), Counter(), Counter()
    for members in groups.values():
        decision = (excluded_customers.get(members[0][1]["customer_id"]) or
                    exclusions.get(members[0][1]["product_id"]) or classify_group([r for _, r in members]))
        decision = {"completed_group_candidate": "include_completed",
                    "excluded_cancelled_group": "exclude_cancelled"}.get(
                        decision, decision.replace("review_", "hold_"))
        group_counts[decision] += 1
        for number, row in members:
            quantity = int(row["quantity"])
            annotations[number] = {
                "source_record_number": number, "policy_version": policy_version,
                "purchase_decision": decision,
                "eligible_quantity": quantity if decision == "include_completed" else "",
                "sequence_barrier": str(decision.startswith("hold_")).lower(),
                "training_ready": "false"}
            row_counts[decision] += 1
            quantity_counts[decision] += quantity
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "rows.csv"
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header + extra)
        writer.writeheader()
        for number, row in enumerate(rows, 1):
            writer.writerow({**row, **annotations[number]})
    report = {
        "policy_version": policy_version, "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_rows": len(rows), "output_rows": len(annotations),
        "order_product_groups": len(groups), "group_decisions": dict(sorted(group_counts.items())),
        "row_decisions": dict(sorted(row_counts.items())),
        "raw_quantity_by_decision": dict(sorted(quantity_counts.items())),
        "training_ready": False,
        "scope": "Latest snapshot eligibility only. Holds are sequence barriers; missing eligible quantity is not zero demand. Historical availability is unverified."}
    if exclusion_manifest:
        report["product_exclusions"] = exclusion_manifest
        report["product_exclusions_sha256"] = hashlib.sha256(product_exclusions.read_bytes()).hexdigest()
        report["unmatched_exclusion_product_ids"] = sorted(set(exclusions) - {r["product_id"] for r in rows})
    if customer_manifest:
        report["customer_exclusions"] = customer_manifest
        report["customer_exclusions_sha256"] = hashlib.sha256(customer_exclusions.read_bytes()).hexdigest()
        report["unmatched_exclusion_customer_ids"] = sorted(set(excluded_customers) - {r["customer_id"] for r in rows})
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--product-exclusions", type=Path)
    parser.add_argument("--customer-exclusions", type=Path)
    args = parser.parse_args()
    print(json.dumps(apply_policy(args.source, args.output_dir, args.product_exclusions, args.customer_exclusions), ensure_ascii=False, indent=2))
