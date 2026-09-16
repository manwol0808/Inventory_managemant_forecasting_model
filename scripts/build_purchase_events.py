"""Build daily purchase events and next-purchase labels from frozen v1 policy rows."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from scripts.apply_training_policy import validate_product_exclusions


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace(" UTC", "+00:00").replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Timezone required")
    return parsed.astimezone(timezone.utc)


def write_csv(path, fields, records):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def build(source, policy_report, output_dir):
    source, policy_report, output_dir = map(Path, (source, policy_report, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    manifest = json.loads(policy_report.read_text(encoding="utf-8"))
    policy_version = manifest["policy_version"]
    if policy_version not in {"purchase-policy-v1", "purchase-policy-v2", "purchase-policy-v3", "purchase-policy-v4", "purchase-policy-v5"} or digest(source) != manifest["output_sha256"]:
        raise ValueError("Policy source hash/version mismatch")
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len(rows) != manifest["output_rows"]:
        raise ValueError("Policy row count mismatch")
    by_day, seen, numbers = defaultdict(list), set(), set()
    allowed = {"include_completed", "exclude_cancelled", "hold_unknown_status",
               "hold_return_exchange_linked", "hold_in_progress_linked", "hold_partial_cancel_linked"}
    excluded_products = {}
    if "product_exclusions" in manifest:
        excluded_products = validate_product_exclusions(manifest["product_exclusions"])
        allowed.update(excluded_products.values())
    excluded_customers = {}
    if "customer_exclusions" in manifest:
        excluded_customers = manifest["customer_exclusions"]["customers"]
        if not excluded_customers or set(excluded_customers.values()) != {"exclude_internal_test"}:
            raise ValueError("Invalid excluded-customer manifest")
        allowed.add("exclude_internal_test")
    raw_quantity = 0
    for row in rows:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("Malformed row")
        key = (row["order_id"], row["order_section_item_no"])
        number = int(row["source_record_number"])
        qty = int(row["quantity"])
        decision = row["purchase_decision"]
        if key in seen or number in numbers or qty <= 0 or decision not in allowed:
            raise ValueError("Invalid or duplicate row")
        seen.add(key)
        numbers.add(number)
        if row["policy_version"] != policy_version or row["training_ready"] != "false":
            raise ValueError("Unexpected policy annotation")
        expected_exclusion = ("exclude_internal_test" if row["customer_id"] in excluded_customers else
                              excluded_products.get(row["product_id"]))
        if ((expected_exclusion and decision != expected_exclusion) or
                (not expected_exclusion and decision.startswith("exclude_") and decision != "exclude_cancelled")):
            raise ValueError("Product exclusion annotation mismatch")
        if row["sequence_barrier"] != str(decision.startswith("hold_")).lower():
            raise ValueError("Invalid barrier annotation")
        expected = str(qty) if decision == "include_completed" else ""
        if row["eligible_quantity"] != expected:
            raise ValueError("Invalid eligible quantity")
        ordered = timestamp(row["ordered_at"])
        extracted = timestamp(row["extracted_at"])
        if ordered.astimezone(ZoneInfo("Asia/Seoul")).date().isoformat() != row["ordered_on"]:
            raise ValueError("Korean order date mismatch")
        if extracted < ordered:
            raise ValueError("Extraction predates order")
        raw_quantity += qty
        by_day[(row["customer_id"], row["product_id"], row["ordered_on"])].append(row)
    if numbers != set(range(1, len(rows) + 1)):
        raise ValueError("Source record numbers are not complete")
    events, links, by_series = [], [], defaultdict(list)
    row_decisions, quantity_by_disposition = Counter(), Counter()
    for (customer, product, day), members in sorted(by_day.items()):
        active = [r for r in members if not r["purchase_decision"].startswith("exclude_")]
        held = any(r["sequence_barrier"] == "true" for r in active)
        state = "held" if held else "completed" if active else "cancelled_only"
        if not active and any(r["purchase_decision"] != "exclude_cancelled" for r in members):
            state = "excluded_only"
        event_id = hashlib.sha256(json.dumps([customer, product, day], ensure_ascii=False).encode()).hexdigest()
        qty = sum(int(r["quantity"]) for r in active) if state == "completed" else ""
        event = {"event_id": event_id, "customer_id": customer, "product_id": product,
                 "ordered_on": day, "event_state": state, "quantity": qty,
                 "source_rows": len(members), "order_count": len({r["order_id"] for r in members}),
                 "snapshot_observed_at": max(timestamp(r["extracted_at"]) for r in members).isoformat(),
                 "episode_id": "", "episode_purchase_index": "",
                 "historical_availability_verified": "false"}
        events.append(event)
        by_series[(customer, product)].append(event)
        for row in members:
            decision = row["purchase_decision"]
            disposition = (decision if decision.startswith("exclude_") else
                           "include_completed" if state == "completed" else
                           "hold_same_day" if decision == "include_completed" else decision)
            contribution = int(row["quantity"]) if disposition == "include_completed" else 0
            row_decisions[disposition] += 1
            quantity_by_disposition[disposition] += int(row["quantity"])
            links.append({"source_record_number": row["source_record_number"],
                          "order_id": row["order_id"], "order_item_code": row["order_item_code"],
                          "order_section_item_no": row["order_section_item_no"], "event_id": event_id,
                          "row_disposition": disposition, "raw_quantity": row["quantity"],
                          "event_quantity_contribution": contribution})
    labels = []
    for sequence in by_series.values():
        active = [e for e in sequence if e["event_state"] in {"completed", "held"}]
        episode, index = "", 0
        for i, event in enumerate(active):
            if event["event_state"] == "held":
                episode, index = "", 0
                continue
            if not episode:
                episode = event["event_id"]
            index += 1
            event["episode_id"], event["episode_purchase_index"] = episode, index
            following = active[i + 1] if i + 1 < len(active) else None
            observed = following is not None and following["event_state"] == "completed"
            state = "observed" if observed else "blocked_by_hold" if following else "right_censored"
            labels.append({"origin_event_id": event["event_id"], "customer_id": event["customer_id"],
                           "product_id": event["product_id"], "origin_on": event["ordered_on"],
                           "episode_id": episode, "label_state": state,
                           "target_event_id": following["event_id"] if observed else "",
                           "target_on": following["ordered_on"] if observed else "",
                           "target_quantity": following["quantity"] if observed else "",
                           "target_gap_days": (date.fromisoformat(following["ordered_on"]) -
                                               date.fromisoformat(event["ordered_on"])).days if observed else "",
                           "blocking_event_id": following["event_id"] if state == "blocked_by_hold" else "",
                           "label_snapshot_observed_at": max(event["snapshot_observed_at"],
                                                             following["snapshot_observed_at"]) if observed else "",
                           "training_ready": "false"})
    contributions = Counter()
    for link in links:
        contributions[link["event_id"]] += link["event_quantity_contribution"]
    if (len(links) != len(rows) or sum(quantity_by_disposition.values()) != raw_quantity or
            any(contributions[e["event_id"]] != (e["quantity"] or 0) for e in events)):
        raise ValueError("Source to event reconciliation failed")
    if any(l["label_state"] == "observed" and l["target_gap_days"] <= 0 for l in labels):
        raise ValueError("Nonpositive purchase gap")
    output_dir.mkdir(parents=True, exist_ok=False)
    event_fields = list(events[0])
    label_fields = ["origin_event_id", "customer_id", "product_id", "origin_on", "episode_id", "label_state",
                    "target_event_id", "target_on", "target_quantity", "target_gap_days", "blocking_event_id",
                    "label_snapshot_observed_at", "training_ready"]
    write_csv(output_dir / "daily-events.csv", event_fields, events)
    write_csv(output_dir / "source-links.csv", list(links[0]), sorted(links, key=lambda r: int(r["source_record_number"])))
    write_csv(output_dir / "next-purchase-labels.csv", label_fields, labels)
    observed_pairs = [l for l in labels if l["label_state"] == "observed"]
    write_csv(output_dir / "observed-pairs.csv", label_fields, observed_pairs)
    report = {
        "version": policy_version.replace("purchase-policy-", "purchase-events-"), "policy_version": policy_version,
        "source": str(source), "source_sha256": digest(source), "raw_source_sha256": manifest["source_sha256"],
        "policy_report_sha256": digest(policy_report), "source_rows": len(rows), "linked_rows": len(links),
        "customer_product_series": len(by_series), "daily_event_rows": len(events),
        "event_states": dict(sorted(Counter(e["event_state"] for e in events).items())),
        "label_states": dict(sorted(Counter(l["label_state"] for l in labels).items())),
        "row_dispositions": dict(sorted(row_decisions.items())),
        "raw_quantity_by_disposition": dict(sorted(quantity_by_disposition.items())),
        "included_quantity": sum(contributions.values()),
        "observed_pair_series": len({(l["customer_id"], l["product_id"]) for l in observed_pairs}),
        "single_completed_event_series": sum(sum(e["event_state"] == "completed" for e in seq) == 1
                                              for seq in by_series.values()),
        "first_ordered_on": min(e["ordered_on"] for e in events),
        "last_ordered_on": max(e["ordered_on"] for e in events),
        "last_day_completeness_verified": False, "historical_availability_verified": False,
        "training_ready": False, "reconciliation_passed": True,
        "files": {p.name: digest(p) for p in sorted(output_dir.glob("*.csv"))},
        "scope": "Retrospective snapshot events/labels only; held days break sequences; cancelled-only/excluded-only days are audit records, not purchase events or model features. No features, splits, or training."}
    if excluded_products:
        report["product_exclusions"] = manifest["product_exclusions"]
        report["product_exclusions_sha256"] = manifest["product_exclusions_sha256"]
    if excluded_customers:
        report["customer_exclusions"] = manifest["customer_exclusions"]
        report["customer_exclusions_sha256"] = manifest["customer_exclusions_sha256"]
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--policy-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.policy_report, args.output_dir), ensure_ascii=False, indent=2))
