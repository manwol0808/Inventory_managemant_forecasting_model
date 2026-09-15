"""Read-only structural audit. Outputs aggregate JSON; never rewrites the CSV."""

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path


EXPECTED = [
    "member_code", "고객명", "연락처", "등급", "사업자여부", "가입일",
    "order_no", "주문일", "주문총액", "결제금액", "섹션상태", "prod_no", "상품명", "수량",
]


def audit(path):
    raw = path.read_bytes()
    text = raw.decode("utf-8-sig", errors="strict")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header = next(reader, [])
    if header != EXPECTED:
        raise ValueError("Unexpected CSV header; inspect schema before continuing")
    records = list(reader)
    bad_width = sum(len(row) != len(header) for row in records)
    if bad_width:
        raise ValueError(f"CSV records with invalid column count: {bad_width}")
    rows = [dict(zip(header, row)) for row in records]
    orders = defaultdict(list)
    customer_orders = defaultdict(set)
    pair_orders = defaultdict(set)
    customer_dates = defaultdict(set)
    customer_day_orders = defaultdict(set)
    key_counts = Counter()
    dates = []
    quantity = []
    invalid_dates = Counter()
    invalid_numbers = Counter()
    for row in rows:
        orders[row["order_no"]].append(row)
        customer_orders[row["member_code"]].add(row["order_no"])
        pair_orders[(row["member_code"], row["prod_no"])].add(row["order_no"])
        key_counts[(row["member_code"], row["order_no"], row["prod_no"])] += 1
        for field in ("주문일", "가입일"):
            if not row[field]:
                continue
            try:
                parsed = date.fromisoformat(row[field])
            except ValueError:
                invalid_dates[field] += 1
            else:
                if field == "주문일":
                    dates.append(parsed)
                    customer_dates[row["member_code"]].add(parsed)
                    customer_day_orders[(row["member_code"], parsed)].add(row["order_no"])
        for field in ("수량", "주문총액", "결제금액"):
            try:
                number = Decimal(row[field])
                if not number.is_finite():
                    raise InvalidOperation
            except InvalidOperation:
                invalid_numbers[field] += 1
            else:
                if field == "수량":
                    quantity.append(number)
    conflicting = {}
    for field in ("member_code", "주문일", "주문총액", "결제금액"):
        conflicting[field] = sum(
            len({row[field] for row in group}) > 1 for group in orders.values()
        )
    exact = Counter(tuple(row) for row in records)
    gaps = []
    for values in customer_dates.values():
        ordered = sorted(values)
        gaps.extend((b - a).days for a, b in zip(ordered, ordered[1:]))
    return {
        "source": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "encoding": "UTF-8",
        "utf8_bom": raw.startswith(b"\xef\xbb\xbf"),
        "replacement_character_count": text.count("\ufffd"),
        "columns": header,
        "row_count": len(rows),
        "column_count": len(header),
        "invalid_width_rows": bad_width,
        "unique_customers": len(customer_orders),
        "unique_orders": len(orders),
        "unique_products": len({r["prod_no"] for r in rows}),
        "unique_customer_product_pairs": len(pair_orders),
        "date_min": min(dates).isoformat() if dates else None,
        "date_max": max(dates).isoformat() if dates else None,
        "calendar_days_inclusive": (max(dates) - min(dates)).days + 1 if dates else 0,
        "multirow_orders": sum(len(group) > 1 for group in orders.values()),
        "repeated_customer_order_product_keys": sum(n > 1 for n in key_counts.values()),
        "surplus_rows_by_customer_order_product_key": sum(n - 1 for n in key_counts.values()),
        "exact_duplicate_record_groups": sum(n > 1 for n in exact.values()),
        "exact_duplicate_surplus_rows": sum(n - 1 for n in exact.values()),
        "order_fields_with_conflicts": conflicting,
        "missing_by_column": {field: sum(not r[field].strip() for r in rows) for field in header},
        "invalid_dates": dict(invalid_dates),
        "invalid_numbers": dict(invalid_numbers),
        "status_counts": dict(sorted(Counter(r["섹션상태"] for r in rows).items())),
        "quantity": {
            "zero_rows": sum(q == 0 for q in quantity),
            "negative_rows": sum(q < 0 for q in quantity),
            "non_integer_rows": sum(q != q.to_integral_value() for q in quantity),
            "min": str(min(quantity)) if quantity else None,
            "max": str(max(quantity)) if quantity else None,
            "raw_sum_not_deduplicated": str(sum(quantity)),
        },
        "customers_by_number_of_orders": dict(sorted(Counter(len(v) for v in customer_orders.values()).items())),
        "customer_product_pairs_by_number_of_orders": dict(sorted(Counter(len(v) for v in pair_orders.values()).items())),
        "customers_with_at_least_three_distinct_purchase_dates": sum(len(v) >= 3 for v in customer_dates.values()),
        "customer_days_with_multiple_order_numbers": sum(len(v) > 1 for v in customer_day_orders.values()),
        "observed_positive_customer_day_gaps": {
            "count": len(gaps), "min": min(gaps) if gaps else None, "max": max(gaps) if gaps else None,
        },
        "scope": "Raw structural profile, all statuses included. No duplicate resolution, no demand labels, no model scores.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output and args.output.resolve() == args.source.resolve():
        parser.error("Output must not overwrite the source CSV")
    report = audit(args.source)
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"Saved aggregate audit: {args.output}")
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
