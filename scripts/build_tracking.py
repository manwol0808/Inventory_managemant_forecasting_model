"""Build a local, lossless tracking index; this is NOT a training dataset.

Every CSV record survives, including identical records and all order statuses.
Only selected fields are copied; the unchanged source CSV holds all originals.
"""

import argparse
import csv
import hashlib
import io
import json
import sqlite3
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path

from scripts.audit_csv import EXPECTED


def build(source, output):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError("Output must be a new path and must not overwrite the source")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
    if next(reader, []) != EXPECTED:
        raise ValueError("Unexpected CSV header")
    records = []
    orders = {}
    for number, values in enumerate(reader, 1):
        if len(values) != len(EXPECTED):
            raise ValueError(f"Invalid column count at record {number}")
        row = dict(zip(EXPECTED, values))
        # Fail the whole build rather than silently dropping malformed records.
        if any(not row[key].strip() for key in ("member_code", "order_no", "prod_no")):
            raise ValueError(f"Missing key at record {number}")
        date.fromisoformat(row["주문일"])
        try:
            quantity = Decimal(row["수량"])
            if not quantity.is_finite() or quantity <= 0 or quantity != quantity.to_integral_value():
                raise InvalidOperation
        except InvalidOperation:
            raise ValueError(f"Invalid quantity at record {number}") from None
        order = (row["member_code"], row["주문일"], row["주문총액"], row["결제금액"])
        if row["order_no"] in orders and orders[row["order_no"]] != order:
            raise ValueError(f"Conflicting order fields at record {number}")
        orders[row["order_no"]] = order
        records.append((number, row["order_no"], row["prod_no"], row["수량"], row["섹션상태"]))

    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also guards against accidental replacement on reruns.
    with output.open("xb"):
        pass
    connection = sqlite3.connect(output)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            connection.executescript("""
                CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE orders (
                    order_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL,
                    ordered_on TEXT NOT NULL, order_total_raw TEXT NOT NULL,
                    paid_total_raw TEXT NOT NULL);
                CREATE TABLE records (
                    source_record_number INTEGER PRIMARY KEY,
                    order_id TEXT NOT NULL REFERENCES orders(order_id),
                    product_id TEXT NOT NULL, quantity_raw TEXT NOT NULL,
                    section_status TEXT NOT NULL);
                CREATE INDEX customer_history ON orders(customer_id, ordered_on, order_id);
                CREATE INDEX product_history ON records(product_id, order_id);
                CREATE VIEW customer_item_history AS
                    SELECT o.customer_id, o.ordered_on, o.order_id, r.product_id,
                           r.quantity_raw, r.section_status, r.source_record_number
                    FROM records r JOIN orders o USING(order_id);
                CREATE VIEW repeated_order_product_keys AS
                    SELECT order_id, product_id, COUNT(*) AS source_record_count
                    FROM records GROUP BY order_id, product_id HAVING COUNT(*) > 1;
            """)
            connection.executemany("INSERT INTO metadata VALUES (?, ?)", [
                ("source_sha256", digest), ("source_filename", source.name),
                ("schema_version", "1"), ("training_ready", "false"),
                ("record_number_definition", "1-based CSV data record, excluding header"),
                ("unresolved", "SKU units, repeated lines, status policy, target event"),
            ])
            connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
                                   [(key, *value) for key, value in orders.items()])
            connection.executemany("INSERT INTO records VALUES (?, ?, ?, ?, ?)", records)
            if connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Broken order linkage")
            summary = {
                "source_sha256": digest, "source_records": len(records),
                "tracked_records": connection.execute("SELECT COUNT(*) FROM customer_item_history").fetchone()[0],
                "orders": len(orders),
                "customers": connection.execute("SELECT COUNT(DISTINCT customer_id) FROM orders").fetchone()[0],
                "repeated_order_product_keys": connection.execute("SELECT COUNT(*) FROM repeated_order_product_keys").fetchone()[0],
                "training_ready": False,
            }
            if summary["tracked_records"] != summary["source_records"]:
                raise ValueError("Record reconciliation failed")
    finally:
        connection.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
