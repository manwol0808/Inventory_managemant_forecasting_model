import csv
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path

from scripts.audit_csv import EXPECTED
from scripts.build_tracking import build


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / "source.csv"
        self.output = Path(self.tmp.name) / "tracking.sqlite"

    def write_rows(self, rows):
        with self.source.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(EXPECTED)
            writer.writerows(rows)

    def row(self, order="001", product="007", status="PURCHASE_CONFIRMATION"):
        return ["0001", "PRIVATE NAME", "PRIVATE PHONE", "", "true", "2026-01-01",
                order, "2026-04-10", "100", "100", status, product, "상품", "3"]

    def test_preserves_identical_lines_statuses_and_order_amount_once(self):
        self.write_rows([self.row(), self.row(), self.row(product="008", status="CANCEL_COMPLETE")])
        before = self.source.read_bytes()
        result = build(self.source, self.output)
        self.assertEqual(result["tracked_records"], 3)
        self.assertEqual(result["orders"], 1)
        self.assertEqual(result["source_sha256"], hashlib.sha256(before).hexdigest())
        with sqlite3.connect(self.output) as db:
            self.assertEqual(db.execute("SELECT SUM(CAST(order_total_raw AS INT)) FROM orders").fetchone()[0], 100)
            self.assertEqual(db.execute("SELECT source_record_number FROM records ORDER BY 1").fetchall(), [(1,), (2,), (3,)])
            self.assertEqual(db.execute("SELECT customer_id FROM orders").fetchone()[0], "0001")
            self.assertEqual(db.execute("SELECT COUNT(*) FROM records WHERE section_status='CANCEL_COMPLETE'").fetchone()[0], 1)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertNotIn(b"PRIVATE", self.output.read_bytes())
        with self.assertRaises(ValueError):
            build(self.source, self.output)

    def test_conflicting_order_fails_before_output(self):
        other = self.row()
        other[0] = "another customer"
        self.write_rows([self.row(), other])
        with self.assertRaises(ValueError):
            build(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_malformed_inputs_do_not_drop_rows(self):
        for index, value in [(0, ""), (7, "not a date"), (13, "NaN"), (13, "-1")]:
            with self.subTest(index=index, value=value):
                row = self.row()
                row[index] = value
                self.write_rows([row])
                with self.assertRaises(ValueError):
                    build(self.source, self.output)
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
