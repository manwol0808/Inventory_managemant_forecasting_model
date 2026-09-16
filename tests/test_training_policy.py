import csv
import tempfile
import unittest
from pathlib import Path

from scripts.apply_training_policy import apply_policy


class TrainingPolicyTests(unittest.TestCase):
    def run_policy(self, statuses):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.csv"
            rows = [{"customer_id": "001", "order_id": "01", "product_id": product,
                     "order_item_code": str(i), "order_section_item_no": str(i),
                     "ordered_at": "2026-05-01 00:00:00 UTC", "ordered_on": "2026-05-01",
                     "quantity": str(qty), "source_updated_at": "2026-05-02 00:00:00 UTC",
                     "extracted_at": "2026-09-16 00:00:00 UTC", "section_status": status}
                    for i, (product, status, qty) in enumerate(statuses)]
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            output = Path(temp) / "output"
            report = apply_policy(source, output)
            with (output / "rows.csv").open() as stream:
                result = list(csv.DictReader(stream))
            self.assertEqual([{k: r[k] for k in rows[0]} for r in result], rows)
            with self.assertRaises(FileExistsError):
                apply_policy(source, output)
            return report, result

    def test_hold_crosses_item_codes_but_not_products(self):
        _, rows = self.run_policy([("A", "PURCHASE_CONFIRMATION", 3),
                                   ("A", "RETURN_COMPLETE", 2),
                                   ("B", "PURCHASE_CONFIRMATION", 4)])
        self.assertEqual([r["purchase_decision"] for r in rows],
                         ["hold_return_exchange_linked"] * 2 + ["include_completed"])
        self.assertEqual([r["eligible_quantity"] for r in rows], ["", "", "4"])

    def test_split_completed_keeps_both_quantities(self):
        report, rows = self.run_policy([("A", "PURCHASE_CONFIRMATION", 2),
                                        ("A", "SHIPPING_COMPLETE", 3)])
        self.assertEqual(report["group_decisions"], {"include_completed": 1})
        self.assertEqual(sum(int(r["eligible_quantity"]) for r in rows), 5)

    def test_cancel_is_not_zero_label_or_barrier(self):
        _, rows = self.run_policy([("A", "CANCEL_COMPLETE", 2)])
        self.assertEqual(rows[0]["purchase_decision"], "exclude_cancelled")
        self.assertEqual(rows[0]["eligible_quantity"], "")
        self.assertEqual(rows[0]["sequence_barrier"], "false")

    def test_partial_cancel_progress_and_unknown_block(self):
        for status in ["CANCEL_COMPLETE", "SHIPPING", "FUTURE_STATUS",
                       "EXCHANGE_PURCHASE_CONFIRMATION"]:
            with self.subTest(status=status):
                _, rows = self.run_policy([("A", "PURCHASE_CONFIRMATION", 2), ("A", status, 1)])
                self.assertTrue(all(r["sequence_barrier"] == "true" for r in rows))
                self.assertTrue(all(r["training_ready"] == "false" for r in rows))
