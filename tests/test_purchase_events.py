import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.apply_training_policy import apply_policy
from scripts.build_purchase_events import build


class PurchaseEventsTests(unittest.TestCase):
    def prepare(self, root, specs):
        rows = [{"customer_id": "001", "order_id": str(i), "product_id": product,
                 "order_item_code": str(i), "order_section_item_no": str(i),
                 "ordered_at": day + " 01:00:00 UTC", "ordered_on": day,
                 "quantity": str(qty), "source_updated_at": "2026-09-16 00:00:00 UTC",
                 "extracted_at": "2026-09-16 01:00:00 UTC", "section_status": status}
                for i, (day, product, status, qty) in enumerate(specs)]
        source = root / "source.csv"
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        policy = root / "policy"
        apply_policy(source, policy)
        return policy

    def run_case(self, specs):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy = self.prepare(root, specs)
            output = root / "events"
            report = build(policy / "rows.csv", policy / "report.json", output)
            result = {}
            for name in ["daily-events", "source-links", "next-purchase-labels", "observed-pairs"]:
                with (output / (name + ".csv")).open() as stream:
                    result[name] = list(csv.DictReader(stream))
            with self.assertRaises(FileExistsError):
                build(policy / "rows.csv", policy / "report.json", output)
            return report, result

    def test_sum_same_day_and_keep_products_separate(self):
        _, result = self.run_case([
            ("2026-05-01", "A", "PURCHASE_CONFIRMATION", 2),
            ("2026-05-01", "A", "SHIPPING_COMPLETE", 3),
            ("2026-05-03", "B", "PURCHASE_CONFIRMATION", 99),
            ("2026-05-10", "A", "PURCHASE_CONFIRMATION", 4)])
        self.assertEqual(result["daily-events"][0]["quantity"], "5")
        pair, = result["observed-pairs"]
        self.assertEqual((pair["target_gap_days"], pair["target_quantity"]), ("9", "4"))
        self.assertEqual(len(result["source-links"]), 4)

    def test_hold_blocks_whole_day_and_does_not_bridge(self):
        report, result = self.run_case([
            ("2026-05-01", "A", "PURCHASE_CONFIRMATION", 2),
            ("2026-05-05", "A", "PURCHASE_CONFIRMATION", 3),
            ("2026-05-05", "A", "RETURN_COMPLETE", 1),
            ("2026-05-10", "A", "PURCHASE_CONFIRMATION", 4),
            ("2026-05-20", "A", "PURCHASE_CONFIRMATION", 5)])
        labels = result["next-purchase-labels"]
        self.assertEqual([r["label_state"] for r in labels], ["blocked_by_hold", "observed", "right_censored"])
        self.assertEqual(labels[0]["target_quantity"], "")
        self.assertNotEqual(labels[0]["episode_id"], labels[1]["episode_id"])
        self.assertEqual(report["row_dispositions"]["hold_same_day"], 1)
        self.assertEqual(result["daily-events"][1]["quantity"], "")

    def test_cancel_is_not_barrier_and_tail_is_not_zero(self):
        _, result = self.run_case([
            ("2026-05-01", "A", "PURCHASE_CONFIRMATION", 2),
            ("2026-05-05", "A", "CANCEL_COMPLETE", 10),
            ("2026-05-10", "A", "PURCHASE_CONFIRMATION", 4)])
        self.assertEqual(len(result["observed-pairs"]), 1)
        self.assertEqual(result["next-purchase-labels"][-1]["target_quantity"], "")
        self.assertEqual(result["daily-events"][1]["event_state"], "cancelled_only")

    def test_no_completed_purchases_still_writes_empty_label_headers(self):
        report, result = self.run_case([("2026-05-01", "A", "SHIPPING", 2)])
        self.assertEqual(report["label_states"], {})
        self.assertEqual(result["next-purchase-labels"], [])

    def test_tamper_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy = self.prepare(root, [("2026-05-01", "A", "PURCHASE_CONFIRMATION", 2)])
            with (policy / "rows.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "hash/version"):
                build(policy / "rows.csv", policy / "report.json", root / "events")
            self.assertFalse((root / "events").exists())

    def test_korean_date_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            policy = self.prepare(root, [("2026-05-01", "A", "PURCHASE_CONFIRMATION", 2)])
            source = policy / "rows.csv"
            source.write_text(source.read_text().replace("2026-05-01 01:00:00 UTC", "2026-05-01 23:00:00 UTC"))
            from scripts.build_purchase_events import digest
            report = json.loads((policy / "report.json").read_text())
            report["output_sha256"] = digest(source)
            (policy / "report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "Korean order date"):
                build(source, policy / "report.json", root / "events")
