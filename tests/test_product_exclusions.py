import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.apply_training_policy import apply_policy
from scripts.build_purchase_events import build


class ProductExclusionTests(unittest.TestCase):
    def test_samples_gifts_do_not_remove_customer_or_promotion_goods(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for day in ["2026-05-01", "2026-05-10"]:
                for product, name in [("sample", "원두 샘플"), ("gift", "[사은품] 행주"),
                                      ("goods", "[사은품증정] 정상 판매 상품")]:
                    rows.append({"customer_id": "001", "order_id": day, "product_id": product,
                                 "product_name": name, "order_item_code": product,
                                 "order_section_item_no": product, "ordered_on": day,
                                 "ordered_at": day + " 01:00:00 UTC", "quantity": "2",
                                 "source_updated_at": "2026-09-16 00:00:00 UTC",
                                 "extracted_at": "2026-09-16 01:00:00 UTC",
                                 "section_status": "PURCHASE_CONFIRMATION", "paid_total": "0"})
            source = root / "source.csv"
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            config = root / "products.json"
            config.write_text(json.dumps({"version": "product-exclusions-v2", "products": {
                "sample": "exclude_sample", "gift": "exclude_gift"}}))
            policy = root / "policy"
            report = apply_policy(source, policy, config)
            self.assertEqual(report["policy_version"], "purchase-policy-v4")
            self.assertEqual(report["row_decisions"], {"exclude_sample": 2, "exclude_gift": 2, "include_completed": 2})
            output = root / "events"
            result = build(policy / "rows.csv", policy / "report.json", output)
            self.assertEqual(result["linked_rows"], 6)
            self.assertEqual(result["event_states"], {"completed": 2, "excluded_only": 4})
            with (output / "observed-pairs.csv").open() as stream:
                pairs = list(csv.DictReader(stream))
            self.assertEqual(len(pairs), 1)
            self.assertEqual((pairs[0]["product_id"], pairs[0]["target_quantity"]), ("goods", "2"))

    def test_credit_and_goods_in_same_order_and_zero_payment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for day in ["2026-05-01", "2026-05-10"]:
                for product in ["236", "beans"]:
                    rows.append({"customer_id": "001", "order_id": day,
                                 "product_id": product, "order_item_code": product,
                                 "order_section_item_no": product, "ordered_on": day,
                                 "ordered_at": day + " 01:00:00 UTC", "quantity": "2",
                                 "source_updated_at": "2026-09-16 00:00:00 UTC",
                                 "extracted_at": "2026-09-16 01:00:00 UTC",
                                 "section_status": "PURCHASE_CONFIRMATION", "paid_total": "0"})
            source = root / "source.csv"
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            config = root / "exclusions.json"
            config.write_text(json.dumps({"version": "product-exclusions-v1", "products": {"236": "exclude_prepaid_credit"}}))
            policy = root / "policy"
            report = apply_policy(source, policy, config)
            self.assertEqual(report["row_decisions"], {"exclude_prepaid_credit": 2, "include_completed": 2})
            with (policy / "rows.csv").open() as stream:
                annotated = list(csv.DictReader(stream))
            self.assertEqual([{k: r[k] for k in rows[0]} for r in annotated], rows)
            output = root / "events"
            result = build(policy / "rows.csv", policy / "report.json", output)
            self.assertEqual(result["included_quantity"], 4)
            self.assertEqual(result["event_states"], {"completed": 2, "excluded_only": 2})
            with (output / "observed-pairs.csv").open() as stream:
                pairs = list(csv.DictReader(stream))
            self.assertEqual(len(pairs), 1)
            self.assertEqual((pairs[0]["product_id"], pairs[0]["target_gap_days"]), ("beans", "9"))
            with (output / "source-links.csv").open() as stream:
                links = list(csv.DictReader(stream))
            self.assertEqual(len(links), 4)
            self.assertTrue(all(r["event_quantity_contribution"] == "0" for r in links if r["order_item_code"] == "236"))
            # A confirmed test account overrides product/state decisions, preserving every row.
            customer_config = root / "customers.json"
            customer_config.write_text(json.dumps({"version": "test-account-exclusions-v1", "customers": {"001": "exclude_internal_test"}}))
            test_policy = root / "test-policy"
            test_report = apply_policy(source, test_policy, config, customer_config)
            self.assertEqual(test_report["row_decisions"], {"exclude_internal_test": 4})
            test_events = root / "test-events"
            test_result = build(test_policy / "rows.csv", test_policy / "report.json", test_events)
            self.assertEqual(test_result["event_states"], {"excluded_only": 4})
            self.assertEqual(test_result["label_states"], {})
            self.assertEqual(test_result["included_quantity"], 0)
            self.assertEqual(test_result["linked_rows"], 4)

    def test_invalid_exclusion_reason_fails_before_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root / "exclusions.json"
            config.write_text(json.dumps({"version": "product-exclusions-v1", "products": {"236": "unknown"}}))
            with self.assertRaisesRegex(ValueError, "Invalid product"):
                apply_policy(root / "source.csv", root / "output", config)
            self.assertFalse((root / "output").exists())
