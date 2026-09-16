import unittest

from scripts.prepare_full_history import partition
from scripts.apply_training_policy import validate_product_exclusions
from scripts.train_xgboost import training_weights


class FullHistoryTests(unittest.TestCase):
    def test_weights_respect_label_cutoff_and_half_life(self):
        rows = [{"target_on": "2026-07-13"}, {"target_on": "2026-01-14"}]
        self.assertEqual(list(training_weights(rows, "2026-07-13", 180)), [1.0, 0.5])
        with self.assertRaisesRegex(ValueError, "after training cutoff"):
            training_weights(rows, "2026-07-12", 180)
        with self.assertRaises(ValueError):
            training_weights(rows, "2026-07-13", 0)

    def test_missing_identity_cannot_create_false_adjacency(self):
        rows = [dict(customer_id=c, product_id=p, ordered_on=d) for c,p,d in [
            ("a", "beans", "2025-01-01"), ("a", "", "2025-01-05"),
            ("a", "syrup", "2025-01-05"), ("a", "beans", "2025-01-09"),
            ("b", "beans", "2025-01-01")]]
        kept, quarantine, _ = partition(rows, "2026-07-13")
        self.assertEqual(kept, rows[3:])
        self.assertEqual(len(quarantine), 3)
        self.assertEqual(quarantine[1]["quarantine_reason"], "missing_product_id")
        with self.assertRaisesRegex(ValueError, "causal barrier"):
            partition(rows, "2025-01-04")

    def test_full_scope_policy_reasons(self):
        manifest = {"version": "product-exclusions-v3", "products": {
            "a": "exclude_test_product", "b": "hold_non_catalog_payment"}}
        self.assertEqual(validate_product_exclusions(manifest), manifest["products"])
        manifest["products"]["c"] = "include_completed"
        with self.assertRaises(ValueError):
            validate_product_exclusions(manifest)
