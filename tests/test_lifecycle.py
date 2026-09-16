import unittest
from scripts.audit_item_lifecycle import classify_group


class LifecycleTests(unittest.TestCase):
    def rows(self, *statuses):
        return [{"product_id": "A", "section_status": s} for s in statuses]

    def test_completed_sibling_does_not_hide_return(self):
        self.assertEqual(classify_group(self.rows("PURCHASE_CONFIRMATION", "RETURN_COMPLETE")),
                         "review_return_exchange_linked")

    def test_partial_cancel_differs_from_whole_cancel(self):
        self.assertEqual(classify_group(self.rows("PURCHASE_CONFIRMATION", "CANCEL_COMPLETE")),
                         "review_partial_cancel_linked")
        self.assertEqual(classify_group(self.rows("CANCEL_COMPLETE", "CANCEL_COMPLETE")),
                         "excluded_cancelled_group")

    def test_split_delivery_is_candidate_not_unknown(self):
        self.assertEqual(classify_group(self.rows("PURCHASE_CONFIRMATION", "SHIPPING_COMPLETE")),
                         "completed_group_candidate")

    def test_unknown_state_and_identity_conflict_hold(self):
        self.assertEqual(classify_group(self.rows("FUTURE_STATUS")), "review_unknown_status")
        rows = self.rows("PURCHASE_CONFIRMATION", "PURCHASE_CONFIRMATION")
        rows[1]["product_id"] = "B"
        self.assertEqual(classify_group(rows), "review_product_identity")
