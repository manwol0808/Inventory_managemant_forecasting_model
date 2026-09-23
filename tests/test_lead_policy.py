import unittest

from scripts.router_champion import extra_lead_days


class LeadPolicyTest(unittest.TestCase):
    def test_fixed_never_moves(self):
        for gap in (1, 7, 30, 60):
            self.assertEqual(extra_lead_days(gap, "fixed"), 0)

    def test_proportional_is_clamped_between_7_and_14_days(self):
        self.assertEqual(extra_lead_days(10, "proportional"), 0)   # 3일 → 하한 7일
        self.assertEqual(extra_lead_days(24, "proportional"), 0)   # 7.2 → 7일
        self.assertEqual(extra_lead_days(30, "proportional"), 2)   # 9일
        self.assertEqual(extra_lead_days(47, "proportional"), 7)   # 14.1 → 상한 14일
        self.assertEqual(extra_lead_days(120, "proportional"), 7)

    def test_unknown_policy_rejected(self):
        with self.assertRaises(ValueError):
            extra_lead_days(30, "linear")


if __name__ == "__main__":
    unittest.main()


from scripts.router_champion import short_history_gap  # noqa: E402


class ShortHistoryRuleTest(unittest.TestCase):
    def test_defaults_leave_existing_flow_alone(self):
        row = {"item_gap_median": "21.4", "pair_gap_lag1": "9"}
        self.assertIsNone(short_history_gap(row, "first", 23, "skip", "model"))
        self.assertIsNone(short_history_gap(row, "second", 23, "skip", "model"))
        self.assertIsNone(short_history_gap(row, "cadence", 23, "item_median", "last_gap"))

    def test_first_purchase_uses_item_median_or_model(self):
        self.assertEqual(short_history_gap({"item_gap_median": "21.4"}, "first", 23, "item_median", "model"),
                         (21, True, "first_purchase_item_median"))
        self.assertEqual(short_history_gap({"item_gap_median": ""}, "first", 23, "item_median", "model"),
                         (23, True, "first_purchase_model_fallback"))

    def test_second_purchase_uses_last_gap(self):
        self.assertEqual(short_history_gap({"pair_gap_lag1": "9"}, "second", 23, "skip", "last_gap"),
                         (9, True, "second_purchase_last_gap"))


from scripts.router_champion import days_before_schedule  # noqa: E402


class DaysBeforeScheduleTest(unittest.TestCase):
    def test_notifies_seven_days_before_predicted_day(self):
        self.assertEqual(days_before_schedule("2026-01-01", 21), ("2026-01-15", "2026-01-15T09:00:00+09:00"))

    def test_short_gap_notifies_next_morning_not_in_the_past(self):
        self.assertEqual(days_before_schedule("2026-01-01", 5)[0], "2026-01-02")
        self.assertEqual(days_before_schedule("2026-01-01", 7)[0], "2026-01-02")
