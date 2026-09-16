import unittest

from scripts.run_timing_experiment import horizon_outcome, recent_gaps, summarize, training_rows


class TimingExperimentTests(unittest.TestCase):
    def test_training_requires_equal_followup_and_past_target(self):
        old = dict(origin_on="2026-01-15", label_state="observed", target_on="2026-02-01", target_gap_days="17")
        fast_recent = dict(origin_on="2026-03-01", label_state="observed", target_on="2026-03-02", target_gap_days="1")
        self.assertEqual(training_rows([old, fast_recent], "2026-03-31", 90, 60), [old])
        with self.assertRaises(ValueError):
            training_rows([{**old, "target_on": "2026-04-01"}], "2026-03-31", 90, 60)

    def test_hold_and_horizon_boundaries(self):
        row = dict(origin_event_id="a", origin_on="2026-04-01", label_state="observed", target_on="2026-05-31")
        self.assertEqual(horizon_outcome(row, 60, "2026-06-01", {}, {}), "observed")
        self.assertEqual(horizon_outcome(row, 60, "2026-05-30", {}, {}), "not_mature")
        self.assertEqual(horizon_outcome({**row, "target_on": "2026-06-01"}, 60, "2026-06-01", {}, {}), "no_repurchase_within_horizon")
        held = {**row, "label_state": "blocked_by_hold"}
        labels = {"a": {"blocking_event_id": "b"}}
        self.assertEqual(horizon_outcome(held, 60, "2026-06-01", labels, {"b": {"ordered_on": "2026-05-31"}}), "unknown_due_to_hold")
        self.assertEqual(horizon_outcome(held, 60, "2026-06-01", labels, {"b": {"ordered_on": "2026-06-01"}}), "no_repurchase_within_horizon")

    def test_recent_gaps_future_invariance_and_hold_reset(self):
        def event(i, day, state="completed"):
            return dict(event_id=i, ordered_on=day, event_state=state, customer_id="c", product_id="p")
        events = {e["event_id"]: e for e in [event("1", "2026-01-01"), event("2", "2026-01-03"), event("3", "2026-01-07"), event("4", "2026-01-15")]}
        before = recent_gaps(events)
        self.assertEqual(before["4"], 4)
        events.update({"5": event("5", "2026-01-16", "held"), "6": event("6", "2026-01-20"), "7": event("7", "2026-01-23")})
        after = recent_gaps(events)
        self.assertEqual({k: after[k] for k in before}, before)
        self.assertIsNone(after["6"])
        self.assertEqual(after["7"], 3)

    def test_customer_equal_weighting_quantity_tolerance_and_nonreturn(self):
        row = dict(outcome="observed", customer_id="a", predicted_gap_days="10", target_gap_days="10", cart_quantity="2", target_quantity="1")
        rows = [row.copy() for _ in range(9)] + [{**row, "customer_id": "b", "predicted_gap_days": "20", "cart_quantity": "4"}]
        rows += [{**row, "outcome": "no_repurchase_within_horizon", "target_gap_days": "", "target_quantity": ""}]
        result = summarize(rows, 60)
        self.assertEqual(result["date_mae"], 1)
        self.assertEqual(result["customer_macro_date_mae"], 5)
        self.assertEqual(result["quantity_within_1_rate"], .9)
        self.assertEqual(result["quantity_within_2_rate"], .9)
        self.assertEqual(result["no_repurchase_but_predicted_within_horizon_rate"], 1)
        self.assertEqual(result["n"], 10)


if __name__ == "__main__":
    unittest.main()
