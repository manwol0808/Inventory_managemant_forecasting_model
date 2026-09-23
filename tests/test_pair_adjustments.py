import tempfile
import unittest
from pathlib import Path

from scripts.pair_adjustments import PairAdjustments


class PairAdjustmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = PairAdjustments(Path(self.tmp.name) / "adjustments.sqlite")
        self.addCleanup(self.store.close)

    def event(self, n, predicted, actual, customer="A", product="P"):
        return {"origin_event_id": f"{customer}-{product}-{n}", "customer_id": customer,
                "product_id": product, "origin_on": f"2026-01-{n:02d}",
                "predicted_gap_days": predicted, "target_gap_days": actual}

    def test_no_history_means_no_adjustment(self):
        self.assertEqual(self.store.adjustment("A", "P"), 0)

    def test_pulls_forward_when_the_model_predicts_too_long(self):
        self.store.record([self.event(1, 25, 18), self.event(2, 24, 17), self.event(3, 26, 19)])
        self.assertEqual(self.store.adjustment("A", "P"), -7)

    def test_never_delays_when_the_model_predicts_too_short(self):
        self.store.record([self.event(1, 10, 30), self.event(2, 10, 31), self.event(3, 10, 29)])
        self.assertEqual(self.store.adjustment("A", "P"), 0)

    def test_respects_the_cap(self):
        store = PairAdjustments(Path(self.tmp.name) / "capped.sqlite", cap_days=3)
        self.addCleanup(store.close)
        store.record([self.event(n, 40, 10) for n in (1, 2, 3)])
        self.assertEqual(store.adjustment("A", "P"), -3)

    def test_recording_the_same_event_twice_does_not_duplicate_history(self):
        self.store.record([self.event(1, 25, 18)])
        before = self.store.stats()["errors"]
        self.store.record([self.event(1, 25, 18)])
        self.assertEqual(self.store.stats()["errors"], before)

    def test_only_uses_observations_before_a_given_date(self):
        self.store.record([self.event(5, 25, 18), self.event(6, 24, 17), self.event(7, 26, 19)])
        self.assertEqual(self.store.adjustment("A", "P", before_on="2026-01-05"), 0)
        self.assertEqual(self.store.adjustment("A", "P", before_on="2026-01-31"), -7)

    def test_skips_events_without_an_observed_repurchase(self):
        rows = [{"origin_event_id": "x", "customer_id": "C", "product_id": "P", "origin_on": "2026-01-01",
                 "predicted_gap_days": 10, "target_gap_days": ""},
                {"origin_event_id": "y", "customer_id": "C", "product_id": "P", "origin_on": "2026-01-01",
                 "predicted_gap_days": 10, "target_gap_days": "0"}]
        self.assertEqual(self.store.record(rows), 0)

    def test_pairs_are_independent(self):
        self.store.record([self.event(n, 25, 18) for n in (1, 2, 3)])
        self.store.record([self.event(n, 10, 30, customer="B") for n in (1, 2, 3)])
        self.assertEqual(self.store.adjustment("A", "P"), -7)
        self.assertEqual(self.store.adjustment("B", "P"), 0)
        self.assertEqual(self.store.all_adjustments(), {("A", "P"): -7})

    def test_rejects_a_negative_cap(self):
        with self.assertRaises(ValueError):
            PairAdjustments(Path(self.tmp.name) / "bad.sqlite", cap_days=-1)


if __name__ == "__main__":
    unittest.main()


class RatchetTest(unittest.TestCase):
    def ev(self, n, predicted, actual, cart_gap, customer="A", product="P"):
        origin = f"2026-01-{n:02d}"
        return {"origin_event_id": f"{customer}{product}{n}", "customer_id": customer, "product_id": product,
                "origin_on": origin, "predicted_gap_days": predicted, "target_gap_days": actual,
                "cart_on": f"2026-01-{n + cart_gap:02d}"}

    def test_pulls_by_the_shortfall_and_never_returns(self):
        import tempfile
        from pathlib import Path
        from scripts.pair_adjustments import PairAdjustments
        with tempfile.TemporaryDirectory() as tmp:
            with PairAdjustments(Path(tmp) / "t.sqlite", cap_days=21, mode="ratchet") as store:
                store.record([self.ev(1, 21, 10, 14)])          # 알림 14일째, 10일째 삼 → 부족 4 + 1
                self.assertEqual(store.adjustment("A", "P"), -5)
                store.record([self.ev(2, 21, 20, 9)])           # 알림 9일째, 20일째 삼 → 받았음, 그대로
                self.assertEqual(store.adjustment("A", "P"), -5)
                store.record([self.ev(3, 21, 3, 9)])            # 또 먼저 삼 → 6 + 1 더
                self.assertEqual(store.adjustment("A", "P"), -12)
                store.record([self.ev(4, 40, 1, 8), self.ev(5, 40, 1, 8)])   # 8+1씩: -20, 한도 -21
                self.assertEqual(store.adjustment("A", "P"), -21)   # 한도

    def test_median_mode_ignores_alarm_column(self):
        import tempfile
        from pathlib import Path
        from scripts.pair_adjustments import PairAdjustments
        with tempfile.TemporaryDirectory() as tmp:
            with PairAdjustments(Path(tmp) / "t.sqlite") as store:
                store.record([self.ev(1, 21, 10, 14)])
                self.assertEqual(store.adjustment("A", "P"), -7)
