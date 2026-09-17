import json
import unittest
from pathlib import Path

from scripts.purchase_segment_features import GROUPS
from scripts.run_multi_agent_router import (MODEL_AGENTS, excluded_from_training, guard, monitor_metrics, purchase_rule, stable_routes,
                                            weekly_cart, weekly_metrics)


CFG = json.loads(Path("config/multi-agent-router-v1.json").read_text())


def origin(eid, day, pair_group, customer="c", product="p", customer_group="irregular"):
    return {"origin_event_id": eid, "origin_on": day, "customer_id": customer, "product_id": product,
            "pair_group": pair_group, "customer_group": customer_group}


class RouterTests(unittest.TestCase):
    def test_config_routes_every_type_to_known_agent(self):
        self.assertEqual(set(CFG["routes"]), set(GROUPS))
        self.assertLessEqual(set(CFG["routes"].values()), set(MODEL_AGENTS) | {"champion"})

    def test_hysteresis_needs_two_of_last_three_and_insufficient_is_immediate(self):
        raws = ["stable", "stable", "irregular", "stable", "irregular", "irregular", "insufficient", "speeding_up"]
        rows = [origin(str(i), f"2026-01-{i+1:02d}", raw) for i, raw in enumerate(raws)]
        routes = stable_routes(rows, CFG)
        self.assertEqual([routes[str(i)]["customer_group"] for i in range(len(raws))],
                         ["stable", "stable", "stable", "stable", "irregular", "irregular", "insufficient", "speeding_up"])
        self.assertEqual([i for i in range(len(raws)) if routes[str(i)]["route_changed"]], [4, 6, 7])
        self.assertEqual(routes["2"]["route_confidence"], round(2/3, 3))
        # A later observation cannot change an earlier route.
        future = stable_routes(rows + [origin("x", "2026-02-01", "rebound")], CFG)
        self.assertEqual(routes, {k: future[k] for k in routes})

    def test_pairs_are_routed_independently_and_customer_unit_uses_one_type_per_day(self):
        rows = [origin("a", "2026-01-01", "stable"), origin("b", "2026-01-01", "irregular", product="q"),
                origin("c", "2026-01-02", "irregular"), origin("d", "2026-01-03", "irregular")]
        routes = stable_routes(rows, CFG)
        self.assertEqual(routes["b"]["customer_group"], "irregular")
        self.assertEqual(routes["c"]["customer_group"], "stable")
        self.assertEqual(routes["d"]["customer_group"], "irregular")
        by_customer = stable_routes(rows, {**CFG, "route_unit": "customer"})
        self.assertEqual({v["raw_group"] for v in by_customer.values()}, {"irregular"})
        with self.assertRaises(ValueError):
            stable_routes([origin("e", "2026-01-01", "stable", customer_group="stable"),
                           origin("f", "2026-01-01", "stable", product="q")], {**CFG, "route_unit": "customer"})

    def test_guard_prefers_champion_when_missing_uncertain_or_guarded_later(self):
        self.assertEqual(guard(None, 20, None, CFG, "why"), (20, "why"))
        self.assertEqual(guard(10, 20, CFG["max_relative_iqr"] + .1, CFG, None), (20, "high_uncertainty"))
        self.assertEqual(guard(25, 20, 0.1, {**CFG, "late_guard": True}, None), (20, "later_than_champion"))
        self.assertEqual(guard(25, 20, 0.1, {**CFG, "late_guard": False}, None), (25, None))
        self.assertEqual(guard(15, 20, 0.1, CFG, None), (15, None))

    def test_thursday_three_weeks_out_lands_in_cart_two_weeks_out_monday(self):
        # Monday 2026-09-14 + 24 days = Thursday 2026-10-08.
        self.assertEqual(weekly_cart("2026-09-14", 24, 22),
                         ("2026-10-08", "2026-09-28", "2026-09-27T22:00:00+09:00"))

    def test_weekly_buyers_wait_for_sunday_night_instead_of_mid_week(self):
        # Wednesday purchase, next one expected Sunday: default policy catches up Thursday 00:00.
        self.assertEqual(weekly_cart("2026-09-16", 4, 22)[2], "2026-09-17T00:00:00+09:00")
        self.assertEqual(weekly_cart("2026-09-16", 4, 22, 7), ("2026-09-20", "2026-09-21", "2026-09-20T22:00:00+09:00"))
        # Saturday purchase: Sunday availability releases the same night.
        self.assertEqual(weekly_cart("2026-09-19", 7, 22, 7)[2], "2026-09-20T22:00:00+09:00")
        # Sunday purchase: Monday availability is already the ordering window.
        self.assertEqual(weekly_cart("2026-09-20", 7, 22, 7)[1:], ("2026-09-21", "2026-09-21T00:00:00+09:00"))
        # Longer gaps keep the prior-week Monday rule.
        self.assertEqual(weekly_cart("2026-09-14", 24, 22, 7)[1], "2026-09-28")

    def test_weekly_metrics_count_same_day_as_not_early(self):
        rows = [{"customer_id": "a", "target_on": "2026-09-28", "cart_at": "2026-09-27T22:00:00+09:00", "predicted_gap_days": 7},
                {"customer_id": "a", "target_on": "2026-09-27", "cart_at": "2026-09-27T22:00:00+09:00", "predicted_gap_days": 20},
                {"customer_id": "b", "target_on": "2026-09-20", "cart_at": "2026-09-27T22:00:00+09:00", "predicted_gap_days": 20}]
        result = weekly_metrics(rows, 7)
        self.assertEqual((result["late"], result["same_day"], result["early_1_to_10"]), (1, 1, 1))
        self.assertEqual((result["sunday_or_monday"], result["weekly_buyer"]), (3, 1))
        self.assertAlmostEqual(result["customer_macro_late_rate"], 0.5)
        absent = [{"customer_id": "c", "outcome": "no_repurchase_within_horizon", "origin_on": "2026-06-01",
                   "cart_at": f"{day}T22:00:00+09:00", "predicted_gap_days": 20} for day in ("2026-06-14", "2026-09-06")]
        result = weekly_metrics(rows + absent, 7)
        self.assertEqual((result["n"], result["nonreturn_n"], result["nonreturn_cart_within90_rate"]), (3, 2, 0.5))

    def test_monitor_window_scores_only_repurchases_inside_it(self):
        rows = [{"outcome": "observed", "target_gap_days": "10", "predicted_gap_days": 14},
                {"outcome": "observed", "target_gap_days": "44", "predicted_gap_days": 30},
                {"outcome": "observed", "target_gap_days": "60", "predicted_gap_days": 58},
                {"outcome": "no_repurchase_within_horizon", "target_gap_days": "", "predicted_gap_days": 20},
                {"outcome": "unknown_due_to_hold", "target_gap_days": "", "predicted_gap_days": 20}]
        result = monitor_metrics(rows, 45)
        self.assertEqual((result["origins"], result["repurchased"], result["hits"]), (4, 2, 1))
        self.assertEqual((result["hit_rate"], result["hit_rate_all_origins"], result["closed_without_purchase_rate"]), (0.5, 0.25, 0.5))

    def test_first_purchase_skipped_and_optional_second_purchase_gap(self):
        self.assertEqual(purchase_rule({"pair_count": "0", "pair_gap_lag1": ""}, "insufficient", CFG), ("first", False, None))
        on = {**CFG, "second_purchase_last_gap": True}
        self.assertEqual(purchase_rule({"pair_count": "1", "pair_gap_lag1": "12"}, "insufficient", on), ("second", True, 12))
        self.assertEqual(purchase_rule({"pair_count": "3", "pair_gap_lag1": "9"}, "insufficient", CFG), ("third_fourth", True, None))
        self.assertEqual(purchase_rule({"pair_count": "5", "pair_gap_lag1": "9"}, "stable", CFG), ("cadence", True, None))
        off = {**CFG, "skip_first_purchase": False, "second_purchase_last_gap": False}
        self.assertEqual(purchase_rule({"pair_count": "0", "pair_gap_lag1": ""}, "insufficient", off), ("first", True, None))
        self.assertEqual(purchase_rule({"pair_count": "1", "pair_gap_lag1": "12"}, "insufficient", off), ("second", True, None))

    def test_exceptional_period_removes_rows_touching_it(self):
        cfg = {"training_exclude": {"start": "2026-01-01", "end": "2026-02-28"}}
        self.assertTrue(excluded_from_training({"origin_on": "2026-01-10", "target_on": "2026-03-10"}, cfg))
        self.assertTrue(excluded_from_training({"origin_on": "2025-12-20", "target_on": "2026-01-05"}, cfg))
        self.assertFalse(excluded_from_training({"origin_on": "2025-12-01", "target_on": "2025-12-20"}, cfg))
        self.assertFalse(excluded_from_training({"origin_on": "2026-03-01", "target_on": ""}, cfg))
        self.assertFalse(excluded_from_training({"origin_on": "2026-01-10", "target_on": "2026-01-20"}, {}))


if __name__ == "__main__":
    unittest.main()
