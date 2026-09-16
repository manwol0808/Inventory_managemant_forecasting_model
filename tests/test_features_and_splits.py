import json
import unittest
from pathlib import Path

from scripts.build_features_and_splits import FEATURE_COLUMNS, feature_rows


CONFIG = json.loads((Path(__file__).resolve().parents[1] / "config/time-split-v1.json").read_text())


def fixture(specs):
    """Hand-authored day events and labels, independent of the production event builder."""
    events, labels, series = [], [], {}
    for day, customer, product, quantity in sorted(specs):
        key = (customer, product)
        sequence = series.setdefault(key, [])
        eid = "|".join((day, customer, product))
        completed = quantity is not None
        previous = sequence[-1] if sequence else None
        episode = previous["episode_id"] if previous and previous["event_state"] == "completed" else eid
        index = int(previous["episode_purchase_index"]) + 1 if previous and previous["event_state"] == "completed" else 1
        event = {"event_id": eid, "customer_id": customer, "product_id": product,
                 "ordered_on": day, "event_state": "completed" if completed else "held",
                 "quantity": str(quantity) if completed else "",
                 "snapshot_observed_at": "2026-09-16T00:00:00+00:00",
                 "episode_id": episode if completed else "", "episode_purchase_index": str(index) if completed else ""}
        sequence.append(event)
        events.append(event)
    from datetime import date
    for sequence in series.values():
        for i, event in enumerate(sequence):
            if event["event_state"] != "completed":
                continue
            following = sequence[i + 1] if i + 1 < len(sequence) else None
            observed = following is not None and following["event_state"] == "completed"
            labels.append({"origin_event_id": event["event_id"], "customer_id": event["customer_id"],
                           "product_id": event["product_id"], "origin_on": event["ordered_on"],
                           "label_state": "observed" if observed else "blocked_by_hold" if following else "right_censored",
                           "target_event_id": following["event_id"] if observed else "",
                           "target_on": following["ordered_on"] if observed else "",
                           "target_quantity": following["quantity"] if observed else "",
                           "target_gap_days": str((date.fromisoformat(following["ordered_on"]) - date.fromisoformat(event["ordered_on"])).days) if observed else "",
                           "label_snapshot_observed_at": "2026-09-16T00:00:00+00:00" if observed else ""})
    return events, labels


class FeatureSplitTests(unittest.TestCase):
    def rows(self, specs):
        return feature_rows(*fixture(specs), CONFIG)

    def test_full_past_history_not_just_two_rows(self):
        rows = self.rows([("2026-05-01", "A", "beans", 2), ("2026-05-10", "A", "beans", 4),
                          ("2026-05-20", "A", "beans", 6), ("2026-06-01", "A", "beans", 99)])
        row = rows[2]
        self.assertEqual(row["episode_purchase_count"], 3)
        self.assertEqual(row["qty_mean"], 4)
        self.assertEqual(row["qty_last"], 6)
        self.assertEqual(row["gap_median_days"], 9.5)
        self.assertEqual(row["target_quantity"], "99")
        self.assertEqual(rows[0]["gap_last_days"], "")

    def test_append_future_does_not_change_earlier_features(self):
        specs = [("2026-05-01", "A", "beans", 2), ("2026-05-10", "A", "beans", 4)]
        before = self.rows(specs)
        after = self.rows(specs + [("2026-09-01", "B", "beans", 999), ("2026-09-05", "A", "beans", 888)])
        for old, new in zip(before, after):
            self.assertEqual({k: old[k] for k in FEATURE_COLUMNS}, {k: new[k] for k in FEATURE_COLUMNS})

    def test_common_calendar_boundaries_and_purged_targets(self):
        rows = self.rows([("2026-06-20", "A", "beans", 2), ("2026-06-25", "A", "beans", 3),
                          ("2026-07-10", "A", "beans", 4), ("2026-07-20", "A", "beans", 5),
                          ("2026-08-20", "A", "beans", 6), ("2026-09-01", "A", "beans", 7)])
        self.assertEqual(rows[0]["fit_role"], "train_core")
        self.assertEqual(rows[0]["split_outcome"], "crosses_role_boundary")
        self.assertEqual(rows[0]["exploratory_train_pool_eligible"], "true")
        self.assertEqual(rows[1]["fit_role"], "early_stop")
        self.assertEqual(rows[1]["exploratory_role_eligible"], "true")
        self.assertEqual(rows[2]["exploratory_train_pool_eligible"], "false")
        self.assertEqual(rows[3]["split_outcome"], "crosses_role_boundary")
        self.assertEqual(rows[4]["fit_role"], "final_test")
        self.assertEqual(rows[-1]["split_outcome"], "right_censored")

    def test_hold_resets_pair_history_without_merging_products(self):
        rows = self.rows([("2026-05-01", "A", "beans", 2), ("2026-05-05", "A", "syrup", 100),
                          ("2026-05-10", "A", "beans", None), ("2026-05-20", "A", "beans", 3)])
        last = rows[-1]
        self.assertEqual((last["episode_purchase_count"], last["qty_mean"], last["gap_count"]), (1, 3, 0))
        self.assertEqual(last["customer_product_count"], 2)

    def test_same_day_global_features_are_independent_of_input_order(self):
        events, labels = fixture([("2026-05-01", "A", "beans", 2), ("2026-05-01", "B", "beans", 4)])
        forward = feature_rows(events, labels, CONFIG)
        reverse = feature_rows(list(reversed(events)), list(reversed(labels)), CONFIG)
        self.assertEqual(forward, reverse)
        self.assertTrue(all(r["product_mean_quantity"] == 3 for r in forward))

    def test_snapshot_unavailability_and_target_columns_not_features(self):
        rows = self.rows([("2026-05-01", "A", "beans", 2), ("2026-05-10", "A", "beans", 4)])
        self.assertTrue(all(r["feature_timestamp_available"] == "false" for r in rows))
        self.assertTrue(all(r["strict_training_ready"] == "false" for r in rows))
        self.assertFalse(set(FEATURE_COLUMNS) & {"target_quantity", "target_gap_days", "customer_id", "label_state", "outer_split"})

    def test_invalid_label_target_rejected(self):
        events, labels = fixture([("2026-05-01", "A", "beans", 2), ("2026-05-10", "A", "beans", 4)])
        labels[0]["target_quantity"] = "999"
        with self.assertRaisesRegex(ValueError, "Invalid next-purchase"):
            feature_rows(events, labels, CONFIG)
