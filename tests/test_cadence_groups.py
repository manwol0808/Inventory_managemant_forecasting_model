import json
import unittest
from pathlib import Path

from scripts.run_cadence_groups import classify, groups_from_events, synthetic_data


class CadenceGroupsTests(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads(Path('config/cadence-groups-v1.json').read_text())

    def test_customer_groups_share_across_products_without_same_day_double_count(self):
        def event(i, day, product='a', state='completed'):
            return dict(event_id=i, ordered_on=day, product_id=product, customer_id='c', event_state=state)
        es = [event('1', '2026-01-01'), event('2', '2026-01-08', 'b'),
              event('3', '2026-01-15'), event('4', '2026-01-22'), event('5', '2026-01-22', 'b')]
        before = groups_from_events(es, self.cfg)
        self.assertEqual(before['3'], 'insufficient_history')
        self.assertEqual(before['4'], 'within_14d')
        self.assertEqual(before['4'], before['5'])
        after = groups_from_events(es + [event('6', '2026-02-01', state='held'), event('7', '2026-02-10')], self.cfg)
        self.assertEqual({k: after[k] for k in before}, before)
        self.assertEqual(after['7'], 'insufficient_history')

    def test_boundaries_and_irregularity_are_only_past_inputs(self):
        self.assertEqual(classify([30, 30, 30], self.cfg), 'within_30d')
        self.assertEqual(classify([75, 75, 75], self.cfg), 'over_60d')
        self.assertEqual(classify([2, 2, 60], self.cfg), 'irregular')
        self.assertEqual(classify([7, 7], self.cfg), 'insufficient_history')

    def test_synthetic_customer_cadence_does_not_replace_product_cadence(self):
        cfg = {**self.cfg, 'synthetic_scenarios': ['weekly_shop_monthly_item'], 'synthetic_customers_per_scenario': 2}
        features, targets, events = synthetic_data(cfg, 'monthly_product', 'weekly_product')
        groups = groups_from_events(events, cfg)
        self.assertEqual(len(features), 2)
        for r in features:
            self.assertEqual(groups[r['origin_event_id']], 'within_14d')
            self.assertGreaterEqual(float(r['gap_median_days']), 28)
            self.assertEqual(r['customer_product_count'], 2)
            self.assertEqual(r['target_gap_days'], '')
            self.assertGreaterEqual(targets[r['origin_event_id']]['target_gap_days'], 28)


if __name__ == '__main__':
    unittest.main()
