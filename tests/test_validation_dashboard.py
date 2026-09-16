import unittest

from scripts.build_validation_dashboard import aggregate, csv_content


class ValidationDashboardTests(unittest.TestCase):
    def test_actual_dates_weighted_errors_and_missing_days(self):
        rows = [{"origin_event_id": "a", "origin_on": "2026-07-14", "target_quantity": 4, "predicted_quantity": 2,
                 "target_on": "2026-07-20", "predicted_on": "2026-07-18"},
                {"origin_event_id": "b", "origin_on": "2026-07-14", "target_quantity": 2, "predicted_quantity": 4,
                 "target_on": "2026-07-21", "predicted_on": "2026-07-24"}]
        result = aggregate(rows, "2026-07-14", "2026-07-15")
        self.assertEqual(result[0]["actual_mean"], result[0]["predicted_mean"])
        self.assertEqual(result[0]["quantity_mae"], 2)
        self.assertEqual(result[0]["quantity_rmse"], 2)
        self.assertEqual(result[0]["date_mae"], 2.5)
        self.assertEqual(result[1]["n"], 0)
        self.assertIsNone(result[1]["quantity_mae"])
        csv = csv_content(result, [("quantity_mae", "MAE")])
        self.assertIn("2026-07-15T00:00:00+09:00,\r\n", csv)
        with self.assertRaises(ValueError):
            aggregate(rows, "2026-07-15", "2026-07-16")
        with self.assertRaises(ValueError):
            aggregate(rows + rows, "2026-07-14", "2026-07-15")
