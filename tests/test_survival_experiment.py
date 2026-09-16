import math
import unittest

import numpy as np

from scripts.run_survival_experiment import (aft_log_survival, conditional_probability,
    fit_km, observed_followup, probability_metrics, survival_training_rows)


class SurvivalExperimentTests(unittest.TestCase):
    def test_future_event_is_censored_and_hold_day_is_unknown(self):
        row = dict(origin_event_id="a", origin_on="2026-01-01", label_state="observed", target_on="2026-03-01", target_gap_days="59")
        self.assertEqual(observed_followup(row, "2026-01-31", 60, {}, {}), (30, False))
        self.assertEqual(observed_followup(row, "2026-03-02", 60, {}, {}), (59, True))
        held = {**row, "label_state":"blocked_by_hold"}
        labels, events = {"a":{"blocking_event_id":"h"}}, {"h":{"ordered_on":"2026-01-11"}}
        self.assertEqual(observed_followup(held, "2026-03-02", 60, labels, events), (9, False))
        self.assertEqual(observed_followup(held, "2026-01-05", 60, labels, events), (4, False))

    def test_training_includes_nonreturn_and_excludes_immature_origins(self):
        row = dict(origin_event_id="a", origin_on="2026-01-01", label_state="right_censored")
        selected, bounds = survival_training_rows([row,{**row,"origin_on":"2026-03-01"}],"2026-03-31",90,60,{}, {})
        self.assertEqual(selected,[row])
        self.assertEqual(bounds[0,0],60)
        self.assertTrue(math.isinf(bounds[0,1]))

    def test_aft_probability_formula_and_extreme_stability(self):
        for distribution in ("normal","logistic"):
            mu = np.array([math.log(20)])
            log_s = aft_log_survival(mu,20,distribution,1)
            np.testing.assert_allclose(np.exp(log_s),[.5])
            np.testing.assert_allclose(conditional_probability([0],log_s),[.5])
            p = conditional_probability(aft_log_survival(mu,10,distribution,1),aft_log_survival(mu,20,distribution,1))
            self.assertTrue(0 < p[0] < .5)
            extremes = np.array([-1000,1000])
            prob = conditional_probability(aft_log_survival(extremes,7,distribution,.5),aft_log_survival(extremes,14,distribution,.5))
            self.assertTrue(np.isfinite(prob).all())
            self.assertTrue(((prob >= 0) & (prob <= 1)).all())

    def test_km_risk_set_counts_censors_and_events_correctly(self):
        curve = fit_km(np.array([[2,2],[2,np.inf],[4,4],[6,np.inf]]),6)
        self.assertEqual(curve,[1,1,.75,.75,.375,.375,.375])

    def test_probability_metrics_use_customer_then_origin_weighting(self):
        origins = [dict(outcome="observed",probability_60d=.5)]
        records = [dict(origin_event_id="1",customer_id="a",probability=0,event_next_7d=0) for _ in range(8)]
        records += [dict(origin_event_id="2",customer_id="b",probability=1,event_next_7d=0)]
        result = probability_metrics(origins,records,[.5])
        self.assertEqual(result["landmark_customer_macro_brier"],.5)
        self.assertAlmostEqual(result["landmark_brier"],1/9)
        self.assertEqual(result["thresholds"]["0.5"]["fp"],1)


if __name__ == "__main__":
    unittest.main()
