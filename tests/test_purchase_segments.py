import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from scripts.build_features_and_splits import FEATURE_COLUMNS
from scripts.purchase_segment_features import feature_names, profile, segment_features
from scripts.run_purchase_segments import SegmentPredictor, design_matrix, metrics, scheduled_gap, train_candidates


CFG=json.loads(Path("config/purchase-segments-v1.json").read_text())


def event(eid, day, customer="c", product="p", state="completed", quantity="2"):
    return {"event_id":eid,"ordered_on":day,"customer_id":customer,"product_id":product,
            "event_state":state,"quantity":quantity}


class SegmentTests(unittest.TestCase):
    def test_patterns_use_order_not_just_mean(self):
        for gaps,expected in (([7,7,8,7,7,8],"stable"),([30,25,20,15,10,5],"speeding_up"),
                              ([5,10,15,20,25,30],"slowing_down"),([30,20,10,5,15,25],"rebound"),
                              ([2,60,2,60,2,60],"irregular"),([7,7,7],"insufficient")):
            self.assertEqual(profile(gaps,CFG)["group"],expected)

    def test_future_events_cannot_change_past_and_day_order_is_stable(self):
        events=[event("a","2026-01-01"),event("b","2026-01-05"),event("c","2026-01-05","other")]
        past=segment_features(events,CFG)
        self.assertEqual(past,segment_features(list(reversed(events)),CFG))
        future=segment_features(events+[event("d","2026-02-01",quantity="999")],CFG)
        self.assertEqual(past,{key:future[key] for key in past})

    def test_cold_item_requires_both_interval_and_customer_support(self):
        events=[]
        # Four intervals per customer. Four customers cannot activate speed.
        for c in range(5):
            for purchase in range(5):
                day=date(2026,1,1)+timedelta(days=7*purchase+c*35)
                events.append(event(f"{c}_{purchase}",day.isoformat(),str(c)))
        profiles=segment_features(events,CFG)
        self.assertEqual(profiles["3_4"]["item_gap_count"],16)
        self.assertEqual(profiles["3_4"]["item_gap_median"],"")
        self.assertEqual(profiles["4_3"]["item_speed_available"],0)
        self.assertEqual(profiles["4_4"]["item_speed_available"],1)
        self.assertEqual(profiles["4_4"]["item_gap_median"],7)
        self.assertEqual(profiles["4_4"]["item_days_per_previous_unit"],3.5)
        one=[event(str(i),(date(2026,1,1)+timedelta(days=i)).isoformat()) for i in range(25)]
        self.assertEqual(segment_features(one,CFG)["24"]["item_speed_available"],0)

    def test_hold_resets_and_customer_visits_do_not_replace_pair_gaps(self):
        events=[event(str(i),(date(2026,1,1)+timedelta(days=7*i)).isoformat(),product="weekly") for i in range(6)]
        events += [event("p0","2026-01-01"),event("p1","2026-02-05")]
        result=segment_features(events,CFG)["p1"]
        self.assertEqual(result["customer_last"],7)
        self.assertEqual(result["pair_last"],35)
        held=events+[event("h","2026-02-06",state="held"),event("p2","2026-02-10")]
        result=segment_features(held,CFG)["p2"]
        self.assertEqual(result["pair_group"],"insufficient")
        self.assertEqual(result["pair_count"],0)
        self.assertEqual(result["customer_count"],0)

    def test_metrics_distinguish_same_day_and_report_nonreturn(self):
        rows=[{"customer_id":"c","outcome":"observed","predicted_gap_days":p,
               "target_gap_days":10,"cart_quantity":2,"target_quantity":2,"route":"global"} for p in (9,10,11)]
        rows += [{"customer_id":"d","outcome":"no_repurchase_within_horizon", "predicted_gap_days":12,"route":"global"}]
        stats=metrics(rows,CFG)["cart_policies"]["none"]
        self.assertEqual((stats["early"],stats["same_day"],stats["late"]),(1,1,1))
        self.assertEqual(stats["nonreturn_n"],1)
        self.assertEqual(scheduled_gap(5,"lead10"),1)
        self.assertEqual(scheduled_gap(40,"relative25_cap7"),33)

    def test_saved_routing_fallback_and_label_independence(self):
        cfg={**CFG,"rounds":2,"minimum_specialist_rows":4,"minimum_specialist_customers":2,
             "candidates":[{"id":"specialists_q25","features":"all","alpha":.25,"specialists":True}]}
        rows=[]
        for i in range(12):
            r={k:1 for k in feature_names("all")}
            r.update(product_id="p",customer_id=str(i%3),origin_event_id=str(i),origin_on="2025-01-01",
                     target_on="2025-01-10",target_gap_days=i+2,target_quantity=2,
                     customer_group="stable" if i<9 else "irregular")
            rows.append(r)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"models"
            values,_=train_candidates(rows,rows,cfg,path)
            predictor=SegmentPredictor(path/"specialists_q25")
            self.assertEqual(values["specialists_q25"][0]["route"],"stable")
            self.assertEqual(values["specialists_q25"][-1]["route"],"global")
            contaminated=[{**r,"target_gap_days":9999,"target_quantity":9999} for r in rows]
            self.assertEqual(predictor.predict(contaminated),values["specialists_q25"])
            unseen={**rows[0],"product_id":"new","customer_group":"insufficient"}
            self.assertEqual(predictor.predict([unseen])[0]["route"],"global")
            self.assertNotIn("target_gap_days",design_matrix(rows,predictor.models[next(iter(predictor.models))][0]).feature_names)


if __name__ == "__main__":
    unittest.main()
