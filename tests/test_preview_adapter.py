import unittest
from datetime import datetime, date, timedelta
from unittest.mock import patch

import numpy as np

from scripts.build_features_and_splits import FEATURE_COLUMNS
from scripts.predict_replenishment import PreviewPredictor, cart_schedule, weekly_cart_schedule, DEFAULT_CART_LEAD_DAYS


class StaticModel:
    def __init__(self, value):
        self.value=value
    def predict(self, dm):
        return np.array([self.value])


class PreviewAdapterTests(unittest.TestCase):
    def fixture(self):
        p=PreviewPredictor.__new__(PreviewPredictor)
        p.report={"split_config":{"start":"2026-04-09","train_end":"2026-07-13"}}
        p.schema={"products":{"p":0}}
        p.model_version="v1"
        p.cart_lead_days=DEFAULT_CART_LEAD_DAYS
        p.cart_policy="lead_days"
        p.weekly_release_hour=22
        p.models={"quantity":StaticModel(2.5),"gap":StaticModel(5)}
        features={k:1 for k in FEATURE_COLUMNS}
        features["product_id"]="p"
        r={"customer_id":"c","product_id":"p","origin_event_id":"e","origin_on":"2026-08-01",
           "feature_asof":"2026-08-02T00:00:00+09:00","history_start":"2026-04-09",
           "latest_purchase_confirmed":True,"has_later_purchase_or_hold":False,
           "as_of":"2026-08-06T00:00:00+09:00","features":features}
        return p,r

    def test_integer_quantity_due_and_model_independent_dedup_key(self):
        p,r=self.fixture()
        with patch("scripts.predict_replenishment.matrix",return_value=None):
            a=p.predict(r)
            p.model_version="v2"
            b=p.predict(r)
        self.assertEqual(a["cart_quantity"],3)
        self.assertEqual(a["reorder_on"],"2026-08-06")
        self.assertEqual(a["cart_on"],"2026-08-02")
        self.assertTrue(a["due"])
        self.assertFalse(a["auto_apply"])
        self.assertEqual(a["suggestion_key"],b["suggestion_key"])

    def test_early_cart_due_at_korean_midnight_before_forecast(self):
        p,r=self.fixture()
        p.models["gap"]=StaticModel(25)
        r["as_of"]="2026-08-15T14:59:59+00:00"
        with patch("scripts.predict_replenishment.matrix",return_value=None):
            before=p.predict(r)
            r["as_of"]="2026-08-15T15:00:00+00:00"
            on=p.predict(r)
            p.cart_lead_days=7
            seven=p.predict(r)
        self.assertEqual(on["reorder_on"],"2026-08-26")
        self.assertEqual(on["cart_on"],"2026-08-16")
        self.assertFalse(before["due"])
        self.assertTrue(on["due"])
        self.assertFalse(seven["due"])
        self.assertEqual(on["suggestion_key"],seven["suggestion_key"])

    def test_cart_schedule_bounds_and_invalid_leads(self):
        self.assertEqual(cart_schedule("2026-09-16","2026-08-20"),"2026-09-06")
        self.assertEqual(cart_schedule("2026-09-16","2026-08-20",7),"2026-09-09")
        self.assertEqual(cart_schedule("2026-09-16","2026-08-20",0),"2026-09-16")
        for lead in (-1,31,True,1.5):
            with self.assertRaises(ValueError):
                cart_schedule("2026-09-16","2026-08-20",lead)

    def test_weekly_sunday_night_boundary_and_policy_independent_key(self):
        p,r=self.fixture()
        p.cart_policy="weekly_order"
        p.models["gap"]=StaticModel(23)
        r.update(origin_on="2026-09-15",feature_asof="2026-09-16T00:00:00+09:00",
                 as_of="2026-09-27T12:59:59+00:00")
        with patch("scripts.predict_replenishment.matrix",return_value=None):
            before=p.predict(r)
            r["as_of"]="2026-09-27T13:00:00+00:00"
            on=p.predict(r)
            p.cart_policy="lead_days"
            fixed=p.predict(r)
        self.assertEqual(on["reorder_on"],"2026-10-08")
        self.assertEqual(on["cart_at"],"2026-09-27T22:00:00+09:00")
        self.assertEqual(on["order_monday"],"2026-09-28")
        self.assertFalse(before["due"])
        self.assertTrue(on["due"])
        self.assertFalse(fixed["due"])
        self.assertEqual(on["suggestion_key"],fixed["suggestion_key"])
        self.assertIsNone(on["cart_lead_days"])

    def test_weekly_all_forecast_weekdays_and_year_boundary(self):
        available=datetime.fromisoformat("2026-01-01T00:00:00+09:00")
        for day in range(7):
            predicted=(date(2026,10,5)+timedelta(days=day)).isoformat()
            planned,actual=weekly_cart_schedule(predicted,available)
            self.assertEqual(planned.isoformat(),"2026-09-27T22:00:00+09:00")
            self.assertEqual(planned,actual)
        planned,_=weekly_cart_schedule("2027-01-01",available)
        self.assertEqual(planned.isoformat(),"2026-12-20T22:00:00+09:00")

    def test_missed_window_prefers_next_order_window_unless_too_late(self):
        available=datetime.fromisoformat("2026-10-01T00:00:00+09:00")
        planned,actual=weekly_cart_schedule("2026-10-08",available)
        self.assertLess(planned,available)
        self.assertEqual(actual.isoformat(),"2026-10-04T22:00:00+09:00")
        _,actual=weekly_cart_schedule("2026-10-02",available)
        self.assertEqual(actual,available)
        monday=datetime.fromisoformat("2026-10-05T10:00:00+09:00")
        self.assertEqual(weekly_cart_schedule("2026-10-08",monday)[1],monday)
        sunday_late=datetime.fromisoformat("2026-10-04T23:00:00+09:00")
        self.assertEqual(weekly_cart_schedule("2026-10-08",sunday_late)[1],sunday_late)
        for hour in (-1,24,True,22.5):
            with self.assertRaises(ValueError):
                weekly_cart_schedule("2026-10-08",available,hour)
        with self.assertRaises(ValueError):
            weekly_cart_schedule("2026-10-08",datetime(2026,10,1))

    def test_rejects_hold_future_features_and_wrong_window(self):
        for field,value in (("has_later_purchase_or_hold",True),("history_start","2024-10-16"),
                            ("as_of","2026-08-01T00:00:00+09:00"),("latest_purchase_confirmed",False)):
            p,r=self.fixture()
            r[field]=value
            with self.assertRaises(ValueError):
                p.predict(r)
        p,r=self.fixture()
        r["features"]["target_quantity"]=5
        with self.assertRaises(ValueError):
            p.predict(r)
