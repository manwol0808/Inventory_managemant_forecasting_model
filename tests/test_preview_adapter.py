import unittest
from unittest.mock import patch

import numpy as np

from scripts.build_features_and_splits import FEATURE_COLUMNS
from scripts.predict_replenishment import PreviewPredictor


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
        self.assertTrue(a["due"])
        self.assertFalse(a["auto_apply"])
        self.assertEqual(a["suggestion_key"],b["suggestion_key"])

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
