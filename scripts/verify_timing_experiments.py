"""Recompute saved trial scores with pandas and survival probabilities with scipy.

Does not fit models or change any experimental choice.
"""
import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import logistic, norm

from scripts.build_features_and_splits import read_csv
from scripts.run_baseline import digest, write_json
from scripts.train_xgboost import matrix


def verify():
    raw = {r["origin_event_id"]:r for r in read_csv(Path("data/features-splits-full-v1/all-origins-audit.csv"))}
    events = {r["event_id"]:r for r in read_csv(Path("data/purchase-events-full-v1/daily-events.csv"))}
    labels = {r["origin_event_id"]:r for r in read_csv(Path("data/purchase-events-full-v1/next-purchase-labels.csv"))}
    timing_cfg = json.loads(Path("config/timing-experiment-v2.json").read_text())
    expected = {k:r for k,r in raw.items() if "2026-04-01" <= r["origin_on"] <= "2026-06-30"}
    truth, followup = {}, {}
    for key,row in expected.items():
        origin = date.fromisoformat(row["origin_on"])
        if row["label_state"] == "observed":
            duration = (date.fromisoformat(row["target_on"])-origin).days
            truth[key] = "observed" if duration <= 60 else "no_repurchase_within_horizon"
            followup[key] = (min(duration,60),duration <= 60)
        elif row["label_state"] == "blocked_by_hold":
            duration = (date.fromisoformat(events[labels[key]["blocking_event_id"]]["ordered_on"])-origin).days
            truth[key] = "unknown_due_to_hold" if duration <= 60 else "no_repurchase_within_horizon"
            followup[key] = (min(duration-1,60),False)
        else:
            truth[key] = "no_repurchase_within_horizon"
            followup[key] = (60,False)
    verification = {"expected_origins":len(expected),"outcomes":dict(Counter(truth.values())),"reports":{},"coverage":{}}
    origin_scores = {}
    for run_name in ("timing-experiment-v2","survival-experiment-v1","tabpfn-experiment-v1"):
        root = Path("artifacts")/run_name
        if not (root/"report.json").exists():
            raise FileNotFoundError(root/"report.json")
        report = json.loads((root/"report.json").read_text())
        checked,landmark_count = 0,0
        for candidate,item in report["candidates"].items():
            path = root/(candidate+".csv")
            assert digest(path) == item["predictions_sha256"]
            df = pd.read_csv(path,dtype={"origin_event_id":str,"customer_id":str,"product_id":str})
            assert not df.origin_event_id.duplicated().any() and set(df.origin_event_id) == set(expected)
            for r in df.itertuples():
                actual = expected[r.origin_event_id]
                assert r.outcome == truth[r.origin_event_id]
                assert (r.customer_id,r.product_id,r.origin_on) == (actual["customer_id"],actual["product_id"],actual["origin_on"])
                if r.outcome == "observed":
                    assert r.target_quantity == int(actual["target_quantity"])
                    assert r.target_gap_days == int(actual["target_gap_days"])
            obs = df[df.outcome == "observed"].copy()
            obs["error"] = (obs.predicted_gap_days-obs.target_gap_days).abs()
            m = item.get("overall",item.get("point_metrics"))
            assert np.isclose(obs.error.mean(),m["date_mae"])
            assert np.isclose(obs.groupby("customer_id").error.mean().mean(),m["customer_macro_date_mae"])
            qe = (obs.cart_quantity-obs.target_quantity).abs()
            assert np.isclose((qe <= 1).mean(),m["quantity_within_1_rate"])
            assert np.isclose((qe <= 2).mean(),m["quantity_within_2_rate"])
            checked += len(df)
            if run_name != "survival-experiment-v1":
                continue
            lm_path = root/(candidate+"-landmarks.csv")
            assert digest(lm_path) == item["landmarks_sha256"]
            lm = pd.read_csv(lm_path,dtype={"origin_event_id":str,"customer_id":str})
            expected_lm = {}
            for key,(duration,event) in followup.items():
                for elapsed in report["config"]["landmarks_days"]:
                    if duration > elapsed and (event or duration >= elapsed+7):
                        expected_lm[(key,elapsed)] = int(event and duration <= elapsed+7)
            assert not lm.duplicated(["origin_event_id","elapsed_days"]).any()
            assert set(zip(lm.origin_event_id,lm.elapsed_days)) == set(expected_lm)
            for r in lm.itertuples():
                assert r.event_next_7d == expected_lm[(r.origin_event_id,r.elapsed_days)]
                assert r.customer_id == expected[r.origin_event_id]["customer_id"]
            lm["sq"] = (lm.probability-lm.event_next_7d)**2
            customer_scores = lm.groupby(["customer_id","origin_event_id"]).sq.mean().groupby("customer_id").mean()
            assert np.isclose(customer_scores.mean(),item["probability_metrics"]["landmark_customer_macro_brier"])
            origin_scores[candidate] = customer_scores
            window,variant = candidate.split("_",1)
            for fold in timing_cfg["folds"]:
                f = df[df.fold == fold["id"]]
                work = root/"models"/fold["id"]/window[1:]
                if variant == "km":
                    s = np.array(json.loads((work/"km.json").read_text()))
                    assert (s[report["config"]["landmarks_days"]] > 0).all()
                    probability60 = np.full(len(f),1-s[60])
                else:
                    spec = next(v for v in report["config"]["variants"] if v["id"] == variant)
                    schema = json.loads((work/"schema.json").read_text())
                    model = xgb.Booster({"nthread":4})
                    model.load_model(work/(variant+".json"))
                    mu = model.predict(matrix([raw[key] for key in f.origin_event_id],schema),output_margin=True).astype(float)
                    distribution = norm if spec["distribution"] == "normal" else logistic
                    probability60 = distribution.cdf((np.log(60)-mu)/spec["scale"])
                np.testing.assert_allclose(f.probability_60d,probability60,rtol=1e-10,atol=1e-12)
                for t in report["config"]["landmarks_days"]:
                    if variant == "km":
                        probs = np.full(len(f),1-s[t+7]/s[t])
                    else:
                        start = distribution.logsf((np.log(t)-mu)/spec["scale"]) if t else np.zeros(len(f))
                        stop = distribution.logsf((np.log(t+7)-mu)/spec["scale"])
                        probs = -np.expm1(stop-start)
                    mapping = dict(zip(f.origin_event_id,probs))
                    part = lm[(lm.fold == fold["id"]) & (lm.elapsed_days == t)]
                    np.testing.assert_allclose(part.probability,[mapping[key] for key in part.origin_event_id],rtol=1e-9,atol=1e-12)
            landmark_count += len(lm)
        verification["reports"][run_name] = {"sha256":digest(root/"report.json"),"candidates":len(report["candidates"]),
            "prediction_rows_verified":checked,"landmark_rows_verified":landmark_count}
    # Coverage is independent of any candidate's probabilities.
    for fold in timing_cfg["folds"]:
        coverage = []
        for elapsed in range(0,50,7):
            counts = Counter()
            for key,row in expected.items():
                if not fold["origin_start"] <= row["origin_on"] <= fold["origin_end"]:
                    continue
                duration,event = followup[key]
                if event and duration <= elapsed:
                    counts["already_repurchased"] += 1
                elif not event and duration < elapsed+7:
                    counts["unknown_due_to_hold"] += 1
                else:
                    counts["evaluable"] += 1
            coverage.append({"elapsed_days":elapsed,**counts})
        verification["coverage"][fold["id"]] = coverage
    survival = json.loads(Path("artifacts/survival-experiment-v1/report.json").read_text())
    best = survival["selected_development_candidate"]
    baseline = best.split("_")[0]+"_km"
    paired = (origin_scores[best]-origin_scores[baseline]).to_numpy()
    rng = np.random.default_rng(42)
    means = [rng.choice(paired,len(paired),replace=True).mean() for _ in range(1000)]
    verification["survival_customer_bootstrap"] = {"candidate":best,"baseline":baseline,"customers":len(paired),
        "mean_brier_delta":float(paired.mean()),"interval_95":np.quantile(means,[.025,.975]).tolist(),
        "interpretation":"Paired customer resampling descriptive interval; does not correct candidate-selection bias or time drift"}
    verification["status"] = "passed"
    write_json(Path("artifacts/timing-verification-v1.json"),verification)
    print(json.dumps(verification,indent=2))
    return verification


if __name__ == "__main__":
    verify()
