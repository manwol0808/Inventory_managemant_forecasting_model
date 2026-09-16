"""Frozen AFT/Kaplan-Meier comparison; origin features only, past-cutoff labels.

Predictions concern a repeat purchase, not depletion of physical inventory.
No probability threshold here grants permission to mutate a shopping cart.
"""
import argparse
import json
import math
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.special import log_ndtr

from scripts.build_features_and_splits import FEATURE_COLUMNS, read_csv
from scripts.build_purchase_events import write_csv
from scripts.run_baseline import digest, history_group, write_json
from scripts.run_timing_experiment import horizon_outcome, rounded, summarize
from scripts.train_xgboost import fit_schema, matrix


def observed_followup(row, cutoff, horizon, labels, events):
    """Return exact event or safe right-censor bound using only time <= cutoff.

    A hold is not known to be a non-purchase: censor the day BEFORE the hold.
    A future exact target is not visible at the fit cutoff and becomes censored.
    """
    origin = date.fromisoformat(row["origin_on"])
    stop = min(date.fromisoformat(cutoff), origin + timedelta(days=horizon))
    if row["label_state"] == "blocked_by_hold":
        hold = events[labels[row["origin_event_id"]]["blocking_event_id"]]
        stop = min(stop, date.fromisoformat(hold["ordered_on"]) - timedelta(days=1))
    elif row["label_state"] not in {"observed", "right_censored"}:
        raise ValueError("Unknown label state")
    if row["label_state"] == "observed" and row["target_on"] <= stop.isoformat():
        duration = (date.fromisoformat(row["target_on"]) - origin).days
        if duration <= 0 or duration != int(row["target_gap_days"]):
            raise ValueError("Inconsistent positive event duration")
        return duration, True
    return max(0, (stop-origin).days), False


def survival_training_rows(rows, cutoff, window, horizon, labels, events):
    end = date.fromisoformat(cutoff)
    lower, mature = end-timedelta(days=window-1), end-timedelta(days=horizon)
    selected, bounds = [], []
    for row in rows:
        if lower.isoformat() <= row["origin_on"] <= mature.isoformat():
            duration, event = observed_followup(row, cutoff, horizon, labels, events)
            if duration > 0:
                selected.append(row)
                bounds.append((duration, duration if event else float("inf")))
    return selected, np.asarray(bounds, dtype=float)


def aft_log_survival(margin, days, distribution, scale):
    if scale <= 0:
        raise ValueError("Positive AFT scale required")
    margin = np.asarray(margin, dtype=float)
    if days == 0:
        return np.zeros_like(margin)
    if days < 0:
        raise ValueError("Nonnegative elapsed time required")
    z = (math.log(days)-margin)/scale
    if distribution == "normal":
        return log_ndtr(-z)
    if distribution == "logistic":
        return -np.logaddexp(0, z)
    raise ValueError("Unsupported AFT distribution")


def conditional_probability(log_survival_start, log_survival_end):
    difference = np.asarray(log_survival_end)-np.asarray(log_survival_start)
    if not np.isfinite(difference).all() or np.any(difference > 1e-9):
        raise ValueError("Invalid survival probabilities")
    return np.clip(-np.expm1(difference), 0, 1)


def fit_km(bounds, horizon):
    """Discrete-day Kaplan-Meier; events at d are counted before censors at d."""
    at_risk = len(bounds)
    curve = [1.0]
    for day in range(1, horizon+1):
        count = int(np.sum((bounds[:, 0] == day) & np.isfinite(bounds[:, 1])))
        curve.append(curve[-1]*(1-count/at_risk) if at_risk else curve[-1])
        at_risk -= int(np.sum(bounds[:, 0] == day))
    return curve


def probability_metrics(origins, landmarks, thresholds):
    known = [r for r in origins if r["outcome"] != "unknown_due_to_hold"]
    brier60 = [(float(r["probability_60d"])-(r["outcome"] == "observed"))**2 for r in known]
    by_origin = defaultdict(list)
    customer_for_origin = {}
    for r in landmarks:
        by_origin[r["origin_event_id"]].append((float(r["probability"])-int(r["event_next_7d"]))**2)
        customer_for_origin[r["origin_event_id"]] = r["customer_id"]
    by_customer = defaultdict(list)
    for origin, scores in by_origin.items():
        by_customer[customer_for_origin[origin]].append(float(np.mean(scores)))
    result = {"horizon_known_n": len(known), "brier_60d": float(np.mean(brier60)),
              "landmarks_n": len(landmarks), "landmark_positive_n": sum(int(r["event_next_7d"]) for r in landmarks),
              "landmark_customer_macro_brier": float(np.mean([np.mean(v) for v in by_customer.values()])),
              "landmark_brier": float(np.mean([e for v in by_origin.values() for e in v])),
              "landmark_customers": len(by_customer), "thresholds": {}, "calibration": []}
    for threshold in thresholds:
        tp = fp = fn = tn = 0
        for r in landmarks:
            alert, truth = float(r["probability"]) >= threshold, bool(int(r["event_next_7d"]))
            tp += alert and truth
            fp += alert and not truth
            fn += not alert and truth
            tn += not alert and not truth
        result["thresholds"][str(threshold)] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp/(tp+fp) if tp+fp else None, "recall": tp/(tp+fn) if tp+fn else None,
            "false_positive_rate": fp/(fp+tn) if fp+tn else None, "alert_rate": (tp+fp)/len(landmarks)}
    for lo, hi in zip([0,.1,.2,.4,.6,.8], [.1,.2,.4,.6,.8,1.000001]):
        group = [r for r in landmarks if lo <= float(r["probability"]) < hi]
        result["calibration"].append({"lower":lo,"upper":min(hi,1),"n":len(group),
            "mean_probability":float(np.mean([float(r["probability"]) for r in group])) if group else None,
            "event_rate":float(np.mean([int(r["event_next_7d"]) for r in group])) if group else None})
    return result


def run(config_path, output):
    config_path, output = Path(config_path), Path(output)
    if output.exists():
        raise FileExistsError("Choose a new experiment version")
    cfg = json.loads(config_path.read_text())
    timing_path = Path(cfg["timing_config"])
    timing = json.loads(timing_path.read_text())
    if cfg["version"] != "survival-experiment-v1" or cfg["horizon_days"] != timing["horizon_days"]:
        raise ValueError("Invalid experiment configuration")
    if max(cfg["landmarks_days"])+cfg["prediction_window_days"] > cfg["horizon_days"]:
        raise ValueError("Landmark exceeds horizon")
    source, event_dir = Path("data/features-splits-full-v1"), Path("data/purchase-events-full-v1")
    input_manifest = json.loads((source/"report.json").read_text())
    event_manifest = json.loads((event_dir/"report.json").read_text())
    for root, manifest, names in [(source,input_manifest,["all-origins-audit.csv"]),(event_dir,event_manifest,["daily-events.csv","next-purchase-labels.csv"])]:
        for name in names:
            if digest(root/name) != manifest["files"][name]:
                raise ValueError("Frozen input changed")
    rows = read_csv(source/"all-origins-audit.csv")
    labels = {r["origin_event_id"]:r for r in read_csv(event_dir/"next-purchase-labels.csv")}
    events = {r["event_id"]:r for r in read_csv(event_dir/"daily-events.csv")}
    output.mkdir(parents=True)
    provenance = {"config_sha256":digest(config_path),"timing_config_sha256":digest(timing_path),
                  "code_sha256":digest(__file__),"input_report_sha256":digest(source/"report.json"),
                  "event_report_sha256":digest(event_dir/"report.json"),"xgboost_version":xgb.__version__,
                  "feature_columns":FEATURE_COLUMNS}
    write_json(output/"config-frozen.json",cfg)
    write_json(output/"provenance.json",provenance)
    all_origins, all_landmarks, folds, model_files = defaultdict(list), defaultdict(list), {}, {}
    started = time.monotonic()
    for fold in timing["folds"]:
        valid = [r for r in rows if fold["origin_start"] <= r["origin_on"] <= fold["origin_end"]]
        outcomes = [horizon_outcome(r,cfg["horizon_days"],event_manifest["last_ordered_on"],labels,events) for r in valid]
        if not valid or "not_mature" in outcomes:
            raise ValueError("Incomplete evaluation followup")
        valid_followup = [observed_followup(r,event_manifest["last_ordered_on"],cfg["horizon_days"],labels,events) for r in valid]
        for window in timing["origin_training_windows_days"]:
            train, bounds = survival_training_rows(rows,fold["train_end"],window,cfg["horizon_days"],labels,events)
            if not train or {r["origin_event_id"] for r in train} & {r["origin_event_id"] for r in valid}:
                raise ValueError("Empty or overlapping train cohort")
            work = output/"models"/fold["id"]/str(window)
            work.mkdir(parents=True)
            folds[f"{fold['id']}_w{window}"] = {"rows":len(train),"exact":int(np.isfinite(bounds[:,1]).sum()),
                "censored":int(np.isinf(bounds[:,1]).sum()),"origin_min":min(r["origin_on"] for r in train),
                "origin_max":max(r["origin_on"] for r in train),"train_end":fold["train_end"]}
            schema = fit_schema(train)
            write_json(work/"schema.json",schema)
            dm = matrix([{k:r[k] for k in FEATURE_COLUMNS} for r in train],schema)
            dm.set_float_info("label_lower_bound",bounds[:,0])
            dm.set_float_info("label_upper_bound",bounds[:,1])
            vm = matrix([{k:r[k] for k in FEATURE_COLUMNS} for r in valid],schema)
            km = fit_km(bounds,cfg["horizon_days"])
            write_json(work/"km.json",km)
            specs = [{"id":"km"}]+cfg["variants"]
            for spec in specs:
                candidate = f"w{window}_{spec['id']}"
                if spec["id"] == "km":
                    median_day = next((d for d,s in enumerate(km) if s <= .5),cfg["horizon_days"]+1)
                    point = [median_day]*len(valid)
                    probs60 = np.full(len(valid),1-km[-1])
                    probabilities = {t:np.full(len(valid),1-km[t+cfg["prediction_window_days"]]/km[t]) for t in cfg["landmarks_days"]}
                else:
                    params = {**timing["params"],"objective":"survival:aft","eval_metric":"aft-nloglik",
                              "max_depth":spec["depth"],"aft_loss_distribution":spec["distribution"],
                              "aft_loss_distribution_scale":spec["scale"]}
                    curve = {}
                    model = xgb.train(params,dm,cfg["rounds"],evals=[(dm,"train")],evals_result=curve,verbose_eval=False)
                    model.save_model(work/(spec["id"]+".json"))
                    write_json(work/(spec["id"]+"-curve.json"),curve)
                    saved = xgb.Booster({"nthread":4})
                    saved.load_model(work/(spec["id"]+".json"))
                    margin = saved.predict(vm,output_margin=True).astype(float)
                    np.testing.assert_array_equal(margin,model.predict(vm,output_margin=True).astype(float))
                    point = [rounded(v) for v in np.exp(margin)]
                    probs60 = -np.expm1(aft_log_survival(margin,cfg["horizon_days"],spec["distribution"],spec["scale"]))
                    probabilities = {t:conditional_probability(aft_log_survival(margin,t,spec["distribution"],spec["scale"]),
                        aft_log_survival(margin,t+cfg["prediction_window_days"],spec["distribution"],spec["scale"])) for t in cfg["landmarks_days"]}
                for i,row in enumerate(valid):
                    observed = outcomes[i] == "observed"
                    item = {"candidate":candidate,"fold":fold["id"],"origin_event_id":row["origin_event_id"],
                        "customer_id":row["customer_id"],"product_id":row["product_id"],"origin_on":row["origin_on"],
                        "outcome":outcomes[i],"history_group":history_group(row["episode_purchase_count"]),
                        "predicted_gap_days":point[i],"cart_quantity":rounded(row["qty_last"]),
                        "target_gap_days":row["target_gap_days"] if observed else "",
                        "target_quantity":row["target_quantity"] if observed else "", "probability_60d":float(probs60[i])}
                    all_origins[candidate].append(item)
                    duration,event = valid_followup[i]
                    for elapsed in cfg["landmarks_days"]:
                        end = elapsed+cfg["prediction_window_days"]
                        if duration <= elapsed or (not event and duration < end):
                            continue
                        all_landmarks[candidate].append({"candidate":candidate,"fold":fold["id"],
                            "origin_event_id":row["origin_event_id"],"customer_id":row["customer_id"],
                            "elapsed_days":elapsed,"probability":float(probabilities[elapsed][i]),
                            "event_next_7d":int(event and duration <= end)})
            model_files.update({str(p.relative_to(output)):digest(p) for p in work.glob("*.json")})
            print(json.dumps({"fold":fold["id"],"window":window,**folds[f"{fold['id']}_w{window}"]}),flush=True)
    candidates = {}
    for candidate,records in all_origins.items():
        landmarks = all_landmarks[candidate]
        write_csv(output/(candidate+".csv"),list(records[0]),records)
        write_csv(output/(candidate+"-landmarks.csv"),list(landmarks[0]),landmarks)
        candidates[candidate] = {"point_metrics":summarize(records,cfg["horizon_days"]),
            "probability_metrics":probability_metrics(records,landmarks,cfg["thresholds"]),
            "folds":{f["id"]:probability_metrics([r for r in records if r["fold"]==f["id"]],
                [r for r in landmarks if r["fold"]==f["id"]],cfg["thresholds"]) for f in timing["folds"]},
            "history_groups":{g:summarize([r for r in records if r["history_group"]==g],cfg["horizon_days"]) for g in ("1","2","3-5","6+")},
            "predictions_sha256":digest(output/(candidate+".csv")),"landmarks_sha256":digest(output/(candidate+"-landmarks.csv"))}
    ranking = sorted(candidates,key=lambda c:(candidates[c]["probability_metrics"][cfg["primary_ranking"]],c))
    report = {"version":cfg["version"],"config":cfg,"provenance":provenance,"folds":folds,
        "candidates":candidates,"ranking":ranking,"selected_development_candidate":ranking[0],"model_files":model_files,
        "duration_seconds":time.monotonic()-started,"operational_approval":False,"independent_final_test":False,
        "limitations":["Latest-state snapshot historical availability unverified",
            "Artificial right censor at 60 days; extrapolated dates beyond 60 are not validated",
            "Weekly landmark alerts are diagnostic opportunities, not deduplicated cart actions",
            "Hold censoring may be informative; probability calibration is evaluated, not guaranteed",
            "Customer macro Brier averages opportunities within origin, origins within customer, then customers",
            "All model features frozen at origin; later calendar/catalog changes not included"]}
    write_json(output/"report.json",report)
    print(json.dumps({"selected":ranking[0],"probability_metrics":candidates[ranking[0]]["probability_metrics"],"duration_seconds":report["duration_seconds"]},indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("config/survival-experiment-v1.json"))
    parser.add_argument("--output",type=Path,default=Path("artifacts/survival-experiment-v1"))
    args = parser.parse_args()
    run(args.config,args.output)
