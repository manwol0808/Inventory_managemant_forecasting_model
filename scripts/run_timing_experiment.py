"""Timing-first fixed candidate comparison across three matured historical cohorts.

No independent final test is claimed. All preprocessing is frozen beforehand.
Model inputs remain the existing past-only feature allowlist, not customer IDs.
"""
import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import median

import numpy as np
import xgboost as xgb

from scripts.build_features_and_splits import FEATURE_COLUMNS, read_csv
from scripts.build_purchase_events import write_csv
from scripts.run_baseline import INPUT_FEATURES, digest, fit, history_group, predict, write_json
from scripts.train_xgboost import fit_schema, matrix


def horizon_outcome(row, horizon, end, labels, events):
    deadline=date.fromisoformat(row["origin_on"])+timedelta(days=horizon)
    if deadline>date.fromisoformat(end):
        return "not_mature"
    if row["label_state"]=="observed":
        return "observed" if row["target_on"]<=deadline.isoformat() else "no_repurchase_within_horizon"
    if row["label_state"]=="right_censored":
        return "no_repurchase_within_horizon"
    if row["label_state"]=="blocked_by_hold":
        blocking=events[labels[row["origin_event_id"]]["blocking_event_id"]]
        return "unknown_due_to_hold" if blocking["ordered_on"]<=deadline.isoformat() else "no_repurchase_within_horizon"
    raise ValueError("Unknown label state")


def training_rows(rows, cutoff, window, horizon):
    end=date.fromisoformat(cutoff)
    lower=end-timedelta(days=window-1)
    mature=end-timedelta(days=horizon)
    selected=[r for r in rows if lower.isoformat()<=r["origin_on"]<=mature.isoformat()
              and r["label_state"]=="observed" and 0<int(r["target_gap_days"])<=horizon]
    if any(r["target_on"]>cutoff for r in selected):
        raise ValueError("Training target exceeds cutoff")
    return selected


def recent_gaps(events):
    histories=defaultdict(list)
    result={}
    for event in sorted(events.values(),key=lambda e:(e["ordered_on"],e["event_id"])):
        key=(event["customer_id"],event["product_id"])
        if event["event_state"]=="held":
            histories[key]=[]
        elif event["event_state"]=="completed":
            history=histories[key]
            history.append(date.fromisoformat(event["ordered_on"]))
            gaps=[(b-a).days for a,b in zip(history[-4:],history[-3:])] if len(history)>=4 else [(b-a).days for a,b in zip(history,history[1:])]
            result[event["event_id"]]=median(gaps) if gaps else None
    return result


def rounded(value):
    value=float(value)
    if not math.isfinite(value):
        raise ValueError("Nonfinite prediction")
    return max(1,math.floor(value+0.5))


def summarize(rows,horizon):
    observed=[r for r in rows if r["outcome"]=="observed"]
    absent=[r for r in rows if r["outcome"]=="no_repurchase_within_horizon"]
    if not observed:
        return {"n":0,"origins":len(rows)}
    de=[int(r["predicted_gap_days"])-int(r["target_gap_days"]) for r in observed]
    qe=[int(r["cart_quantity"])-int(r["target_quantity"]) for r in observed]
    by_customer=defaultdict(list)
    for row,error in zip(observed,de):
        by_customer[row["customer_id"]].append(abs(error))
    small=[i for i,r in enumerate(observed) if int(r["target_quantity"])<=2]
    return {"n":len(observed),"origins":len(rows),"customers":len(by_customer),
            "outcomes":dict(Counter(r["outcome"] for r in rows)),
            "date_mae":sum(map(abs,de))/len(de),
            "customer_macro_date_mae":sum(sum(v)/len(v) for v in by_customer.values())/len(by_customer),
            "date_within_3_rate":sum(abs(e)<=3 for e in de)/len(de),
            "date_within_7_rate":sum(abs(e)<=7 for e in de)/len(de),
            "date_exact_rate":sum(e==0 for e in de)/len(de),
            "date_bias":sum(de)/len(de),"date_p90_abs_error":float(np.quantile(np.abs(de),0.9)),
            "quantity_mae":sum(map(abs,qe))/len(qe),
            "quantity_within_1_rate":sum(abs(e)<=1 for e in qe)/len(qe),
            "quantity_within_2_rate":sum(abs(e)<=2 for e in qe)/len(qe),
            "quantity_exact_rate":sum(e==0 for e in qe)/len(qe),
            "quantity_over_by_more_than_2_rate":sum(e>2 for e in qe)/len(qe),
            "quantity_under_by_more_than_2_rate":sum(e< -2 for e in qe)/len(qe),
            "small_quantity_n":len(small),
            "small_quantity_exact_rate":sum(qe[i]==0 for i in small)/len(small) if small else None,
            "small_quantity_over_by_2_or_more_rate":sum(qe[i]>=2 for i in small)/len(small) if small else None,
            "no_repurchase_n":len(absent),
            "no_repurchase_but_predicted_within_horizon_rate":sum(int(r["predicted_gap_days"])<=horizon for r in absent)/len(absent) if absent else None}


def run(config_path,output):
    config_path,output=map(Path,(config_path,output))
    if output.exists():
        raise FileExistsError("Choose a new experiment version, not an overwrite")
    cfg=json.loads(config_path.read_text())
    if cfg["version"]!="timing-experiment-v2":
        raise ValueError("Unsupported experiment")
    source=Path("data/features-splits-full-v1")
    event_dir=Path("data/purchase-events-full-v1")
    manifest=json.loads((source/"report.json").read_text())
    event_manifest=json.loads((event_dir/"report.json").read_text())
    for root,metadata,filenames in ((source,manifest,["all-origins-audit.csv"]),(event_dir,event_manifest,["daily-events.csv","next-purchase-labels.csv"])):
        for filename in filenames:
            if digest(root/filename)!=metadata["files"][filename]:
                raise ValueError("Frozen input changed")
    rows=read_csv(source/"all-origins-audit.csv")
    labels={r["origin_event_id"]:r for r in read_csv(event_dir/"next-purchase-labels.csv")}
    events={r["event_id"]:r for r in read_csv(event_dir/"daily-events.csv")}
    gaps3=recent_gaps(events)
    horizon=cfg["horizon_days"]
    baseline_cfg=json.loads(Path("config/baseline-v1.json").read_text())
    output.mkdir(parents=True)
    write_json(output/"config-frozen.json",cfg)
    started=time.monotonic()
    metadata={"config_sha256":digest(config_path),"code_sha256":digest(__file__),
              "input_report_sha256":digest(source/"report.json"),"event_report_sha256":digest(event_dir/"report.json"),
              "feature_columns":FEATURE_COLUMNS,"xgboost_version":xgb.__version__,"numpy_version":np.__version__}
    write_json(output/"provenance.json",metadata)
    candidate_files={}
    folds={}
    model_files={}
    for fold in cfg["folds"]:
        valid=[r for r in rows if fold["origin_start"]<=r["origin_on"]<=fold["origin_end"]]
        outcomes={r["origin_event_id"]:horizon_outcome(r,horizon,event_manifest["last_ordered_on"],labels,events) for r in valid}
        if not valid or "not_mature" in outcomes.values():
            raise ValueError("Evaluation cohort lacks complete horizon")
        folds[fold["id"]]={"config":fold,"origins":len(valid),"outcomes":dict(Counter(outcomes.values())),"training":{}}
        for window in cfg["origin_training_windows_days"]:
            train=training_rows(rows,fold["train_end"],window,horizon)
            if not train or {r["origin_event_id"] for r in train}&{r["origin_event_id"] for r in valid}:
                raise ValueError("Empty training or identity overlap")
            if max(r["target_on"] for r in train)>=fold["origin_start"]:
                raise ValueError("Future training target")
            folds[fold["id"]]["training"][str(window)]={"rows":len(train),"origin_min":min(r["origin_on"] for r in train),
                "origin_max":max(r["origin_on"] for r in train),"target_max":max(r["target_on"] for r in train)}
            work=output/"models"/fold["id"]/str(window)
            work.mkdir(parents=True)
            baseline=fit([{**r,"outer_split":"train"} for r in train],baseline_cfg,fold["train_end"])
            write_json(work/"baseline.json",baseline)
            basepred=[predict(baseline,{k:r[k] for k in INPUT_FEATURES})["predicted_gap_days"] for r in valid]
            predictions={"pair_median":basepred,
                         "last_gap":[rounded(r["gap_last_days"]) if int(r["gap_count"]) else basepred[i] for i,r in enumerate(valid)],
                         "recent3_gap":[rounded(gaps3[r["origin_event_id"]]) if gaps3[r["origin_event_id"]] is not None else basepred[i] for i,r in enumerate(valid)]}
            schema=fit_schema(train)
            write_json(work/"schema.json",schema)
            dm=matrix(train,schema,"target_gap_days")
            validation_dm=matrix([{k:r[k] for k in FEATURE_COLUMNS} for r in valid],schema)
            original_labels=dm.get_label().copy()
            for spec in cfg["xgboost_variants"]:
                dm.set_label(np.log1p(original_labels) if spec.get("log_target") else original_labels)
                params={**cfg["params"],"max_depth":spec["depth"],"objective":spec["objective"]}
                curve={}
                model=xgb.train(params,dm,spec["rounds"],evals=[(dm,"train")],evals_result=curve,verbose_eval=False)
                model.save_model(work/(spec["id"]+".json"))
                write_json(work/(spec["id"]+"-curve.json"),curve)
                saved=xgb.Booster({"nthread":4})
                saved.load_model(work/(spec["id"]+".json"))
                raw=saved.predict(validation_dm)
                np.testing.assert_array_equal(raw,model.predict(validation_dm))
                if spec.get("log_target"):
                    raw=np.expm1(raw.astype(float))
                predictions[spec["id"]]=[rounded(value) for value in raw]
            predictions["personal_hybrid"]=[rounded(gaps3[r["origin_event_id"]]) if int(r["gap_count"])>=3 else predictions["abs_d2_200"][i] for i,r in enumerate(valid)]
            for method,values in predictions.items():
                for quantity_rule in cfg["quantity_rules"]:
                    candidate=f"w{window}_{method}_q{quantity_rule}"
                    target=output/(candidate+".csv")
                    candidate_files[candidate]=target
                    records=[]
                    for i,row in enumerate(valid):
                        observed=outcomes[row["origin_event_id"]]=="observed"
                        records.append({"candidate":candidate,"fold":fold["id"],"origin_event_id":row["origin_event_id"],
                            "customer_id":row["customer_id"],"product_id":row["product_id"],"origin_on":row["origin_on"],
                            "outcome":outcomes[row["origin_event_id"]],"history_group":history_group(row["episode_purchase_count"]),
                            "predicted_gap_days":values[i],"cart_quantity":rounded(row["qty_last"] if quantity_rule=="last" else row["qty_median_last3"]),
                            "target_gap_days":row["target_gap_days"] if observed else "",
                            "target_quantity":row["target_quantity"] if observed else ""})
                    new=not target.exists()
                    with target.open("a",newline="",encoding="utf-8") as stream:
                        writer=csv.DictWriter(stream,fieldnames=list(records[0]))
                        if new:
                            writer.writeheader()
                        writer.writerows(records)
            model_files.update({str(p.relative_to(output)):digest(p) for p in work.glob("*.json")})
            print(json.dumps({"fold":fold["id"],"window":window,"training_rows":len(train),"evaluation_origins":len(valid),"status":"completed"}),flush=True)
    candidates={}
    for candidate,path in candidate_files.items():
        values=read_csv(path)
        overall=summarize(values,horizon)
        fold_scores={fold["id"]:summarize([r for r in values if r["fold"]==fold["id"]],horizon) for fold in cfg["folds"]}
        group_scores={g:summarize([r for r in values if r["history_group"]==g],horizon) for g in ("1","2","3-5","6+")}
        candidates[candidate]={"overall":overall,"folds":fold_scores,"history_groups":group_scores,"predictions_sha256":digest(path)}
        customers=defaultdict(list)
        for row in values:
            customers[row["customer_id"]].append(row)
        customer_rows=[]
        for customer,group in sorted(customers.items()):
            item=summarize(group,horizon)
            customer_rows.append({"customer_id":customer,"origins":len(group),"observed":item["n"],
                                  "date_mae":item.get("date_mae",""),"quantity_within_1_rate":item.get("quantity_within_1_rate","")})
        write_csv(output/(candidate+"-customers.csv"),list(customer_rows[0]),customer_rows)
    eligible=[]
    for candidate,item in candidates.items():
        baseline=candidates[candidate.split("_")[0]+"_pair_median_qlast"]
        passed=item["overall"]["quantity_within_1_rate"]+1e-12>=baseline["overall"]["quantity_within_1_rate"]
        item["quantity_guardrail_passed"]=passed
        if passed:
            eligible.append(candidate)
    ranked=sorted(eligible,key=lambda name:(candidates[name]["overall"]["customer_macro_date_mae"],candidates[name]["overall"]["date_mae"],-candidates[name]["overall"]["quantity_within_1_rate"],name))
    result={"version":cfg["version"],"config":cfg,"provenance":metadata,"folds":folds,"candidates":candidates,
            "ranking":ranked,"selected_development_candidate":ranked[0],"model_files":model_files,
            "duration_seconds":time.monotonic()-started,"operational_approval":False,"independent_final_test":False,
            "strict_training_ready":False,"limitations":["Latest-state historical availability unverified",
            "Date and quantity errors conditional on repurchase within 60 days; no-repurchase behavior reported separately",
            "Adaptive development after an earlier test was consumed; new future evaluation required",
            "Customer macro score weights only customers with observed outcomes; small-sample uncertainty remains"]}
    write_json(output/"report.json",result)
    print(json.dumps({"candidates":len(candidates),"folds":len(folds),"selected":ranked[0],"metrics":candidates[ranked[0]]["overall"],"duration_seconds":result["duration_seconds"]},ensure_ascii=False,indent=2))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("config/timing-experiment-v2.json"))
    parser.add_argument("--output",type=Path,default=Path("artifacts/timing-experiment-v2"))
    args=parser.parse_args()
    run(args.config,args.output)
