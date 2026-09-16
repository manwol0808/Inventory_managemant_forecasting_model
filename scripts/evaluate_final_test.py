"""One-time evaluation of frozen saved candidates. No fitting or candidate search."""
import argparse
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import xgboost as xgb

from scripts.build_features_and_splits import FEATURE_COLUMNS, next_midnight, read_csv
from scripts.build_purchase_events import write_csv
from scripts.run_baseline import INPUT_FEATURES, digest, history_group, metrics, predict, write_json
from scripts.train_xgboost import matrix, postprocess


def evaluate(split_dir, run_dir, baseline_dir, output):
    split_dir, run_dir, baseline_dir, output = map(Path,(split_dir,run_dir,baseline_dir,output))
    if output.exists():
        raise FileExistsError("Final test already evaluated or started; do not overwrite")
    report=json.loads((run_dir/"report.json").read_text())
    baseline_report=json.loads((baseline_dir/"report.json").read_text())
    manifest=json.loads((split_dir/"report.json").read_text())
    if digest(baseline_dir/"report.json")!=report["baseline_report_sha256"] or digest(split_dir/"report.json")!=baseline_report["input_report_sha256"]:
        raise ValueError("Provenance mismatch")
    if digest(run_dir/"feature-schema.json")!=report["files"]["feature-schema.json"] or digest(baseline_dir/"model.json")!=baseline_report["files"]["model.json"]:
        raise ValueError("Schema/baseline changed")
    models={}
    for target in ("quantity","gap"):
        path=run_dir/(target+"-model.json")
        if digest(path)!=report["files"][path.name]:
            raise ValueError("Candidate changed")
        models[target]=xgb.Booster({"nthread":4})
        models[target].load_model(path)
    cfg=manifest["split_config"]
    output.mkdir(parents=True)
    # Persist the frozen choice BEFORE loading any holdout rows.
    freeze={"candidate":str(run_dir),"candidate_report_sha256":digest(run_dir/"report.json"),
            "baseline":str(baseline_dir),"baseline_report_sha256":digest(baseline_dir/"report.json"),
            "split_report_sha256":digest(split_dir/"report.json"),"test_sha256_expected":manifest["files"]["final_test.csv"],
            "train_end":cfg["train_end"],"test_after":cfg["validation_end"],"test_through":cfg["test_end"],
            "refit":False,"post_test_tuning_allowed":False,"operational_approval":False}
    write_json(output/"selection-frozen.json",freeze)
    if digest(split_dir/"final_test.csv")!=freeze["test_sha256_expected"]:
        raise ValueError("Holdout changed")
    rows=read_csv(split_dir/"final_test.csv")
    if not rows or len(rows)!=manifest["exploratory_file_rows"]["final_test"]:
        raise ValueError("Test count mismatch")
    seen=set()
    for row in rows:
        if (not cfg["validation_end"] < row["origin_on"] < row["target_on"] <= cfg["test_end"]
                or row["fit_role"]!="final_test" or row["outer_split"]!="test" or row["label_state"]!="observed"
                or row["origin_event_id"] in seen
                or row["prediction_asof"]!=next_midnight(row["origin_on"]).isoformat()
                or int(row["target_quantity"])<=0
                or int(row["target_gap_days"])!=(date.fromisoformat(row["target_on"])-date.fromisoformat(row["origin_on"])).days):
            raise ValueError("Holdout identity/date/label contract violation")
        seen.add(row["origin_event_id"])
    schema=json.loads((run_dir/"feature-schema.json").read_text())
    # Target/metadata columns are removed before predicting.
    features=[{k:r[k] for k in FEATURE_COLUMNS} for r in rows]
    dm=matrix(features,schema)
    qty,cart,gap=postprocess(models["quantity"].predict(dm),models["gap"].predict(dm))
    baseline=json.loads((baseline_dir/"model.json").read_text())
    predicted,cart_rows,base_rows=[],[],[]
    for i,row in enumerate(rows):
        common={k:row[k] for k in ("origin_event_id","customer_id","product_id","origin_on","target_on","target_quantity","target_gap_days")}
        common["history_group"]=history_group(row["episode_purchase_count"])
        p={**common,"predicted_quantity":float(qty[i]),"cart_quantity":int(cart[i]),"predicted_gap_days":int(gap[i]),
           "predicted_on":(date.fromisoformat(row["origin_on"])+timedelta(days=int(gap[i]))).isoformat()}
        predicted.append(p)
        cart_rows.append({**p,"predicted_quantity":int(cart[i])})
        base_rows.append({**common,**predict(baseline,{k:row[k] for k in INPUT_FEATURES})})
    for name,values in (("predictions.csv",predicted),("baseline-predictions.csv",base_rows)):
        write_csv(output/name,list(values[0]),values)
    groups={g:{"candidate":metrics([r for r in predicted if r["history_group"]==g],7),
               "cart":metrics([r for r in cart_rows if r["history_group"]==g],7),
               "baseline":metrics([r for r in base_rows if r["history_group"]==g],7)} for g in ("1","2","3-5","6+")}
    result={"version":"final-test-v1","selection":freeze,"rows":len(rows),"metrics":metrics(predicted,7),
            "cart_metrics":metrics(cart_rows,7),"baseline_metrics":metrics(base_rows,7),"history_groups":groups,
            "coverage":{"origins":manifest["origins_by_outer_split"]["test"],"outcomes":manifest["role_outcomes"]["final_test"],
                        "evaluated_fraction":len(rows)/manifest["origins_by_outer_split"]["test"]},
            "final_test_metrics_computed":True,"final_test_consumed":True,"operational_approval":False,
            "strict_training_ready":False,"guardrail_status":"unassessed","date_tolerance_days_descriptive":7,
            "limitations":report["limitations"],"code_sha256":digest(__file__),
            "files":{p.name:digest(p) for p in output.glob("*.csv")}}
    write_json(output/"report.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",required=True)
    parser.add_argument("--candidate",required=True)
    parser.add_argument("--baseline",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    r=evaluate(args.split,args.candidate,args.baseline,args.output)
    print(json.dumps({k:r[k] for k in ("rows","metrics","cart_metrics","baseline_metrics","coverage")},ensure_ascii=False,indent=2))
