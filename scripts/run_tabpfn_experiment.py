"""Offline TabPFN v2 and sample-matched XGBoost benchmark on fixed time cohorts."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

from scripts.build_features_and_splits import FEATURE_COLUMNS, read_csv
from scripts.build_purchase_events import write_csv
from scripts.run_baseline import digest, history_group, write_json
from scripts.run_timing_experiment import horizon_outcome, rounded, summarize, training_rows
from scripts.train_xgboost import fit_schema, matrix


def sample_context(rows, limit):
    return sorted(rows,key=lambda r:hashlib.sha256(r["origin_event_id"].encode()).hexdigest())[:limit]


def feature_array(rows, schema):
    # Only the allowlisted origin-time fields reach the foundation model.
    return np.asarray([[schema["products"].get(r[k],np.nan) if k == "product_id"
        else float(r[k]) if r[k] != "" else np.nan for k in FEATURE_COLUMNS] for r in rows],dtype=np.float32)


def run(config_path, output):
    config_path,output = Path(config_path),Path(output)
    if output.exists():
        raise FileExistsError("Choose a new experiment version")
    cfg = json.loads(config_path.read_text())
    timing_path = Path(cfg["timing_config"])
    timing = json.loads(timing_path.read_text())
    if cfg["version"] != "tabpfn-experiment-v1":
        raise ValueError("Unexpected experiment config")
    weights = Path(cfg["weights"]).resolve()
    if not weights.is_file():
        raise FileNotFoundError("Download the frozen official model checkpoint first")
    # No login/API, remote data upload, or implicit checkpoint download during fitting.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["SKB_DATA_DIRECTORY"] = str(Path("artifacts/cache/skrub").resolve())
    os.environ["TABPFN_MODEL_CACHE_DIR"] = str(weights.parent)
    os.environ["MPLCONFIGDIR"] = str(Path("artifacts/cache/matplotlib").resolve())
    os.environ["XDG_CACHE_HOME"] = str(Path("artifacts/cache").resolve())
    source,event_dir = Path("data/features-splits-full-v1"),Path("data/purchase-events-full-v1")
    manifest = json.loads((source/"report.json").read_text())
    em = json.loads((event_dir/"report.json").read_text())
    for root,meta,names in [(source,manifest,["all-origins-audit.csv"]),(event_dir,em,["daily-events.csv","next-purchase-labels.csv"])]:
        for name in names:
            if digest(root/name) != meta["files"][name]:
                raise ValueError("Frozen input changed")
    rows = read_csv(source/"all-origins-audit.csv")
    labels = {r["origin_event_id"]:r for r in read_csv(event_dir/"next-purchase-labels.csv")}
    events = {r["event_id"]:r for r in read_csv(event_dir/"daily-events.csv")}
    output.mkdir(parents=True)
    provenance = {"config_sha256":digest(config_path),"timing_config_sha256":digest(timing_path),
        "code_sha256":digest(__file__),"weights_sha256":digest(weights),"model_revision":cfg["model_revision"],
        "tabpfn_version":version("tabpfn"),"torch_version":version("torch"),"xgboost_version":xgb.__version__,
        "worker_code_sha256":digest(Path("scripts/tabpfn_worker.py")),
        "input_report_sha256":digest(source/"report.json"),"event_report_sha256":digest(event_dir/"report.json")}
    write_json(output/"config-frozen.json",cfg)
    write_json(output/"provenance.json",provenance)
    started = time.monotonic()
    candidates,folds = defaultdict(list),{}
    for fold in timing["folds"]:
        pool = training_rows(rows,fold["train_end"],cfg["window_days"],timing["horizon_days"])
        train = sample_context(pool,cfg["max_training_rows"])
        valid = [r for r in rows if fold["origin_start"] <= r["origin_on"] <= fold["origin_end"]]
        outcomes = [horizon_outcome(r,timing["horizon_days"],em["last_ordered_on"],labels,events) for r in valid]
        if not train or not valid or "not_mature" in outcomes:
            raise ValueError("Invalid training/evaluation cohort")
        if {r["origin_event_id"] for r in train} & {r["origin_event_id"] for r in valid}:
            raise ValueError("Overlapping cohorts")
        work = output/fold["id"]
        work.mkdir()
        schema = fit_schema(train)
        write_json(work/"schema.json",schema)
        write_json(work/"context-origin-ids.json",[r["origin_event_id"] for r in train])
        x_train,x_valid = feature_array(train,schema),feature_array(valid,schema)
        y = np.asarray([float(r["target_gap_days"]) for r in train])
        # Separate native runtimes: combined XGBoost/PyTorch process crashed in torch
        # initialization on this host. The worker contains no XGBoost imports.
        np.savez(work/"context-and-features.npz",x_train=x_train,y_train=y,x_valid=x_valid)
        subprocess.run([sys.executable,"-X","faulthandler","-m","scripts.tabpfn_worker",
            "--input",str(work/"context-and-features.npz"),"--weights",str(weights),
            "--config",str(config_path),"--output",str(work/"tabpfn-raw.npy")],check=True)
        raw = np.load(work/"tabpfn-raw.npy",allow_pickle=False)
        point = [rounded(v) for v in raw]
        spec = cfg["matched_xgboost"]
        booster = xgb.train({**timing["params"],"objective":spec["objective"],"max_depth":spec["depth"]},
            matrix(train,schema,"target_gap_days"),spec["rounds"])
        booster.save_model(work/"xgboost.json")
        matched = [rounded(v) for v in booster.predict(matrix(valid,schema))]
        folds[fold["id"]] = {"eligible_training_rows":len(pool),"context_rows":len(train),"evaluation_rows":len(valid),
            "context_sha256":digest(work/"context-origin-ids.json"),"schema_sha256":digest(work/"schema.json"),
            "xgboost_sha256":digest(work/"xgboost.json")}
        for name,values in [("tabpfn_v2_1000",point),("xgboost_matched_1000",matched)]:
            for i,row in enumerate(valid):
                observed = outcomes[i] == "observed"
                candidates[name].append({"candidate":name,"fold":fold["id"],"origin_event_id":row["origin_event_id"],
                    "customer_id":row["customer_id"],"product_id":row["product_id"],"origin_on":row["origin_on"],
                    "outcome":outcomes[i],"history_group":history_group(row["episode_purchase_count"]),
                    "predicted_gap_days":values[i],"cart_quantity":rounded(row["qty_median_last3"]),
                    "target_gap_days":row["target_gap_days"] if observed else "",
                    "target_quantity":row["target_quantity"] if observed else ""})
    results = {}
    for name,records in candidates.items():
        write_csv(output/(name+".csv"),list(records[0]),records)
        results[name] = {"overall":summarize(records,timing["horizon_days"]),
            "folds":{f["id"]:summarize([r for r in records if r["fold"] == f["id"]],timing["horizon_days"]) for f in timing["folds"]},
            "predictions_sha256":digest(output/(name+".csv"))}
    report = {"version":cfg["version"],"config":cfg,"provenance":provenance,"folds":folds,"candidates":results,
        "duration_seconds":time.monotonic()-started,"operational_approval":False,"independent_final_test":False,
        "limitations":["TabPFN v2, not the latest noncommercial v3.5 weights",
            "1000-example context and two estimators: bounded benchmark, not maximum attainable performance",
            "Predictions conditional on repurchase within 60 days; not a survival model",
            "Latest-state snapshot; historical state availability unverified",
            "Adaptive development comparison after previous test consumption"]}
    write_json(output/"report.json",report)
    print(json.dumps({"candidates":results,"duration_seconds":report["duration_seconds"]},indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("config/tabpfn-experiment-v1.json"))
    parser.add_argument("--output",type=Path,default=Path("artifacts/tabpfn-experiment-v1"))
    args = parser.parse_args()
    run(args.config,args.output)
