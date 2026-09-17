"""Frozen historical ablation of cadence changes, item speed and routed specialists."""
import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb

from scripts.build_features_and_splits import read_csv
from scripts.build_purchase_events import write_csv
from scripts.predict_replenishment import base_champion_path
from scripts.purchase_segment_features import GROUPS, feature_names, segment_features
from scripts.run_baseline import digest, write_json
from scripts.run_timing_experiment import horizon_outcome, rounded, summarize, training_rows


def schema_for(rows, mode):
    return {"mode":mode,"features":feature_names(mode),
            "products":{p:i for i,p in enumerate(sorted({r["product_id"] for r in rows}))}}


def design_matrix(rows, schema, target=None):
    if schema["features"] != feature_names(schema["mode"]):
        raise ValueError("Unexpected feature allowlist")
    values=[]
    for row in rows:
        data=[]
        for field in schema["features"]:
            value=row[field]
            if field == "product_id":
                value=schema["products"].get(value,np.nan)
            elif value == "":
                value=np.nan
            else:
                value=float(value)
                if not math.isfinite(value):
                    raise ValueError("Nonfinite feature")
            data.append(value)
        values.append(data)
    return xgb.DMatrix(np.asarray(values,dtype=np.float32),
                       label=np.asarray([float(r[target]) for r in rows]) if target else None,
                       feature_names=schema["features"], feature_types=["c"]+["q"]*(len(schema["features"])-1),
                       enable_categorical=True,nthread=4)


class SegmentPredictor:
    """Route already-built, origin-time features without reading target columns."""
    def __init__(self, folder):
        self.root=Path(folder)
        self.router=json.loads((self.root/"router.json").read_text())
        self.models={}
        for ref in set(self.router["models"].values()):
            path=self.root.parent/ref
            schema=json.loads((path/"schema.json").read_text())
            loaded=[]
            for target in ("gap","quantity"):
                model=xgb.Booster({"nthread":4})
                model.load_model(path/(target+".json"))
                loaded.append(model)
            self.models[ref]=(schema,loaded)

    def predict(self, rows):
        output=[None]*len(rows)
        buckets=defaultdict(list)
        for i,row in enumerate(rows):
            group=row["customer_group"]
            if group not in GROUPS:
                raise ValueError("Unknown customer group")
            route=group if group in self.router["models"] else "global"
            buckets[route].append(i)
        for route,indices in buckets.items():
            schema,models=self.models[self.router["models"][route]]
            dm=design_matrix([rows[i] for i in indices],schema)
            gaps,quantities=(m.predict(dm) for m in models)
            for i,gap,quantity in zip(indices,gaps,quantities):
                output[i]={"predicted_gap_days":rounded(gap),"cart_quantity":rounded(quantity),"route":route}
        return output


def scheduled_gap(gap, policy):
    if policy == "none":
        lead=0
    elif policy in {"lead7","lead10"}:
        lead=int(policy[4:])
    elif policy == "relative25_cap7":
        lead=min(7,max(1,math.ceil(gap*.25)))
    else:
        raise ValueError("Unknown cart policy")
    return max(1,gap-lead)


def metrics(rows, cfg):
    result=summarize(rows,90)
    result["outcomes"]=dict(Counter(r["outcome"] for r in rows))
    result["routes"]=dict(Counter(r["route"] for r in rows))
    observed=[r for r in rows if r["outcome"] == "observed"]
    absent=[r for r in rows if r["outcome"] == "no_repurchase_within_horizon"]
    policies={}
    for policy in cfg["cart_policies"]:
        margins=[int(r["target_gap_days"])-scheduled_gap(int(r["predicted_gap_days"]),policy) for r in observed]
        n=len(margins)
        counts={"early":sum(m>0 for m in margins),"same_day":sum(m==0 for m in margins),
                "late":sum(m<0 for m in margins),"early_1_to_10":sum(1<=m<=10 for m in margins),
                "early_over_30":sum(m>30 for m in margins),"at_least_7_days":sum(m>=7 for m in margins)}
        by_customer=defaultdict(list)
        for row,margin in zip(observed,margins):
            by_customer[row["customer_id"]].append(margin<0)
        policies[policy]={"observed_n":n,**counts,
            **{k+"_rate":v/n if n else None for k,v in counts.items()},
            "customer_macro_late_rate":float(np.mean([np.mean(v) for v in by_customer.values()])) if by_customer else None,
            "median_margin_days":float(np.median(margins)) if margins else None,
            "next_day_suggestion_rate_all":sum(scheduled_gap(int(r["predicted_gap_days"]),policy)==1 for r in rows)/len(rows) if rows else None,
            "nonreturn_n":len(absent),
            "nonreturn_suggestion_within90_rate":sum(scheduled_gap(int(r["predicted_gap_days"]),policy)<=90 for r in absent)/len(absent) if absent else None}
    result["cart_policies"]=policies
    return result


def train_candidates(train, valid, cfg, work):
    work.mkdir(parents=True)
    counts=Counter(r["customer_group"] for r in train)
    eligible=[g for g in GROUPS if g != "insufficient" and counts[g]>=cfg["minimum_specialist_rows"]
              and len({r["customer_id"] for r in train if r["customer_group"]==g})>=cfg["minimum_specialist_customers"]]
    fitted={}
    quantity_cache={}
    predictions={}
    for spec in cfg["candidates"]:
        routes={}
        for group in ["global"]+(eligible if spec["specialists"] else []):
            key=f'{spec["features"]}_{group}_q{int(spec["alpha"]*100)}'
            routes[group]=key
            if key in fitted:
                continue
            selected=train if group == "global" else [r for r in train if r["customer_group"]==group]
            schema=schema_for(selected,spec["features"])
            folder=work/key
            folder.mkdir()
            write_json(folder/"schema.json",schema)
            write_json(folder/"training.json",{"origin_ids":[r["origin_event_id"] for r in selected],
                "target_max":max(r["target_on"] for r in selected),"customer_count":len({r["customer_id"] for r in selected})})
            dm=design_matrix(selected,schema,"target_gap_days")
            model=xgb.train({**cfg["params"],"objective":"reg:quantileerror","quantile_alpha":spec["alpha"]},dm,cfg["rounds"])
            model.save_model(folder/"gap.json")
            quantity_key=(spec["features"],group)
            if quantity_key not in quantity_cache:
                dm.set_label(np.asarray([float(r["target_quantity"]) for r in selected]))
                quantity_cache[quantity_key]=xgb.train({**cfg["params"],"objective":"reg:absoluteerror"},dm,cfg["rounds"])
            quantity_cache[quantity_key].save_model(folder/"quantity.json")
            # Predictions below load saved artifacts. Check model IO against training implementation.
            saved=xgb.Booster({"nthread":4})
            saved.load_model(folder/"gap.json")
            np.testing.assert_array_equal(model.predict(dm),saved.predict(dm))
            fitted[key]=True
        folder=work/spec["id"]
        folder.mkdir()
        write_json(folder/"router.json",{"candidate":spec,"models":routes,
            "grouping":"customer purchase-day cadence at origin, recalculated from past events", "fallback":"global"})
        predictions[spec["id"]]=SegmentPredictor(folder).predict(valid)
    return predictions,{"training_rows":len(train),"training_groups":dict(counts),"specialists":eligible,
        "target_max":max(r["target_on"] for r in train),"origin_max":max(r["origin_on"] for r in train),"gap_models":len(fitted)}


def run(config_path, output):
    config_path,output=Path(config_path),Path(output)
    if output.exists():
        raise FileExistsError(output)
    cfg=json.loads(config_path.read_text())
    protocol=json.loads(Path(cfg["reference_protocol"]).read_text())
    source=Path(cfg.get("feature_source", "data/features-splits-full-v1"))
    event_root=Path(cfg.get("event_source", "data/purchase-events-full-v1"))
    for root,names in ((source,["all-origins-audit.csv"]),(event_root,["daily-events.csv","next-purchase-labels.csv"])):
        manifest=json.loads((root/"report.json").read_text())
        for name in names:
            if digest(root/name)!=manifest["files"][name]:
                raise ValueError("Frozen input changed")
    rows=read_csv(source/"all-origins-audit.csv")
    events=read_csv(event_root/"daily-events.csv")
    event_manifest=json.loads((event_root/"report.json").read_text())
    feature_manifest=json.loads((source/"report.json").read_text())
    history_start=cfg.get("expected_history_start", feature_manifest["split_config"]["start"])
    if (feature_manifest["split_config"]["start"] != history_start
            or event_manifest["first_ordered_on"] != history_start
            or any(r["origin_on"] < history_start for r in rows)
            or any(e["ordered_on"] < history_start for e in events)):
        raise ValueError("Features and event history must use the same declared start")
    if feature_manifest["source_event_files"]["daily-events.csv"] != digest(event_root/"daily-events.csv"):
        raise ValueError("Features belong to a different event dataset")
    excluded_customers=set(event_manifest["customer_exclusions"]["customers"])
    excluded_products={p for p,action in event_manifest["product_exclusions"]["products"].items() if action.startswith("exclude_")}
    if any(r["customer_id"] in excluded_customers or r["product_id"] in excluded_products for r in rows):
        raise ValueError("Excluded customer/product reappeared")
    started=time.monotonic()
    output.mkdir(parents=True)
    write_json(output/"config-frozen.json",cfg)
    additions=segment_features(events,cfg)
    if set(additions)!={r["origin_event_id"] for r in rows}:
        raise ValueError("Preprocessing changed eligible origin population")
    rows=[{**r,**additions[r["origin_event_id"]]} for r in rows]
    write_csv(output/"enriched-origins.csv",list(rows[0]),rows)
    print(json.dumps({"stage":"preprocessed","origins":len(rows),"customer_groups":dict(Counter(r["customer_group"] for r in rows))}),flush=True)
    labels={r["origin_event_id"]:r for r in read_csv(event_root/"next-purchase-labels.csv")}
    event_by_id={r["event_id"]:r for r in events}
    champion=base_champion_path()
    saved_champion=read_csv(champion/"validation-predictions.csv")
    champion_ids={r["origin_event_id"] for r in saved_champion}
    champion_report=json.loads((champion/"report.json").read_text())
    if digest(champion/"validation-predictions.csv") != champion_report["files"]["validation-predictions.csv"]:
        raise ValueError("Champion predictions changed")
    evaluation_mode=cfg.get("evaluation_mode", "mature90_and_champion_validation")
    if evaluation_mode not in {"mature90_and_champion_validation", "champion_validation_only"}:
        raise ValueError("Unknown evaluation mode")
    folds=(protocol["folds"] if evaluation_mode == "mature90_and_champion_validation" else [])
    folds=folds+[{"id":"champion_validation","train_end":"2026-07-13"}]
    records=[]
    training={}
    for fold in folds:
        original=fold["id"] == "champion_validation"
        if original:
            train=[r for r in rows if r["origin_on"]<=fold["train_end"] and r["label_state"]=="observed" and r["target_on"]<=fold["train_end"]]
            valid=[r for r in rows if r["origin_event_id"] in champion_ids]
            outcomes={r["origin_event_id"]:"observed" for r in valid}
            assert {r["origin_event_id"] for r in valid}==champion_ids
            baseline={r["origin_event_id"]:r for r in saved_champion}
            for r in valid:
                b=baseline[r["origin_event_id"]]
                assert all(r[k]==b[k] for k in ("customer_id","product_id","origin_on","target_on","target_gap_days","target_quantity"))
        else:
            train=training_rows(rows,fold["train_end"],protocol["training_window_days"],protocol["horizon_days"])
            valid=[r for r in rows if fold["origin_start"]<=r["origin_on"]<=fold["origin_end"]]
            outcomes={r["origin_event_id"]:horizon_outcome(r,90,"2026-09-14",labels,event_by_id) for r in valid}
        assert train and valid and "not_mature" not in outcomes.values()
        assert not {r["origin_event_id"] for r in train}&{r["origin_event_id"] for r in valid}
        assert max(r["target_on"] for r in train)<min(r["origin_on"] for r in valid)
        values,training[fold["id"]]=train_candidates(train,valid,cfg,output/"models"/fold["id"])
        if original:
            values["frozen_champion"]=[{"predicted_gap_days":int(baseline[r["origin_event_id"]]["predicted_gap_days"]),
                "cart_quantity":int(baseline[r["origin_event_id"]]["cart_quantity"]),"route":"frozen_champion"} for r in valid]
        for candidate,predictions in values.items():
            for row,prediction in zip(valid,predictions):
                outcome=outcomes[row["origin_event_id"]]
                record={k:row[k] for k in ("origin_event_id","customer_id","product_id","origin_on","customer_group","pair_group")}
                record.update(fold=fold["id"],candidate=candidate,outcome=outcome,
                    target_gap_days=row["target_gap_days"] if outcome=="observed" else "",
                    target_quantity=row["target_quantity"] if outcome=="observed" else "",**prediction)
                records.append(record)
        print(json.dumps({"stage":"trained","fold":fold["id"],**training[fold["id"]]}),flush=True)
    write_csv(output/"predictions.csv",list(records[0]),records)
    comparisons={}
    for section in ("mature90","champion_validation"):
        selected=[r for r in records if (r["fold"]=="champion_validation")==(section=="champion_validation")]
        comparisons[section]={}
        for candidate in sorted({r["candidate"] for r in selected}):
            rs=[r for r in selected if r["candidate"]==candidate]
            comparisons[section][candidate]={"overall":metrics(rs,cfg),
                "groups":{g:metrics([r for r in rs if r["customer_group"]==g],cfg) for g in GROUPS},
                "folds":{f:metrics([r for r in rs if r["fold"]==f],cfg) for f in sorted({r["fold"] for r in rs})}}
    report={"version":cfg["version"],"config":cfg,"comparisons":comparisons,"training":training,
        "preprocessing":{"origins_before":len(rows),"origins_after":len(additions),
            "history_start":history_start,"history_end":event_manifest["last_ordered_on"],
            "customer_groups":dict(Counter(r["customer_group"] for r in rows)),
            "pair_groups":dict(Counter(r["pair_group"] for r in rows)),
            "source_policy_version":event_manifest["policy_version"],
            "customer_exclusions_sha256":event_manifest["customer_exclusions_sha256"],
            "product_exclusions_sha256":event_manifest["product_exclusions_sha256"]},
        "provenance":{"config_sha256":digest(config_path),"code_sha256":digest(__file__),
            "features_code_sha256":digest("scripts/purchase_segment_features.py"),
            "source_sha256":digest(source/"all-origins-audit.csv"),"events_sha256":digest(event_root/"daily-events.csv"),
            "labels_sha256":digest(event_root/"next-purchase-labels.csv"),"champion_report_sha256":digest(champion/"report.json"),
            "champion_predictions_sha256":digest(champion/"validation-predictions.csv"),"xgboost_version":xgb.__version__},
        "duration_seconds":time.monotonic()-started,"operational_approval":False,"independent_final_test":False,
        "limitations":["Customer types are rules recomputed at each origin, not permanent identity or future labels",
            "Item speed is completed repurchase intervals, not measured depletion; repeated buyers contribute more intervals",
            "Latest-state snapshot historical availability unverified; previously consumed development cohorts",
            "mature90 and original boundary-limited champion validation must not be pooled",
            "Cart results assume daily execution from next midnight; same-day is not early",
            "Quantile .25 does not guarantee calibrated probabilities or zero lateness",
            "Non-return and held outcomes not used as zero-day regression labels; scored separately"],
        "files":{str(p.relative_to(output)):digest(p) for p in output.rglob("*") if p.is_file()}}
    write_json(output/"report.json",report)
    print(json.dumps({"stage":"completed","duration_seconds":report["duration_seconds"],"output":str(output)}),flush=True)
    return report


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path("config/purchase-segments-v1.json"))
    parser.add_argument("--output",type=Path,default=Path("artifacts/purchase-segments-v1"))
    args=parser.parse_args()
    run(args.config,args.output)
