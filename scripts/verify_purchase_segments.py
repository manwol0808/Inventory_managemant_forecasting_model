"""Replay saved predictions and independently recompute descriptive metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_baseline import digest
from scripts.run_purchase_segments import SegmentPredictor


def verify(root):
    root=Path(root)
    report=json.loads((root/"report.json").read_text())
    for name,expected in report["files"].items():
        assert digest(root/name)==expected,(root,name)
    features=pd.read_csv(root/"enriched-origins.csv",dtype=str,keep_default_na=False).set_index("origin_event_id",drop=False)
    predictions=pd.read_csv(root/"predictions.csv",dtype=str,keep_default_na=False)
    event_source=Path(report["config"].get("event_source", "data/purchase-events-full-v1"))
    labels=pd.read_csv(event_source/"next-purchase-labels.csv",dtype=str,keep_default_na=False).set_index("origin_event_id")
    replayed=0
    for (fold,candidate),part in predictions.groupby(["fold","candidate"]):
        assert not part.origin_event_id.duplicated().any()
        if candidate=="frozen_champion":
            continue
        rows=features.loc[part.origin_event_id].to_dict("records")
        # Drop every target/metadata field except the origin-time model contract.
        predictor=SegmentPredictor(root/"models"/fold/candidate)
        names=next(iter(predictor.models.values()))[0]["features"]
        inputs=[{**{k:r[k] for k in names},"customer_group":r["customer_group"]} for r in rows]
        output=predictor.predict(inputs)
        assert [r["predicted_gap_days"] for r in output]==part.predicted_gap_days.astype(int).tolist()
        assert [r["cart_quantity"] for r in output]==part.cart_quantity.astype(int).tolist()
        assert [r["route"] for r in output]==part.route.tolist()
        replayed+=len(output)
        for ref in set(predictor.router["models"].values()):
            metadata=json.loads((root/"models"/fold/ref/"training.json").read_text())
            train=features.loc[metadata["origin_ids"]]
            assert train.target_on.max()<part.origin_on.min()
            assert set(metadata["origin_ids"]).isdisjoint(part.origin_event_id)
    observed=predictions[predictions.outcome.eq("observed")]
    truth=labels.loc[observed.origin_event_id]
    for field in ("target_gap_days","target_quantity","customer_id","product_id"):
        assert observed[field].tolist()==truth[field].tolist()
    checked=0
    for section,candidates in report["comparisons"].items():
        pool=predictions[predictions.fold.eq("champion_validation")== (section=="champion_validation")]
        for candidate,reported in candidates.items():
            rs=pool[pool.candidate.eq(candidate)]
            obs=rs[rs.outcome.eq("observed")]
            pred=obs.predicted_gap_days.astype(int).to_numpy()
            actual=obs.target_gap_days.astype(int).to_numpy()
            m=reported["overall"]
            assert m["n"]==len(obs) and m["origins"]==len(rs)
            assert abs(m["date_mae"]-np.abs(pred-actual).mean())<1e-10
            assert abs(m["date_within_7_rate"]-(np.abs(pred-actual)<=7).mean())<1e-10
            qerr=np.abs(obs.cart_quantity.astype(int).to_numpy()-obs.target_quantity.astype(int).to_numpy())
            assert abs(m["quantity_within_2_rate"]-(qerr<=2).mean())<1e-10
            assert sum(g["n"] for g in reported["groups"].values())==len(obs)
            for policy,p in m["cart_policies"].items():
                lead={"none":0,"lead7":7,"lead10":10}.get(policy)
                offsets=np.minimum(7,np.maximum(1,(pred+3)//4)) if lead is None else lead
                margin=actual-np.maximum(1,pred-offsets)
                for name,count in {"early":(margin>0).sum(),"same_day":(margin==0).sum(),"late":(margin<0).sum(),
                                   "early_1_to_10":((margin>=1)&(margin<=10)).sum(),"early_over_30":(margin>30).sum(),
                                   "at_least_7_days":(margin>=7).sum()}.items():
                    assert p[name]==count,(section,candidate,policy,name)
                checked+=1
    return {"root":str(root),"report_sha256":digest(root/"report.json"),"replayed_predictions":replayed,
            "checked_policy_summaries":checked,"raw_observed_label_matches":len(observed),
            "file_hashes_verified":len(report["files"]),"status":"passed"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots",type=Path,nargs="+")
    args=parser.parse_args()
    print(json.dumps({"verification":[verify(root) for root in args.roots]},indent=2))
