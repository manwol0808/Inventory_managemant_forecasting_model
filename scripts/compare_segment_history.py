"""Paired full/short history comparison on the identical consumed validation cohort."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_baseline import digest


SOURCES={"full":"artifacts/purchase-segments-v1", "full_controls":"artifacts/purchase-segments-controls-v1",
         "short":"artifacts/purchase-segments-short-v1"}


def compare():
    data={}
    for scope,path in SOURCES.items():
        root=Path(path)
        report=json.loads((root/"report.json").read_text())
        assert digest(root/"predictions.csv")==report["files"]["predictions.csv"]
        rows=pd.read_csv(root/"predictions.csv",dtype={"customer_id":str,"product_id":str})
        rows=rows[rows.fold.eq("champion_validation")]
        for candidate,part in rows.groupby("candidate"):
            if scope=="full_controls" and candidate=="frozen_champion":
                continue
            key=("full" if scope=="full_controls" else scope,candidate)
            data[key]=part.set_index("origin_event_id").sort_index()
    reference=data['short','frozen_champion']
    customer_codes,customers=pd.factorize(reference.customer_id)
    rng=np.random.default_rng(20260917)
    n_customers=len(customers)
    # Shared paired customer-cluster draws preserve within-customer dependence.
    weights=rng.multinomial(n_customers,np.full(n_customers,1/n_customers),size=2000)
    denominator=weights@np.bincount(customer_codes,minlength=n_customers)
    def flags(part):
        gap=part.predicted_gap_days.to_numpy(dtype=int)
        actual=part.target_gap_days.to_numpy(dtype=int)
        cart=np.maximum(1,gap-np.minimum(7,np.maximum(1,(gap+3)//4)))
        margin=actual-cart
        return {"window":(margin>=0)&(margin<=10),"late":margin<0,"same_day":margin==0,
                "too_early_over10":margin>10,"too_early_over30":margin>30,
                "date7":np.abs(gap-actual)<=7,
                "quantity2":np.abs(part.cart_quantity.to_numpy(dtype=int)-part.target_quantity.to_numpy(dtype=int))<=2}
    baseline=flags(reference)
    results=[]
    for (scope,candidate),part in sorted(data.items()):
        assert part.index.equals(reference.index)
        for field in ("customer_id","product_id","origin_on","target_gap_days","target_quantity"):
            assert np.array_equal(part[field].to_numpy(),reference[field].to_numpy()),field
        values=flags(part)
        stats={name:{"n":int(value.sum()),"rate":float(value.mean())} for name,value in values.items()}
        ci={}
        for name in ("window","late","too_early_over10"):
            diff=values[name].astype(int)-baseline[name].astype(int)
            totals=np.bincount(customer_codes,weights=diff,minlength=n_customers)
            sampled=100*(weights@totals)/denominator
            ci[name]={"delta_pp":float(diff.mean()*100),"ci95_pp":np.quantile(sampled,[.025,.975]).tolist()}
        results.append({"history":scope,"candidate":candidate,"n":len(part),"metrics":stats,
                        "vs_frozen_champion_paired_customer_bootstrap":ci})
    full_cfg=json.loads(Path(SOURCES['full']+'/config-frozen.json').read_text())
    short_cfg=json.loads(Path(SOURCES['short']+'/config-frozen.json').read_text())
    for field in ('params','rounds','minimum_gaps','recent_gaps','stable_cv_max','trend_relative_change_min',
                  'trend_direction_fraction_min','product_recent_intervals','minimum_product_intervals',
                  'minimum_product_customers','minimum_specialist_rows','minimum_specialist_customers','cart_policies'):
        assert full_cfg[field]==short_cfg[field],field
    return {"version":"segment-history-comparison-v1","validation_n":len(reference),"customers":n_customers,
            "cart_policy":"relative25_cap7; at earliest next day; window includes actual repurchase day",
            "results":results,"provenance":{root+'/report.json':digest(Path(root)/'report.json') for root in SOURCES.values()},
            "limits":["Same reused, boundary-limited validation; not fresh prospective accuracy",
                      "Bootstrap 2000 paired customer clusters, seed20260917; exploratory intervals not multiple-comparison adjusted",
                      "Customer grouping changes with history length; only overall same-cohort metrics are compared",
                      "No stockout ground truth or operational auto-promotion"]}


if __name__=="__main__":
    print(json.dumps(compare(),ensure_ascii=False,indent=2))
