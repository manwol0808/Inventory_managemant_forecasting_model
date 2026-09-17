"""Trace deterministic real examples from predictions through labels to DB export rows."""
import json
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from scripts.build_features_and_splits import read_csv
from scripts.purchase_segment_features import profile
from scripts.run_baseline import digest
from scripts.run_purchase_segments import scheduled_gap


def build():
    cfg=json.loads(Path("config/purchase-segments-v1.json").read_text())
    root=Path("artifacts/purchase-segments-v1")
    features={r["origin_event_id"]:r for r in read_csv(root/"enriched-origins.csv")}
    records=read_csv(root/"predictions.csv")
    candidates={(r["origin_event_id"],r["candidate"]):r for r in records if r["fold"]=="champion_validation"}
    groups=("irregular","stable","rebound","speeding_up")
    selected=[]
    for group in groups:
        eligible=sorted((r for r in records if r["fold"]=="champion_validation" and r["candidate"]=="patterns_items_q25"
                         and r["customer_group"]==group and int(features[r["origin_event_id"]]["pair_count"])>=3),
                        key=lambda r:(r["origin_on"],r["origin_event_id"]))
        customers=set()
        for row in eligible:
            if row["customer_id"] not in customers:
                selected.append(row)
                customers.add(row["customer_id"])
                if len(customers)==2:
                    break
    events=read_csv("data/purchase-events-full-v1/daily-events.csv")
    by_customer=defaultdict(list)
    by_event={e["event_id"]:e for e in events}
    for e in events:
        by_customer[e["customer_id"]].append(e)
    labels={r["origin_event_id"]:r for r in read_csv("data/purchase-events-full-v1/next-purchase-labels.csv")}
    links=defaultdict(list)
    for link in read_csv("data/purchase-events-full-v1/source-links.csv"):
        if link["event_quantity_contribution"] and int(link["event_quantity_contribution"])>0:
            links[link["event_id"]].append(link)
    raw={str(i):r for i,r in enumerate(read_csv("data/full-history-v1/orders.csv"),1)}
    examples=[]
    all_orders=set()
    for row in selected:
        customer_days=[]
        pair=[]
        batches=defaultdict(list)
        for e in by_customer[row["customer_id"]]:
            if e["ordered_on"]<=row["origin_on"] and e["event_state"] in {"completed","held"}:
                batches[e["ordered_on"]].append(e)
        for day,batch in sorted(batches.items()):
            if any(e["event_state"]=="held" for e in batch):
                customer_days=[]
            if any(e["event_state"]=="completed" for e in batch):
                customer_days.append(day)
            for e in batch:
                if e["product_id"]==row["product_id"]:
                    if e["event_state"]=="held":
                        pair=[]
                    else:
                        pair.append(e)
        customer_days=customer_days[-7:]
        gaps=[(date.fromisoformat(b)-date.fromisoformat(a)).days for a,b in zip(customer_days,customer_days[1:])]
        assert profile(gaps,cfg)["group"]==row["customer_group"]
        pair=pair[-7:]
        label=labels[row["origin_event_id"]]
        target=by_event[label["target_event_id"]]
        assert target["ordered_on"]==label["target_on"]
        assert (date.fromisoformat(target["ordered_on"])-date.fromisoformat(row["origin_on"])).days==int(row["target_gap_days"])
        trace=[]
        for e in pair+[target]:
            contributions=[]
            for link in links[e["event_id"]]:
                source=raw[link["source_record_number"]]
                assert (source["customer_id"],source["product_id"],source["order_id"],source["ordered_on"]) == (row["customer_id"],row["product_id"],link["order_id"],e["ordered_on"])
                all_orders.add(source["order_id"])
                contributions.append({"order_id":source["order_id"],"order_section_item_no":source["order_section_item_no"],
                    "source_record_number":link["source_record_number"],"quantity":int(link["event_quantity_contribution"]),
                    "product_name":source["product_name"],"section_status":source["section_status"]})
            assert sum(r["quantity"] for r in contributions)==int(e["quantity"])
            trace.append({"event_id":e["event_id"],"ordered_on":e["ordered_on"],"quantity":int(e["quantity"]),"orders":contributions})
        forecasts={}
        for name in ("frozen_champion","patterns_items_q25","specialists_q25"):
            p=candidates[row["origin_event_id"],name]
            gap=int(p["predicted_gap_days"])
            cart=date.fromisoformat(row["origin_on"])+timedelta(days=scheduled_gap(gap,"relative25_cap7"))
            margin=(date.fromisoformat(label["target_on"])-cart).days
            forecasts[name]={"gap_days":gap,"predicted_on":(date.fromisoformat(row["origin_on"])+timedelta(days=gap)).isoformat(),
                "cart_on":cart.isoformat(),"actual_minus_cart_days":margin,"inside_actual_minus10_to_day":0<=margin<=10,
                "cart_quantity":int(p["cart_quantity"]),"route":p["route"]}
        pair_days=[e["ordered_on"] for e in pair]
        examples.append({"group":row["customer_group"],"customer_id":row["customer_id"],"product_id":row["product_id"],
            "product_name":trace[-2]["orders"][0]["product_name"],"origin_event_id":row["origin_event_id"],"origin_on":row["origin_on"],
            "customer_recent_purchase_days":customer_days,"customer_recent_gaps":gaps,
            "pair_recent_purchase_days":pair_days,"pair_recent_gaps":[(date.fromisoformat(b)-date.fromisoformat(a)).days for a,b in zip(pair_days,pair_days[1:])],
            "actual_next_gap_days":int(label["target_gap_days"]),"actual_next_on":label["target_on"],"actual_next_quantity":int(label["target_quantity"]),
            "forecasts":forecasts,"pair_event_trace":trace})
    output=Path("artifacts/purchase-segment-examples-v1")
    output.mkdir(exist_ok=False)
    result={"selection":"First two different customers per group, ordered by origin_on and origin_event_id, in common champion validation with at least 3 past item intervals. No prediction-error filter. Illustrations, not group performance estimates.",
            "source":"Frozen BigQuery order export, not fabricated events; live DB recheck recorded separately",
            "examples":examples,"provenance":{p:digest(p) for p in ("data/full-history-v1/orders.csv","data/purchase-events-full-v1/source-links.csv","data/purchase-events-full-v1/next-purchase-labels.csv",str(root/"predictions.csv"))}}
    (output/"examples.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    # Fixed known source table, exact order IDs traced above. Read-only query, no identities/names beyond needed order fields.
    ids=", ".join("'"+oid+"'" for oid in sorted(all_orders))
    assert all(oid.isdigit() for oid in all_orders)
    query=f"""WITH latest AS (
  SELECT order_no, member_code, wtime, updated_at, sections,
    ROW_NUMBER() OVER (PARTITION BY order_no ORDER BY updated_at DESC, TO_JSON_STRING(sections) DESC,
      CAST(member_code AS STRING) DESC, wtime DESC, total_price DESC, total_payment_price DESC) AS rn
  FROM `manwol-core.manwol_core_mirror.public_imweb_orders`
  WHERE channel = 'b2s' AND CAST(order_no AS STRING) IN ({ids})
)
SELECT CAST(order_no AS STRING) AS order_id, CAST(member_code AS STRING) AS customer_id,
  CAST(DATE(wtime, 'Asia/Seoul') AS STRING) AS ordered_on,
  JSON_VALUE(i, '$.orderSectionItemNo') AS order_section_item_no,
  JSON_VALUE(i, '$.productInfo.prodNo') AS product_id,
  JSON_VALUE(i, '$.productInfo.prodName') AS product_name,
  SAFE_CAST(JSON_VALUE(i, '$.qty') AS INT64) AS quantity,
  JSON_VALUE(s, '$.orderSectionStatus') AS section_status
FROM latest CROSS JOIN UNNEST(JSON_QUERY_ARRAY(sections)) s
CROSS JOIN UNNEST(JSON_QUERY_ARRAY(s, '$.sectionItems')) i
WHERE rn=1
ORDER BY order_id, order_section_item_no
"""
    (output/"live-check.sql").write_text(query)
    print(json.dumps({"examples":len(examples),"orders_to_recheck":len(all_orders),"output":str(output)}))


if __name__=="__main__":
    build()
