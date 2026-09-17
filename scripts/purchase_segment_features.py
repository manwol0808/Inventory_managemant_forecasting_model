"""Chronological customer/pair cadence profiles and observed item interval features."""
from collections import defaultdict, deque
from datetime import date
from statistics import mean, median, pstdev

import numpy as np

from scripts.build_features_and_splits import FEATURE_COLUMNS


GROUPS = ("insufficient", "irregular", "rebound", "speeding_up", "slowing_down", "stable")
PROFILE_FIELDS = ("count", "last", "median", "cv", "relative_change", "down_fraction", "up_fraction")
PATTERN_FEATURES = [f"{unit}_{field}" for unit in ("customer", "pair") for field in PROFILE_FIELDS]
PATTERN_FEATURES += [f"{unit}_is_{group}" for unit in ("customer", "pair") for group in GROUPS]
PATTERN_FEATURES += [f"pair_gap_lag{i}" for i in range(1, 7)]
ITEM_FEATURES = ["item_gap_count", "item_interval_customers", "item_speed_available", "item_gap_median", "item_gap_q25", "item_gap_cv",
                 "item_days_per_previous_unit", "item_fast", "item_medium", "item_slow",
                 "pair_to_item_gap_ratio"]


def profile(gaps, cfg):
    recent = list(gaps)[-cfg["recent_gaps"]:]
    if any(g <= 0 for g in recent):
        raise ValueError("Gaps must be positive")
    n = len(recent)
    typical = median(recent) if recent else None
    changes = [b-a for a,b in zip(recent, recent[1:])]
    relative = (recent[-1]-recent[0])/typical if recent else None
    cv = pstdev(recent)/mean(recent) if recent else None
    down = sum(d < 0 for d in changes)/len(changes) if changes else 0
    up = sum(d > 0 for d in changes)/len(changes) if changes else 0
    group = "insufficient"
    if n >= cfg["minimum_gaps"]:
        threshold = cfg["trend_relative_change_min"]
        direction = cfg["trend_direction_fraction_min"]
        trough = recent.index(min(recent))
        rebound = (n >= 5 and 1 < trough < n-2
                   and (recent[0]-recent[trough])/typical >= threshold
                   and (recent[-1]-recent[trough])/typical >= threshold
                   and all(b <= a for a,b in zip(recent[:trough],recent[1:trough+1]))
                   and all(b >= a for a,b in zip(recent[trough:-1],recent[trough+1:])))
        if rebound:
            group = "rebound"
        elif relative <= -threshold and down >= direction:
            group = "speeding_up"
        elif relative >= threshold and up >= direction:
            group = "slowing_down"
        elif cv <= cfg["stable_cv_max"]:
            group = "stable"
        else:
            group = "irregular"
    return {"group":group, "count":n, "last":recent[-1] if recent else "",
            "median":typical if recent else "", "cv":cv if recent else "",
            "relative_change":relative if recent else "", "down_fraction":down, "up_fraction":up}


def segment_features(events, cfg):
    """No labels accepted. Entire purchase-day batch is known at next midnight.

    Holds reset pair history and conservatively reset customer visit history.
    Item statistics only contain intervals whose second completed event has arrived.
    Same-day interval updates are batched and order independent.
    """
    batches = defaultdict(list)
    ids = set()
    for e in events:
        if e["event_id"] in ids:
            raise ValueError("Duplicate event ID")
        ids.add(e["event_id"])
        if e["event_state"] in {"completed", "held"}:
            batches[e["ordered_on"]].append(e)
    pairs = defaultdict(lambda: deque(maxlen=cfg["recent_gaps"]+1))
    visits = defaultdict(lambda: deque(maxlen=cfg["recent_gaps"]+1))
    item_intervals = defaultdict(lambda: deque(maxlen=cfg["product_recent_intervals"]))
    result = {}
    for day, batch in sorted(batches.items()):
        today = date.fromisoformat(day)
        batch = sorted(batch, key=lambda e:e["event_id"])
        customers = defaultdict(list)
        pair_keys = set()
        for e in batch:
            key = e["customer_id"],e["product_id"]
            if key in pair_keys:
                raise ValueError("Duplicate customer-product day")
            pair_keys.add(key)
            customers[e["customer_id"]].append(e)
            if e["event_state"] == "held":
                pairs[key].clear()
            else:
                qty = int(e["quantity"])
                if qty <= 0:
                    raise ValueError("Nonpositive quantity")
                h = pairs[key]
                if h:
                    gap = (today-h[-1][0]).days
                    if gap <= 0:
                        raise ValueError("Nonpositive pair gap")
                    item_intervals[e["product_id"]].append((gap,gap/h[-1][1],e["customer_id"]))
                h.append((today,qty))
        for customer, es in customers.items():
            if any(e["event_state"] == "held" for e in es):
                visits[customer].clear()
            if any(e["event_state"] == "completed" for e in es):
                visits[customer].append(today)
        item_stats = {}
        for product in {e["product_id"] for e in batch if e["event_state"] == "completed"}:
            intervals = list(item_intervals[product])
            gaps = [g for g,_,_ in intervals]
            customer_count = len({c for _,_,c in intervals})
            enough = (len(gaps) >= cfg["minimum_product_intervals"]
                      and customer_count >= cfg["minimum_product_customers"])
            typical = median(gaps) if enough else None
            item_stats[product] = {"item_gap_count":len(gaps), "item_interval_customers":customer_count,
                "item_speed_available":int(enough),
                "item_gap_median":typical if enough else "",
                "item_gap_q25":float(np.quantile(gaps,.25)) if enough else "",
                "item_gap_cv":pstdev(gaps)/mean(gaps) if enough else "",
                "item_days_per_previous_unit":median(x for _,x,_ in intervals) if enough else "",
                "item_fast":int(enough and typical <= 7),
                "item_medium":int(enough and 7 < typical <= 30),
                "item_slow":int(enough and typical > 30)}
        for e in batch:
            if e["event_state"] != "completed":
                continue
            h = list(pairs[e["customer_id"],e["product_id"]])
            v = list(visits[e["customer_id"]])
            pair_gaps = [(b[0]-a[0]).days for a,b in zip(h,h[1:])]
            customer_gaps = [(b-a).days for a,b in zip(v,v[1:])]
            features = dict(item_stats[e["product_id"]])
            for unit,gaps in (("customer",customer_gaps),("pair",pair_gaps)):
                p = profile(gaps,cfg)
                features[unit+"_group"] = p["group"]
                features.update({f"{unit}_{field}":p[field] for field in PROFILE_FIELDS})
                features.update({f"{unit}_is_{g}":int(p["group"] == g) for g in GROUPS})
            features.update({f"pair_gap_lag{i}":pair_gaps[-i] if len(pair_gaps)>=i else "" for i in range(1,7)})
            item_gap = features["item_gap_median"]
            features["pair_to_item_gap_ratio"] = median(pair_gaps)/item_gap if pair_gaps and item_gap != "" else ""
            result[e["event_id"]] = features
    return result


def feature_names(mode):
    if mode not in {"base","patterns","all"}:
        raise ValueError("Unknown feature mode")
    return FEATURE_COLUMNS + (PATTERN_FEATURES if mode != "base" else []) + (ITEM_FEATURES if mode == "all" else [])
