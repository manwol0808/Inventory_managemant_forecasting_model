"""Cadence-type router: TabPFN for speeding-up/slowing-down pairs, champion for the rest.

Each customer-product origin is routed by its past-only cadence type with hysteresis.
The champion covers missing, uncertain or later agent predictions. Exploratory only.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import date, datetime, time as clock, timedelta
from pathlib import Path

import numpy as np
import xgboost as xgb

from scripts.build_features_and_splits import next_midnight, read_csv
from scripts.build_purchase_events import write_csv
from scripts.predict_replenishment import KOREA, base_champion_path, weekly_cart_schedule
from scripts.purchase_segment_features import GROUPS, feature_names
from scripts.run_baseline import digest, write_json
from scripts.run_purchase_segments import design_matrix, metrics, schema_for
from scripts.run_tabpfn_experiment import sample_context
from scripts.run_timing_experiment import horizon_outcome, rounded, training_rows
from scripts.train_xgboost import fit_schema, matrix, postprocess


MODEL_AGENTS = {"stable_xgboost": "xgboost", "speeding_up_tabpfn": "tabpfn", "slowing_down_tabpfn": "tabpfn",
                "rebound_tabpfn": "tabpfn", "speeding_up_xgboost": "xgboost", "rebound_xgboost": "xgboost",
                "xgboost": "xgboost", "tabpfn": "tabpfn"}
FEATURE_MODE = "all"


def excluded_from_training(row, cfg):
    """Exceptional period: drop training rows whose purchase or repurchase falls inside it."""
    window = cfg.get("training_exclude")
    return bool(window) and any(window["start"] <= row[k] <= window["end"] for k in ("origin_on", "target_on") if row[k])


def route_key(row, unit):
    return (row["customer_id"], row["product_id"]) if unit == "pair" else row["customer_id"]


def stable_routes(rows, cfg):
    """Past-only hysteresis per unit: switch when the new type fills enough of the recent window.

    Leaving or entering `insufficient` is immediate; there is no earlier type to protect.
    """
    unit = cfg["route_unit"]
    if unit not in {"pair", "customer"}:
        raise ValueError("Unknown route unit")
    series = defaultdict(dict)
    for row in rows:
        raw = row[unit + "_group"]
        if series[route_key(row, unit)].setdefault(row["origin_on"], raw) != raw:
            raise ValueError("Conflicting same-day cadence type")
    state = {}
    for key, days in series.items():
        current, seen = None, []
        for day in sorted(days):
            raw = days[day]
            seen.append(raw)
            window = seen[-cfg["switch_window"]:]
            previous = current
            if raw == "insufficient" or current in (None, "insufficient"):
                current = raw
            elif raw != current and window.count(raw) >= cfg["switch_min_votes"]:
                current = raw
            state[key, day] = {"customer_group": current, "raw_group": raw,
                               "route_confidence": round(window.count(current)/len(window), 3),
                               "route_changed": previous is not None and current != previous}
    return {r["origin_event_id"]: state[route_key(r, unit), r["origin_on"]] for r in rows}


def purchase_rule(row, group, cfg):
    """(stage, suggest, fixed gap) from how often this pair was bought since its last hold.

    First purchases get no cart; the second reuses its single observed gap.
    """
    purchases = int(row["pair_count"])+1
    if purchases == 1:
        return "first", not cfg["skip_first_purchase"], None
    if purchases == 2:
        return "second", True, rounded(row["pair_gap_lag1"]) if cfg["second_purchase_last_gap"] else None
    return ("third_fourth" if group == "insufficient" else "cadence"), True, None


def guard(agent_gap, champion_gap, relative_iqr, cfg, missing_reason):
    """Return (gap, fallback reason). The champion is used when the agent is missing, uncertain or (if guarded) later."""
    if agent_gap is None:
        return champion_gap, missing_reason
    if relative_iqr is not None and relative_iqr > cfg["max_relative_iqr"]:
        return champion_gap, "high_uncertainty"
    if cfg["late_guard"] and agent_gap > champion_gap:
        return champion_gap, "later_than_champion"
    return agent_gap, None


def weekly_cart(origin_on, gap, release_hour, weekly_buyer_max_gap=0):
    """Predicted day, ordering day and release time: Sunday night before the prior week's Monday.

    Weekly buyers (gap <= weekly_buyer_max_gap) get the next Sunday-night cart instead of a
    mid-week catch-up; Monday availability is already the ordering window.
    """
    predicted = date.fromisoformat(origin_on)+timedelta(days=gap)
    available = next_midnight(origin_on)
    if gap <= weekly_buyer_max_gap:
        sunday = available.date()+timedelta(days=(6-available.weekday()) % 7)
        cart_at = available if available.weekday() == 0 else datetime.combine(sunday, clock(release_hour), KOREA)
    else:
        _, cart_at = weekly_cart_schedule(predicted.isoformat(), available, release_hour)
    order_day = cart_at.date()+timedelta(days=1 if cart_at.weekday() == 6 else 0)
    return predicted.isoformat(), order_day.isoformat(), cart_at.isoformat()


def weekly_metrics(records, weekly_buyer_max_gap=0):
    """Margins from cart release date to the actual repurchase date; same day is not early.

    Non-returners within the horizon count as unnecessary carts when released inside it.
    """
    observed = [r for r in records if r.get("outcome", "observed") == "observed"]
    absent = [r for r in records if r.get("outcome") == "no_repurchase_within_horizon"]
    margins = [(date.fromisoformat(r["target_on"])-date.fromisoformat(r["cart_at"][:10])).days for r in observed]
    n = len(margins)
    if not n:
        return {"n": 0}
    late_by_customer = defaultdict(list)
    for row, margin in zip(observed, margins):
        late_by_customer[row["customer_id"]].append(margin < 0)
    counts = {"late": sum(m < 0 for m in margins), "same_day": sum(m == 0 for m in margins),
              "early_1_to_10": sum(1 <= m <= 10 for m in margins), "at_least_7_days": sum(m >= 7 for m in margins),
              "early_over_30": sum(m > 30 for m in margins),
              "sunday_or_monday": sum(date.fromisoformat(r["cart_at"][:10]).weekday() in (6, 0) for r in observed),
              "weekly_buyer": sum(int(r["predicted_gap_days"]) <= weekly_buyer_max_gap for r in observed)}
    unnecessary = sum((date.fromisoformat(r["cart_at"][:10])-date.fromisoformat(r["origin_on"])).days <= 90 for r in absent)
    return {"n": n, **counts, **{k+"_rate": v/n for k, v in counts.items()},
            "customer_macro_late_rate": float(np.mean([np.mean(v) for v in late_by_customer.values()])),
            "median_margin_days": float(np.median(margins)),
            "nonreturn_n": len(absent), "nonreturn_cart_within90_rate": unnecessary/len(absent) if absent else None}


def monitor_metrics(records, days):
    """Primary score: repurchases within the monitoring window predicted within 7 days.

    Cases without a purchase inside the window are closed as misses for the all-origin rate.
    """
    scored = [r for r in records if r.get("outcome", "observed") != "unknown_due_to_hold"]
    within = [r for r in scored if r.get("outcome", "observed") == "observed" and int(r["target_gap_days"]) <= days]
    hits = sum(abs(int(r["predicted_gap_days"])-int(r["target_gap_days"])) <= 7 for r in within)
    return {"days": days, "origins": len(scored), "repurchased": len(within), "hits": hits,
            "hit_rate": hits/len(within) if within else None,
            "hit_rate_all_origins": hits/len(scored) if scored else None,
            "closed_without_purchase_rate": 1-len(within)/len(scored) if scored else None}


def feature_array(rows, schema):
    return np.asarray([[schema["products"].get(r[k], np.nan) if k == "product_id"
                        else float(r[k]) if r[k] != "" else np.nan for k in schema["features"]] for r in rows],
                      dtype=np.float32)


def train_xgboost(train, rows, cfg, folder):
    schema = schema_for(train, FEATURE_MODE)
    write_json(folder/"schema.json", schema)
    dm = design_matrix(train, schema, "target_gap_days")
    gap = xgb.train({**cfg["xgb_params"], "objective": "reg:quantileerror", "quantile_alpha": cfg["quantiles"]},
                    dm, cfg["xgb_rounds"])
    dm.set_label(np.asarray([float(r["target_quantity"]) for r in train]))
    quantity = xgb.train({**cfg["xgb_params"], "objective": "reg:absoluteerror"}, dm, cfg["xgb_rounds"])
    gap.save_model(folder/"gap.json")
    quantity.save_model(folder/"quantity.json")
    features = design_matrix(rows, schema)
    return gap.predict(features), [rounded(q) for q in quantity.predict(features)]


def run_tabpfn(x_train, y_train, x_valid, worker_cfg, weights, folder):
    """Gap quantiles from a separate process; XGBoost and PyTorch crash when sharing one."""
    np.savez(folder/"input.npz", x_train=x_train, y_train=y_train, x_valid=x_valid)
    write_json(folder/"worker.json", worker_cfg)
    weights = Path(weights).resolve()
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TABPFN_MODEL_CACHE_DIR=str(weights.parent),
               SKB_DATA_DIRECTORY=str(Path("artifacts/cache/skrub").resolve()),
               XDG_CACHE_HOME=str(Path("artifacts/cache").resolve()),
               MPLCONFIGDIR=str(Path("artifacts/cache/matplotlib").resolve()))
    subprocess.run([sys.executable, "-X", "faulthandler", "-m", "scripts.tabpfn_worker",
                    "--input", str(folder/"input.npz"), "--weights", str(weights),
                    "--config", str(folder/"worker.json"), "--output", str(folder/"gap.npy")], check=True, env=env)
    values = np.load(folder/"gap.npy", allow_pickle=False)
    if values.shape != (len(x_valid), len(worker_cfg["quantiles"])):
        raise ValueError("Unexpected TabPFN quantile shape")
    return values


def tabpfn_quantiles(train, rows, cfg, folder):
    tab = cfg["tabpfn"]
    schema = schema_for(train, FEATURE_MODE)
    write_json(folder/"schema.json", schema)
    context = sample_context(train, tab["context_rows"])
    worker = {**{k: v for k, v in tab.items() if k not in {"weights", "context_rows"}},
              "output_type": "quantiles", "quantiles": cfg["quantiles"], "feature_count": len(schema["features"])}
    values = run_tabpfn(feature_array(context, schema), np.asarray([float(r["target_gap_days"]) for r in context]),
                        feature_array(rows, schema), worker, tab["weights"], folder)
    # Quantity stays a transparent recent-median rule on this branch.
    return values, [rounded(r["qty_median_last3"]) for r in rows], len(context)


def champion_recipe(train, valid, folder):
    """Stand-in for folds before the champion existed: its features, objectives, params and rounds, refit here."""
    report = json.loads((base_champion_path()/"report.json").read_text())
    params, targets = report["config"]["params"], report["config"]["targets"]
    schema = fit_schema(train)
    models = {}
    for target in ("gap", "quantity"):
        models[target] = xgb.train({**params, "objective": targets[target]["objective"]},
                                   matrix(train, schema, targets[target]["label"]),
                                   report["training"][target]["selected_rounds"])
        models[target].save_model(folder/f"champion-recipe-{target}.json")
    dm = matrix(valid, schema)
    _, carts, gaps = postprocess(models["quantity"].predict(dm), models["gap"].predict(dm))
    return {r["origin_event_id"]: (int(g), int(c)) for r, g, c in zip(valid, gaps, carts)}


def route_fold(train, valid, outcomes, fallback, routing, cfg, folder, fold_id):
    """Train routed agents on `train`, predict `valid`; `fallback` maps origin id to champion (gap, quantity)."""
    point = cfg["quantiles"].index(cfg["point_quantile"])
    train_by_group, valid_by_group = defaultdict(list), defaultdict(list)
    for r in train:
        train_by_group[routing[r["origin_event_id"]]["customer_group"]].append(r)
    for r in valid:
        valid_by_group[routing[r["origin_event_id"]]["customer_group"]].append(r)
    predictions, training = {}, {}
    for group, agent in cfg["routes"].items():
        subset, pending = train_by_group[group], valid_by_group[group]
        customers = len({r["customer_id"] for r in subset})
        info = training[group] = {"agent": agent, "rows": len(subset), "customers": customers,
                                  "validation_rows": len(pending)}
        if agent not in MODEL_AGENTS:
            continue
        info["trained"] = len(subset) >= cfg["minimum_group_rows"] and customers >= cfg["minimum_group_customers"]
        if not info["trained"] or not pending:
            continue
        group_folder = folder/"models"/group
        group_folder.mkdir(parents=True)
        write_json(group_folder/"training.json", {"origin_ids": [r["origin_event_id"] for r in subset],
                                                  "target_max": max(r["target_on"] for r in subset)})
        if MODEL_AGENTS[agent] == "xgboost":
            quantiles, quantities = train_xgboost(subset, pending, cfg, group_folder)
        else:
            quantiles, quantities, info["context_rows"] = tabpfn_quantiles(subset, pending, cfg, group_folder)
        for r, q, qty in zip(pending, quantiles, quantities):
            predictions[r["origin_event_id"]] = (q, qty)
        print(json.dumps({"stage": "agent_trained", "fold": fold_id, "group": group, **info}), flush=True)

    schedule = lambda r, gap: weekly_cart(r["origin_on"], gap, cfg["weekly_release_hour"], cfg["weekly_buyer_max_gap_days"])
    records, champion_records = [], []
    for r in valid:
        route = routing[r["origin_event_id"]]
        group = route["customer_group"]
        agent = cfg["routes"][group]
        champion_gap, champion_qty = fallback[r["origin_event_id"]]
        relative_iqr = None
        if r["origin_event_id"] in predictions:
            q, agent_qty = predictions[r["origin_event_id"]]
            agent_gap, relative_iqr = rounded(q[point]), float((q[2]-q[0])/max(q[1], 1))
            missing = None
        else:
            agent_gap = agent_qty = None
            missing = "routed_to_champion" if agent == "champion" else "specialist_sample_insufficient"
        gap, reason = guard(agent_gap, champion_gap, relative_iqr, cfg, missing)
        stage, suggest, fixed_gap = purchase_rule(r, group, cfg)
        if fixed_gap is not None:
            agent, agent_gap, agent_qty, gap, reason = "second_purchase_last_gap", fixed_gap, champion_qty, fixed_gap, None
        outcome = outcomes[r["origin_event_id"]]
        observed = outcome == "observed"
        common = {"fold": fold_id, "origin_event_id": r["origin_event_id"], "customer_id": r["customer_id"],
                  "product_id": r["product_id"], "origin_on": r["origin_on"], "customer_group": group,
                  "purchase_stage": stage, "suggest": suggest, "outcome": outcome,
                  "target_on": r["target_on"] if observed else "", "target_gap_days": r["target_gap_days"] if observed else "",
                  "target_quantity": r["target_quantity"] if observed else ""}
        predicted_on, cart_on, cart_at = schedule(r, gap)
        records.append({**common, "raw_group": route["raw_group"], "selected_agent": agent,
            "fallback_agent": "champion" if reason else "", "route": "champion" if reason else agent,
            "agent_gap_days": "" if agent_gap is None else agent_gap, "champion_gap_days": champion_gap,
            "relative_iqr": "" if relative_iqr is None else round(relative_iqr, 4),
            "predicted_gap_days": gap, "predicted_repurchase_on": predicted_on,
            "cart_quantity": champion_qty if reason else agent_qty, "cart_on": cart_on, "cart_at": cart_at,
            "route_confidence": route["route_confidence"], "route_changed": route["route_changed"],
            "reason": reason or f"{group}_{cfg['route_unit']}",
            # Close the case (stop suggesting) after the window, or a week after a later prediction.
            "monitor_until": (date.fromisoformat(r["origin_on"])+timedelta(days=max(cfg["monitoring_days"], gap+7))).isoformat(),
            "feature_snapshot": json.dumps({k: r[k] for k in feature_names(FEATURE_MODE)}, ensure_ascii=False)})
        champion_on, champion_cart_on, champion_cart_at = schedule(r, champion_gap)
        champion_records.append({**common, "route": "champion", "predicted_gap_days": champion_gap,
                                 "predicted_repurchase_on": champion_on, "cart_quantity": champion_qty,
                                 "cart_on": champion_cart_on, "cart_at": champion_cart_at})
    return records, champion_records, training


def section_report(records, champion_records, cfg):
    def evaluate(rs):
        return {**metrics(rs, cfg), "weekly_order": weekly_metrics(rs, cfg["weekly_buyer_max_gap_days"]),
                "monitor": monitor_metrics(rs, cfg["monitoring_days"])}
    # Both sides are scored on the origins the router would suggest; skipped ones are reported as coverage.
    everything = champion_records
    records = [r for r in records if r["suggest"]]
    champion_records = [r for r in champion_records if r["suggest"]]
    groups = sorted({r["customer_group"] for r in records})
    stages = sorted({r["purchase_stage"] for r in everything})
    folds = sorted({r["fold"] for r in records})
    confirm_from = cfg["mature"]["confirm_from"]
    periods = {"select": lambda f: f < confirm_from, "confirm": lambda f: f >= confirm_from} if len(folds) > 1 else {}
    comparison = {name: {"overall": evaluate(rs),
                         "groups": {g: evaluate([r for r in rs if r["customer_group"] == g]) for g in groups},
                         "stages": {g: evaluate([r for r in rs if r["purchase_stage"] == g]) for g in stages},
                         "folds": {f: evaluate([r for r in rs if r["fold"] == f]) for f in folds},
                         "periods": {k: evaluate([r for r in rs if test(r["fold"])]) for k, test in periods.items()}}
                  for name, rs in (("router", records), ("champion", champion_records))}
    comparison["champion_all_origins"] = {"overall": evaluate(everything),
                                          "stages": {g: evaluate([r for r in everything if r["purchase_stage"] == g]) for g in stages}}
    coverage = {"origins": len(everything), "suggested": len(records),
                "stages": dict(Counter(r["purchase_stage"] for r in everything))}
    attempted = [r for r in records if r["agent_gap_days"] != ""]
    fallbacks = [r for r in records if r["fallback_agent"]]
    routing = {
        "customer_groups": dict(Counter(r["customer_group"] for r in records)),
        "raw_groups": dict(Counter(r["raw_group"] for r in records)),
        "final_agents": dict(Counter(r["route"] for r in records)),
        "fallback_rate": len(fallbacks)/len(records),
        "fallback_reasons_by_group": {g: dict(Counter(r["reason"] for r in fallbacks if r["customer_group"] == g)) for g in groups},
        "later_than_champion_rate": sum(int(r["agent_gap_days"]) > r["champion_gap_days"] for r in attempted)/len(attempted) if attempted else None,
        "route_changed": sum(r["route_changed"] for r in records),
        "late_rate_route_changed": weekly_metrics([r for r in records if r["route_changed"]]).get("late_rate"),
        "late_rate_route_kept": weekly_metrics([r for r in records if not r["route_changed"]]).get("late_rate"),
    }
    router_week, champion_week = comparison["router"]["overall"]["weekly_order"], comparison["champion"]["overall"]["weekly_order"]
    router_monitor, champion_monitor = (comparison[k]["overall"]["monitor"] for k in ("router", "champion"))
    acceptance = {
        "monitor_hit_rate_not_worse": router_monitor["hit_rate"] >= champion_monitor["hit_rate"],
        "monitor_hit_rate_not_worse_by_period": {k: comparison["router"]["periods"][k]["monitor"]["hit_rate"]
                                                 >= comparison["champion"]["periods"][k]["monitor"]["hit_rate"] for k in periods},
        "monitor_hit_rate_not_worse_all_folds": all(comparison["router"]["folds"][f]["monitor"]["hit_rate"]
                                                    >= comparison["champion"]["folds"][f]["monitor"]["hit_rate"] for f in folds),
        "late_not_worse": router_week["late_rate"] <= champion_week["late_rate"],
        "early_1_to_10_improved": router_week["early_1_to_10_rate"] > champion_week["early_1_to_10_rate"],
        "early_over_30_not_increased": router_week["early_over_30_rate"] <= champion_week["early_over_30_rate"],
        "no_group_later": all(comparison["router"]["groups"][g]["weekly_order"]["late_rate"]
                              <= comparison["champion"]["groups"][g]["weekly_order"]["late_rate"] for g in groups
                              if comparison["router"]["groups"][g]["weekly_order"]["n"]),
        "same_direction_across_folds": len({comparison["router"]["folds"][f]["weekly_order"]["late_rate"]
                                            <= comparison["champion"]["folds"][f]["weekly_order"]["late_rate"] for f in folds}) == 1,
        "fallback_rate": routing["fallback_rate"],
    }
    return {"n": len(records), "coverage": coverage, "comparison": comparison, "routing": routing, "acceptance": acceptance}


def run(config_path, output):
    config_path, output = Path(config_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    cfg = json.loads(config_path.read_text())
    if set(cfg["routes"]) != set(GROUPS) or not set(cfg["routes"].values()) <= set(MODEL_AGENTS) | {"champion"}:
        raise ValueError("Routes must map every cadence type to a known agent")
    if cfg["quantiles"] != sorted(cfg["quantiles"]) or len(cfg["quantiles"]) != 3 or cfg["point_quantile"] not in cfg["quantiles"]:
        raise ValueError("Three ascending quantiles including the point quantile required")
    started = time.monotonic()
    rows = read_csv(cfg["enriched_source"])
    routing = stable_routes(rows, cfg)
    output.mkdir(parents=True)
    write_json(output/"config-frozen.json", cfg)
    sections, training = {}, {}

    # 1. Frozen champion validation: the real champion predictions are the fallback.
    champion = base_champion_path()
    champion_report = json.loads((champion/"report.json").read_text())
    if digest(champion/"validation-predictions.csv") != champion_report["files"]["validation-predictions.csv"]:
        raise ValueError("Champion predictions changed")
    baseline = {r["origin_event_id"]: r for r in read_csv(champion/"validation-predictions.csv")}
    valid = [r for r in rows if r["origin_event_id"] in baseline]
    train = [r for r in rows if r["origin_on"] <= cfg["train_end"] and r["label_state"] == "observed"
             and r["target_on"] <= cfg["train_end"] and not excluded_from_training(r, cfg)]
    assert len(valid) == len(baseline)
    for r in valid:
        b = baseline[r["origin_event_id"]]
        assert all(r[k] == b[k] for k in ("customer_id", "product_id", "origin_on", "target_on", "target_gap_days", "target_quantity"))
    assert not {r["origin_event_id"] for r in train} & set(baseline)
    assert max(r["target_on"] for r in train) < min(r["origin_on"] for r in valid)
    fallback = {k: (int(b["predicted_gap_days"]), int(b["cart_quantity"])) for k, b in baseline.items()}
    records, champion_records, training["champion_validation"] = route_fold(
        train, valid, {k: "observed" for k in baseline}, fallback, routing, cfg, output/"champion_validation", "champion_validation")
    sections["champion_validation"] = (records, champion_records)

    # 2. Mature 90-day folds: includes non-returners; champion recipe refit per fold.
    protocol = cfg["mature"]
    event_root = Path(cfg["event_source"])
    labels = {r["origin_event_id"]: r for r in read_csv(event_root/"next-purchase-labels.csv")}
    events = {r["event_id"]: r for r in read_csv(event_root/"daily-events.csv")}
    mature, mature_champion = [], []
    for fold in protocol["folds"]:
        train = [r for r in training_rows(rows, fold["train_end"], protocol["training_window_days"], protocol["horizon_days"])
                 if not excluded_from_training(r, cfg)]
        valid = [r for r in rows if fold["origin_start"] <= r["origin_on"] <= fold["origin_end"]]
        outcomes = {r["origin_event_id"]: horizon_outcome(r, protocol["horizon_days"], cfg["mature_end"], labels, events) for r in valid}
        assert train and valid and "not_mature" not in outcomes.values()
        assert not {r["origin_event_id"] for r in train} & set(outcomes)
        assert max(r["target_on"] for r in train) < min(r["origin_on"] for r in valid)
        folder = output/"mature90"/fold["id"]
        folder.mkdir(parents=True)
        fallback = champion_recipe(train, valid, folder)
        records, champion_records, training[fold["id"]] = route_fold(
            train, valid, outcomes, fallback, routing, cfg, folder, fold["id"])
        mature += records
        mature_champion += champion_records
    sections["mature90"] = (mature, mature_champion)

    all_records = [r for rs, _ in sections.values() for r in rs]
    write_csv(output/"predictions.csv", list(all_records[0]), all_records)
    champion_all = [r for _, cs in sections.values() for r in cs]
    write_csv(output/"champion-schedule.csv", list(champion_all[0]), champion_all)
    weights = Path(cfg["tabpfn"]["weights"])
    report = {"version": cfg["version"], "config": cfg, "training": training,
              "sections": {name: section_report(rs, cs, cfg) for name, (rs, cs) in sections.items()},
              "hysteresis": {"all_origin_switches": sum(v["route_changed"] for v in routing.values()),
                             "all_origin_held_back": sum(v["raw_group"] != v["customer_group"] for v in routing.values())},
              "operational_approval": False, "independent_final_test": False,
              "limitations": [
                  "champion_validation: consumed cohort of observed repurchases only; the real frozen champion is the fallback",
                  "mature90: monthly walk-forward folds (April-June 2026 were used in earlier experiments); the fallback is the champion recipe refit on full history, not the frozen champion",
                  "Point quantile and routes were chosen after viewing champion_validation",
                  "Unnecessary carts count non-returners whose cart releases within 90 days; no measured stockout",
                  "No significance testing"],
              "provenance": {"config_sha256": digest(config_path), "code_sha256": digest(__file__),
                             "enriched_sha256": digest(cfg["enriched_source"]),
                             "champion_report_sha256": digest(champion/"report.json"),
                             "champion_predictions_sha256": digest(champion/"validation-predictions.csv"),
                             "labels_sha256": digest(event_root/"next-purchase-labels.csv"),
                             "events_sha256": digest(event_root/"daily-events.csv"),
                             "tabpfn_worker_sha256": digest("scripts/tabpfn_worker.py"),
                             "tabpfn_weights_sha256": digest(weights) if weights.exists() else None,
                             "xgboost_version": xgb.__version__},
              "duration_seconds": time.monotonic()-started,
              "files": {str(p.relative_to(output)): digest(p) for p in output.rglob("*") if p.is_file()}}
    write_json(output/"report.json", report)
    print(json.dumps({name: {"acceptance": s["acceptance"], "final_agents": s["routing"]["final_agents"]}
                      for name, s in report["sections"].items()}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/multi-agent-router-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/multi-agent-router-v1-walkforward"))
    args = parser.parse_args()
    run(args.config, args.output)
