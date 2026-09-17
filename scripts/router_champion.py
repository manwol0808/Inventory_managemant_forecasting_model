"""Promote the evaluated cadence router to champion, and schedule carts with it.

promote: freeze a router run's agents with the current single-XGBoost champion as its base model.
predict: route every pair's latest purchase and schedule its cart; local CSV only, no app writes.
"""
import argparse
import json
import shutil
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import xgboost as xgb

from scripts.build_features_and_splits import next_midnight, read_csv
from scripts.build_purchase_events import write_csv
from scripts.predict_replenishment import KOREA, base_champion_path, champion_path
from scripts.run_baseline import digest, write_json
from scripts.run_multi_agent_router import (FEATURE_MODE, MODEL_AGENTS, feature_array, guard, purchase_rule,
                                            run_tabpfn, stable_routes, weekly_cart)
from scripts.run_purchase_segments import design_matrix
from scripts.run_timing_experiment import rounded
from scripts.train_xgboost import matrix, postprocess

REGISTRY = Path("config/champion.json")


def promote(source, output, registry_path=REGISTRY, *, selected_on, evaluation):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(output)
    registry = json.loads(registry_path.read_text())
    if registry["version"] != "champion-v1":
        raise ValueError("Base must be a single-model champion")
    champion_path(registry_path)
    cfg = json.loads((source/"config-frozen.json").read_text())
    agents = {g: a for g, a in cfg["routes"].items() if a in MODEL_AGENTS}
    output.mkdir(parents=True)
    for key in ("skip_first_purchase", "second_purchase_last_gap", "monitoring_days", "training_exclude"):
        if key not in cfg:
            raise ValueError(f"Source run lacks {key}")
    write_json(output/"router-config.json", cfg)
    for group, agent in agents.items():
        src, dst = source/"champion_validation"/"models"/group, output/"models"/group
        dst.mkdir(parents=True)
        shutil.copy(src/"schema.json", dst/"schema.json")
        if MODEL_AGENTS[agent] == "xgboost":
            for name in ("gap.json", "quantity.json"):
                shutil.copy(src/name, dst/name)
        else:
            data = np.load(src/"input.npz", allow_pickle=False)
            np.savez(dst/"context.npz", x_train=data["x_train"], y_train=data["y_train"])
            shutil.copy(src/"worker.json", dst/"worker.json")
    base = registry["model_dir"]
    base_report = json.loads((Path(base)/"report.json").read_text())
    write_json(output/"report.json", {
        "model_id": "multi-agent-router-v1", "source_run": str(source),
        "source_report_sha256": digest(source/"report.json"), "source_config_sha256": digest(source/"config-frozen.json"),
        "agents": agents, "base_model_id": registry["model_id"],
        "base_feature_source": "data/features-splits-short-v2/all-origins-audit.csv",
        "base_history_start": base_report["split_config"]["start"],
        "train_end": cfg["train_end"], "evaluation": evaluation,
        "tabpfn_weights_sha256": digest(cfg["tabpfn"]["weights"]),
        "limitations": ["Selected by the user from exploratory results on consumed cohorts; no fresh future test",
                        "The stable-group gain is 118 champion-validation repurchases within about 30 days",
                        "Monthly 90-day folds (45-day window) scored below the champion recipe after the 2026-01/02 training exclusion",
                        "Base XGBoost uses 2026-04-09+ features; agents use 2024-10-16+ enriched features"]})
    files = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "report.json"}
    promoted = {"version": "champion-v2", "model_id": "multi-agent-router-v1", "kind": "router",
                "model_dir": str(output), "role": "champion", "selected_by": "user", "selected_on": selected_on,
                "selection_basis": "User-selected router (stable XGBoost q25, speeding-up TabPFN q25, champion elsewhere, first purchases skipped, 2026-01/02 excluded from training) after its stable-group champion-validation result; monthly 90-day folds did not confirm an overall improvement",
                "report_sha256": digest(output/"report.json"), "model_files": files,
                "base_model": registry, "previous_champion": registry["model_id"],
                "operational_approval": False, "auto_apply": False}
    write_json(registry_path, promoted)
    return promoted


class RouterChampion:
    def __init__(self, registry_path=REGISTRY):
        self.root = champion_path(registry_path)
        self.base_root = base_champion_path(registry_path)
        self.report = json.loads((self.root/"report.json").read_text())
        self.cfg = json.loads((self.root/"router-config.json").read_text())
        self.base_schema = json.loads((self.base_root/"feature-schema.json").read_text())
        self.base = {}
        for target in ("gap", "quantity"):
            self.base[target] = xgb.Booster({"nthread": 4})
            self.base[target].load_model(self.base_root/f"{target}-model.json")

    def base_predictions(self, rows):
        short = {r["origin_event_id"]: r for r in read_csv(self.report["base_feature_source"])}
        known = [r for r in rows if r["origin_event_id"] in short]
        dm = matrix([short[r["origin_event_id"]] for r in known], self.base_schema)
        _, carts, gaps = postprocess(self.base["quantity"].predict(dm), self.base["gap"].predict(dm))
        return {r["origin_event_id"]: (int(g), int(c)) for r, g, c in zip(known, gaps, carts)}

    def agent_predictions(self, group, agent, rows, work):
        folder = self.root/"models"/group
        schema = json.loads((folder/"schema.json").read_text())
        if MODEL_AGENTS[agent] == "xgboost":
            models = []
            for name in ("gap.json", "quantity.json"):
                model = xgb.Booster({"nthread": 4})
                model.load_model(folder/name)
                models.append(model)
            dm = design_matrix(rows, schema)
            return models[0].predict(dm), [rounded(q) for q in models[1].predict(dm)]
        context = np.load(folder/"context.npz", allow_pickle=False)
        target = work/group
        target.mkdir(parents=True)
        values = run_tabpfn(context["x_train"], context["y_train"], feature_array(rows, schema),
                            json.loads((folder/"worker.json").read_text()), self.cfg["tabpfn"]["weights"], target)
        return values, [rounded(r["qty_median_last3"]) for r in rows]

    def schedule(self, as_of, output):
        cfg = self.cfg
        rows = read_csv(cfg["enriched_source"])
        routing = stable_routes(rows, cfg)
        latest = [r for r in rows if r["label_state"] == "right_censored" and next_midnight(r["origin_on"]) <= as_of]
        base = self.base_predictions(latest)
        latest = [r for r in latest if r["origin_event_id"] in base]
        output.mkdir(parents=True)
        point = cfg["quantiles"].index(cfg["point_quantile"])
        agents = {}
        for group, agent in cfg["routes"].items():
            pending = [r for r in latest if routing[r["origin_event_id"]]["customer_group"] == group]
            if agent in MODEL_AGENTS and pending and (self.root/"models"/group).exists():
                quantiles, quantities = self.agent_predictions(group, agent, pending, output/"work")
                agents.update({r["origin_event_id"]: (q, qty) for r, q, qty in zip(pending, quantiles, quantities)})
        records = []
        for r in latest:
            route = routing[r["origin_event_id"]]
            group, agent = route["customer_group"], cfg["routes"][route["customer_group"]]
            champion_gap, champion_qty = base[r["origin_event_id"]]
            relative_iqr, agent_gap, agent_qty = None, None, None
            if r["origin_event_id"] in agents:
                q, agent_qty = agents[r["origin_event_id"]]
                agent_gap, relative_iqr = rounded(q[point]), float((q[2]-q[0])/max(q[1], 1))
            missing = "routed_to_champion" if agent == "champion" else "specialist_unavailable"
            gap, reason = guard(agent_gap, champion_gap, relative_iqr, cfg, missing)
            stage, suggest, fixed_gap = purchase_rule(r, group, cfg)
            if fixed_gap is not None:
                agent, gap, reason, agent_qty = "second_purchase_last_gap", fixed_gap, None, champion_qty
            predicted_on, cart_on, cart_at = weekly_cart(r["origin_on"], gap, cfg["weekly_release_hour"], cfg["weekly_buyer_max_gap_days"])
            monitor_until = date.fromisoformat(r["origin_on"])+timedelta(days=max(cfg["monitoring_days"], gap+7))
            status = ("no_suggestion_first_purchase" if not suggest else "closed" if as_of.date() > monitor_until
                      else "due" if datetime.fromisoformat(cart_at) <= as_of else "scheduled")
            records.append({"customer_id": r["customer_id"], "product_id": r["product_id"],
                "origin_event_id": r["origin_event_id"], "origin_on": r["origin_on"], "customer_group": group,
                "purchase_stage": stage, "selected_agent": agent, "fallback_agent": "champion" if reason else "",
                "predicted_gap_days": gap, "predicted_repurchase_on": predicted_on,
                "cart_quantity": champion_qty if reason else agent_qty, "cart_on": cart_on, "cart_at": cart_at,
                "monitor_until": monitor_until.isoformat(), "status": status,
                "route_confidence": route["route_confidence"], "route_changed": route["route_changed"],
                "reason": reason or f"{group}_{cfg['route_unit']}"})
        write_csv(output/"schedule.csv", list(records[0]), records)
        summary = {"as_of": as_of.isoformat(), "champion": self.report["model_id"], "rows": len(records),
                   "status": {s: sum(r["status"] == s for r in records) for s in sorted({r["status"] for r in records})},
                   "open_by_final_agent": dict(Counter(r["fallback_agent"] or r["selected_agent"] for r in records
                                                       if r["status"] in {"due", "scheduled"})),
                   "mode": "preview_only", "auto_apply": False,
                   "provenance": {"registry_sha256": digest(REGISTRY), "enriched_sha256": digest(cfg["enriched_source"])}}
        write_json(output/"summary.json", summary)
        return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("promote")
    p.add_argument("--source", type=Path, default=Path("artifacts/router-champion-source-v1"))
    p.add_argument("--output", type=Path, default=Path("artifacts/router-champion-v1"))
    p.add_argument("--selected-on", required=True)
    p.add_argument("--evaluation", type=Path, required=True, help="JSON summary of the results the user selected from")
    s = sub.add_parser("predict")
    s.add_argument("--as-of", required=True, help="Timezone-aware ISO time, e.g. 2026-09-16T00:00:00+09:00")
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "promote":
        print(json.dumps(promote(args.source, args.output, selected_on=args.selected_on,
                                 evaluation=json.loads(args.evaluation.read_text())), ensure_ascii=False, indent=2))
    else:
        as_of = datetime.fromisoformat(args.as_of)
        if as_of.tzinfo is None:
            raise ValueError("Timezone required")
        print(json.dumps(RouterChampion().schedule(as_of.astimezone(KOREA), args.output), ensure_ascii=False, indent=2))
