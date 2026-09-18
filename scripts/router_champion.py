"""Promote the evaluated cadence router to champion, and schedule carts with it.

promote: freeze a router run's agents with the current single-XGBoost champion as its base model.
predict: route every pair's latest purchase and schedule its cart; local CSV only, no app writes.

담는 날에는 두 가지 조정이 얹힌다. 매장 주문 리듬 모델이 더 이른 날을 내면 그쪽으로 당기고
(최대 STORE_RHYTHM_MAX_PULL_DAYS), 짝별 과거 오차 보정이 있으면 예측 간격에서 뺀다.
수량도 매장 리듬 모델 값이 있으면 그것을 쓴다. 둘 다 입력이 없으면 기존 동작 그대로다.
"""
import argparse
import json
import shutil
from collections import Counter
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import xgboost as xgb

from scripts.build_features_and_splits import next_midnight, read_csv
from scripts.build_purchase_events import write_csv
from scripts.pair_adjustments import DEFAULT_CAP_DAYS, DEFAULT_DB, PairAdjustments
from scripts.predict_replenishment import KOREA, base_champion_path, champion_path
from scripts.run_baseline import digest, write_json
from scripts.run_multi_agent_router import (FEATURE_MODE, MODEL_AGENTS, feature_array, guard, purchase_rule,
                                            run_tabpfn, stable_routes, weekly_cart)
from scripts.run_purchase_segments import design_matrix
from scripts.run_timing_experiment import rounded
from scripts.train_xgboost import matrix, postprocess

REGISTRY = Path("config/champion.json")
STORE_RHYTHM_MAX_PULL_DAYS = 7


def read_store_rhythm(path):
    """매장 주문 리듬 모델의 짝별 예측. customer_id, product_id, cart_on, cart_quantity 를 가진다."""
    out = {}
    for row in read_csv(path):
        key = (row["customer_id"], row["product_id"])
        if not row.get("cart_on"):
            continue
        date.fromisoformat(row["cart_on"])
        quantity = row.get("cart_quantity") or ""
        out[key] = (row["cart_on"], int(float(quantity)) if quantity else None)
    return out


def pull_forward(order_day, cart_at, rhythm_on, available_at, release_hour, max_days=STORE_RHYTHM_MAX_PULL_DAYS):
    """매장 리듬 예측이 더 이르면 담는 날을 당긴다. 앞당김은 max_days 로 제한한다.

    담는 시각은 새 담는 날 직전 일요일 밤이며, 피처가 준비되기 전으로는 가지 않는다.
    반환은 (담는 날, 담는 시각)이고, 당길 이유가 없으면 입력을 그대로 돌려준다.
    """
    planned, rhythm = date.fromisoformat(order_day), date.fromisoformat(rhythm_on)
    if rhythm >= planned:
        return order_day, cart_at, False
    target = max(rhythm, planned - timedelta(days=max_days))
    release = datetime.combine(target - timedelta(days=(target.weekday() + 1) % 7 or 7),
                               time(release_hour), KOREA)
    available = available_at.astimezone(KOREA)
    if release < available:
        release = available
    if release >= datetime.fromisoformat(cart_at):
        return order_day, cart_at, False
    day = release.date() + timedelta(days=1 if release.weekday() == 6 else 0)
    return day.isoformat(), release.isoformat(), True


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
    def __init__(self, registry_path=REGISTRY, *, store_rhythm=None, adjustments_db=None,
                 adjustment_cap=DEFAULT_CAP_DAYS):
        self.store_rhythm = read_store_rhythm(store_rhythm) if store_rhythm else {}
        self.adjustments = {}
        if adjustments_db:
            with PairAdjustments(adjustments_db, adjustment_cap) as store:
                self.adjustments = store.all_adjustments()
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
            pair = (r["customer_id"], r["product_id"])
            adjustment = self.adjustments.get(pair, 0) if suggest else 0
            scheduled_gap = max(1, gap + adjustment)
            predicted_on, cart_on, cart_at = weekly_cart(r["origin_on"], scheduled_gap, cfg["weekly_release_hour"], cfg["weekly_buyer_max_gap_days"])
            predicted_on = (date.fromisoformat(r["origin_on"])+timedelta(days=gap)).isoformat()
            rhythm_on, rhythm_qty = self.store_rhythm.get(pair, (None, None))
            pulled = False
            if rhythm_on and suggest:
                cart_on, cart_at, pulled = pull_forward(cart_on, cart_at, rhythm_on,
                                                        next_midnight(r["origin_on"]), cfg["weekly_release_hour"])
            monitor_until = date.fromisoformat(r["origin_on"])+timedelta(days=max(cfg["monitoring_days"], gap+7))
            status = ("no_suggestion_first_purchase" if not suggest else "closed" if as_of.date() > monitor_until
                      else "due" if datetime.fromisoformat(cart_at) <= as_of else "scheduled")
            records.append({"customer_id": r["customer_id"], "product_id": r["product_id"],
                "origin_event_id": r["origin_event_id"], "origin_on": r["origin_on"], "customer_group": group,
                "purchase_stage": stage, "selected_agent": agent, "fallback_agent": "champion" if reason else "",
                "predicted_gap_days": gap, "predicted_repurchase_on": predicted_on,
                "pair_adjustment_days": adjustment, "scheduled_gap_days": scheduled_gap,
                "store_rhythm_on": rhythm_on or "", "store_rhythm_pulled": pulled,
                "cart_quantity": rhythm_qty if rhythm_qty else (champion_qty if reason else agent_qty),
                "router_quantity": champion_qty if reason else agent_qty, "cart_on": cart_on, "cart_at": cart_at,
                "monitor_until": monitor_until.isoformat(), "status": status,
                "route_confidence": route["route_confidence"], "route_changed": route["route_changed"],
                "reason": reason or f"{group}_{cfg['route_unit']}"})
        write_csv(output/"schedule.csv", list(records[0]), records)
        summary = {"as_of": as_of.isoformat(), "champion": self.report["model_id"], "rows": len(records),
                   "status": {s: sum(r["status"] == s for r in records) for s in sorted({r["status"] for r in records})},
                   "open_by_final_agent": dict(Counter(r["fallback_agent"] or r["selected_agent"] for r in records
                                                       if r["status"] in {"due", "scheduled"})),
                   "store_rhythm": {"pairs_loaded": len(self.store_rhythm),
                                    "carts_pulled": sum(r["store_rhythm_pulled"] for r in records),
                                    "quantities_used": sum(bool(self.store_rhythm.get((r["customer_id"], r["product_id"]), (None, None))[1])
                                                           for r in records),
                                    "max_pull_days": STORE_RHYTHM_MAX_PULL_DAYS},
                   "pair_adjustment": {"pairs_loaded": len(self.adjustments),
                                       "carts_adjusted": sum(r["pair_adjustment_days"] != 0 for r in records)},
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
    s.add_argument("--store-rhythm", type=Path, help="매장 주문 리듬 모델의 짝별 예측 CSV")
    s.add_argument("--adjustments-db", type=Path, nargs="?", const=DEFAULT_DB,
                   help="짝별 보정 저장소. 생략하면 보정하지 않는다")
    s.add_argument("--adjustment-cap-days", type=int, default=DEFAULT_CAP_DAYS)
    args = parser.parse_args()
    if args.command == "promote":
        print(json.dumps(promote(args.source, args.output, selected_on=args.selected_on,
                                 evaluation=json.loads(args.evaluation.read_text())), ensure_ascii=False, indent=2))
    else:
        as_of = datetime.fromisoformat(args.as_of)
        if as_of.tzinfo is None:
            raise ValueError("Timezone required")
        champion = RouterChampion(store_rhythm=args.store_rhythm, adjustments_db=args.adjustments_db,
                                  adjustment_cap=args.adjustment_cap_days)
        print(json.dumps(champion.schedule(as_of.astimezone(KOREA), args.output), ensure_ascii=False, indent=2))
