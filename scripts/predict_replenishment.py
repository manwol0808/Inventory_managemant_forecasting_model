"""Local preview adapter for the frozen candidate; no cart writes or network calls."""
import argparse
import hashlib
import json
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import xgboost as xgb

from scripts.build_features_and_splits import FEATURE_COLUMNS, next_midnight
from scripts.run_baseline import digest, write_json
from scripts.train_xgboost import matrix, postprocess


DEFAULT_CART_LEAD_DAYS = 10
DEFAULT_WEEKLY_RELEASE_HOUR = 22
KOREA = ZoneInfo("Asia/Seoul")


def cart_schedule(predicted_on, available_on, lead_days=DEFAULT_CART_LEAD_DAYS):
    """Schedule an early suggestion, never before the features are available."""
    if type(lead_days) is not int or not 0 <= lead_days <= 30:
        raise ValueError("cart lead days must be an integer between 0 and 30")
    desired = date.fromisoformat(predicted_on) - timedelta(days=lead_days)
    available = date.fromisoformat(available_on)
    return max(desired, available).isoformat()


def weekly_cart_schedule(predicted_on, available_at, release_hour=DEFAULT_WEEKLY_RELEASE_HOUR):
    """Prepare Sunday night for the Monday before the predicted repurchase week.

    Return nominal and feasible release times. Prefer a Sunday/Monday window;
    if the remaining window would be after the prediction, catch up immediately.
    """
    if type(release_hour) is not int or not 0 <= release_hour <= 23:
        raise ValueError("Sunday release hour must be an integer between 0 and 23")
    if available_at.tzinfo is None or available_at.utcoffset() is None:
        raise ValueError("Timezone required for feature availability")
    predicted = date.fromisoformat(predicted_on)
    forecast_monday = predicted - timedelta(days=predicted.weekday())
    release_sunday = forecast_monday - timedelta(days=8)
    planned = datetime.combine(release_sunday, time(release_hour), KOREA)
    available = available_at.astimezone(KOREA)
    if available <= planned:
        return planned, planned
    # Monday itself is an ordering window. Otherwise use the next Sunday night
    # only if it is strictly before the predicted repurchase day starts.
    if available.weekday() == 0:
        return planned, available
    next_sunday = available.date() + timedelta(days=(6-available.weekday()) % 7)
    next_release = datetime.combine(next_sunday, time(release_hour), KOREA)
    if next_release < available:
        # Already Sunday night: catch up in the current ordering window.
        return planned, available
    if next_release < datetime.combine(predicted, time(), KOREA):
        return planned, next_release
    return planned, available


def verified_model_dir(entry):
    root=Path(entry["model_dir"])
    if digest(root/"report.json")!=entry["report_sha256"]:
        raise ValueError("Champion report changed since selection")
    for name,expected in entry["model_files"].items():
        if digest(root/name)!=expected:
            raise ValueError("Champion artifact changed since selection")
    return root


def champion_path(registry_path=Path("config/champion.json")):
    """Registered champion folder: a single XGBoost (v1) or a router bundle (v2)."""
    registry=json.loads(Path(registry_path).read_text())
    if registry.get("version") not in {"champion-v1","champion-v2"} or registry.get("role")!="champion":
        raise ValueError("Invalid champion registry")
    if registry["version"]=="champion-v2":
        verified_model_dir(registry["base_model"])
    return verified_model_dir(registry)


def base_champion_path(registry_path=Path("config/champion.json")):
    """Single XGBoost behind the champion: the champion itself (v1) or the router's base model (v2)."""
    registry=json.loads(Path(registry_path).read_text())
    champion_path(registry_path)
    return Path(registry["base_model"]["model_dir"]) if registry["version"]=="champion-v2" else Path(registry["model_dir"])


class PreviewPredictor:
    def __init__(self, model_dir, cart_lead_days=DEFAULT_CART_LEAD_DAYS, *,
                 cart_policy="weekly_order", weekly_release_hour=DEFAULT_WEEKLY_RELEASE_HOUR):
        cart_schedule("2026-01-01", "2026-01-01", cart_lead_days)
        if cart_policy not in {"weekly_order", "lead_days"}:
            raise ValueError("Unknown cart policy")
        if type(weekly_release_hour) is not int or not 0 <= weekly_release_hour <= 23:
            raise ValueError("Sunday release hour must be an integer between 0 and 23")
        self.cart_lead_days = cart_lead_days
        self.cart_policy = cart_policy
        self.weekly_release_hour = weekly_release_hour
        self.root=Path(model_dir)
        self.report=json.loads((self.root/"report.json").read_text())
        self.schema=json.loads((self.root/"feature-schema.json").read_text())
        self.model_version=self.root.name+":"+digest(self.root/"report.json")[:12]
        self.models={}
        for target in ("quantity","gap"):
            path=self.root/(target+"-model.json")
            if digest(path)!=self.report["files"][path.name]:
                raise ValueError("Saved model changed")
            model=xgb.Booster({"nthread":4})
            model.load_model(path)
            self.models[target]=model
        if digest(self.root/"feature-schema.json")!=self.report["files"]["feature-schema.json"]:
            raise ValueError("Feature schema changed")

    def predict(self, request):
        required={"customer_id","product_id","origin_event_id","origin_on","feature_asof","history_start",
                  "latest_purchase_confirmed","has_later_purchase_or_hold","as_of","features"}
        if set(request)!=required or set(request["features"])!=set(FEATURE_COLUMNS):
            raise ValueError("Exact request/feature contract required; no targets or extra fields")
        if not all(isinstance(request[k],str) and request[k] for k in ("customer_id","product_id","origin_event_id")):
            raise ValueError("Nonempty string identifiers required")
        if request["latest_purchase_confirmed"] is not True or request["has_later_purchase_or_hold"] is not False:
            raise ValueError("Require latest eligible purchase with no later purchase or hold")
        if request["history_start"]!=self.report["split_config"]["start"]:
            raise ValueError("Feature history window differs from fitted candidate")
        if str(request["features"]["product_id"])!=request["product_id"]:
            raise ValueError("Product identity mismatch")
        as_of=datetime.fromisoformat(request["as_of"])
        feature_asof=datetime.fromisoformat(request["feature_asof"])
        if as_of.tzinfo is None or feature_asof.tzinfo is None:
            raise ValueError("Timezone required")
        origin=date.fromisoformat(request["origin_on"])
        if feature_asof!=next_midnight(request["origin_on"]) or feature_asof>as_of:
            raise ValueError("Feature timestamp must be next local midnight and not in future")
        if request["origin_on"]<=self.report["split_config"]["train_end"]:
            raise ValueError("Preview origin must follow fitted training period")
        dm=matrix([request["features"]],self.schema)
        qty,cart,gap=postprocess(self.models["quantity"].predict(dm),self.models["gap"].predict(dm))
        reorder_on=(origin+timedelta(days=int(gap[0]))).isoformat()
        lead_days = self.cart_lead_days
        available_on = feature_asof.astimezone(KOREA).date().isoformat()
        if self.cart_policy == "weekly_order":
            planned_at, cart_at = weekly_cart_schedule(reorder_on, feature_asof, self.weekly_release_hour)
            order_monday = ((cart_at.date() + timedelta(days=1)).isoformat() if cart_at.weekday() == 6
                            else cart_at.date().isoformat() if cart_at.weekday() == 0 else None)
        else:
            cart_on = cart_schedule(reorder_on, available_on, lead_days)
            planned_on = date.fromisoformat(reorder_on) - timedelta(days=lead_days)
            planned_at = datetime.combine(planned_on, time(), KOREA)
            cart_at = datetime.combine(date.fromisoformat(cart_on), time(), KOREA)
            order_monday = None
        cart_on = cart_at.date().isoformat()
        # Exclude model version: a new model must not create a second cart insertion
        # for the same purchase event. The app persists accepted/dismissed keys.
        key=hashlib.sha256(json.dumps([request["customer_id"],request["product_id"],request["origin_event_id"]],ensure_ascii=False).encode()).hexdigest()
        return {"customer_id":request["customer_id"],"product_id":request["product_id"],
                "origin_event_id":request["origin_event_id"],"model_version":self.model_version,
                "suggestion_key":key,"reorder_on":reorder_on,"predicted_quantity":float(qty[0]),
                "cart_quantity":int(cart[0]),"cart_on":cart_on,
                "cart_lead_days":lead_days if self.cart_policy == "lead_days" else None,
                "cart_policy_version":"weekly-order-v1" if self.cart_policy == "weekly_order" else "early-cart-v1",
                "planned_cart_at":planned_at.isoformat(),"cart_at":cart_at.isoformat(),
                "order_monday":order_monday,"schedule_adjusted_for_availability":cart_at>planned_at,
                "due":cart_at<=as_of,
                "mode":"preview_only","auto_apply":False,"operational_approval":False,
                "product_seen_in_training":request["product_id"] in self.schema["products"],
                "limitation":"Post-purchase forecast; not measured inventory or confidence probability"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request",type=Path)
    parser.add_argument("--model",type=Path,help="Explicit model override; defaults to the champion's single XGBoost (a router champion is served by scripts.router_champion)")
    parser.add_argument("--cart-lead-days", type=int, default=DEFAULT_CART_LEAD_DAYS,
                        help="Lead for --cart-policy lead_days (default: 10)")
    parser.add_argument("--cart-policy", choices=("weekly_order", "lead_days"), default="weekly_order",
                        help="Default: Sunday night before the prior week's Monday ordering window")
    parser.add_argument("--weekly-release-hour", type=int, default=DEFAULT_WEEKLY_RELEASE_HOUR,
                        help="Sunday release hour in Asia/Seoul (default: 22)")
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result=PreviewPredictor(args.model or base_champion_path(), args.cart_lead_days,
                            cart_policy=args.cart_policy, weekly_release_hour=args.weekly_release_hour).predict(json.loads(args.request.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
        stream.write("\n")
    print(json.dumps({"mode":result["mode"],"auto_apply":result["auto_apply"],"output":str(args.output)},ensure_ascii=False))
