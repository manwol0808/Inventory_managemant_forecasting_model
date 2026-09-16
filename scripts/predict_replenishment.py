"""Local preview adapter for the frozen candidate; no cart writes or network calls."""
import argparse
import hashlib
import json
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import xgboost as xgb

from scripts.build_features_and_splits import FEATURE_COLUMNS, next_midnight
from scripts.run_baseline import digest, write_json
from scripts.train_xgboost import matrix, postprocess


def champion_path(registry_path=Path("config/champion.json")):
    registry=json.loads(Path(registry_path).read_text())
    root=Path(registry["model_dir"])
    if registry.get("version")!="champion-v1" or registry.get("role")!="champion":
        raise ValueError("Invalid champion registry")
    if digest(root/"report.json")!=registry["report_sha256"]:
        raise ValueError("Champion report changed since selection")
    for name,expected in registry["model_files"].items():
        if digest(root/name)!=expected:
            raise ValueError("Champion artifact changed since selection")
    return root


class PreviewPredictor:
    def __init__(self, model_dir):
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
        # Exclude model version: a new model must not create a second cart insertion
        # for the same purchase event. The app persists accepted/dismissed keys.
        key=hashlib.sha256(json.dumps([request["customer_id"],request["product_id"],request["origin_event_id"]],ensure_ascii=False).encode()).hexdigest()
        return {"customer_id":request["customer_id"],"product_id":request["product_id"],
                "origin_event_id":request["origin_event_id"],"model_version":self.model_version,
                "suggestion_key":key,"reorder_on":reorder_on,"predicted_quantity":float(qty[0]),
                "cart_quantity":int(cart[0]),"due":reorder_on<=as_of.astimezone(ZoneInfo("Asia/Seoul")).date().isoformat(),
                "mode":"preview_only","auto_apply":False,"operational_approval":False,
                "product_seen_in_training":request["product_id"] in self.schema["products"],
                "limitation":"Post-purchase forecast; not measured inventory or confidence probability"}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request",type=Path)
    parser.add_argument("--model",type=Path,help="Explicit model override; defaults to config/champion.json")
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result=PreviewPredictor(args.model or champion_path()).predict(json.loads(args.request.read_text()))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
        stream.write("\n")
    print(json.dumps({"mode":result["mode"],"auto_apply":result["auto_apply"],"output":str(args.output)},ensure_ascii=False))
