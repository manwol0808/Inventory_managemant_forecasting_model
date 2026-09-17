"""Isolate PyTorch inference from the XGBoost native runtime; local arrays only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion


def run(input_path, weights, config_path, output):
    cfg = json.loads(config_path.read_text())
    if output.exists():
        raise FileExistsError(output)
    data = np.load(input_path,allow_pickle=False)
    if set(data.files) != {"x_train","y_train","x_valid"} or data["x_train"].shape[1] != cfg.get("feature_count",21):
        raise ValueError("Unexpected local feature arrays")
    extra = {"quantiles":cfg["quantiles"]} if cfg["output_type"] == "quantiles" else {}
    torch.set_num_threads(cfg["torch_threads"])
    model = TabPFNRegressor.create_default_for_version(ModelVersion.V2,model_path=weights,
        n_estimators=cfg["n_estimators"],auto_scale_n_estimators=cfg["auto_scale_n_estimators"],
        categorical_features_indices=[0],device=cfg["device"],random_state=cfg["seed"],
        n_preprocessing_jobs=1,fit_mode="fit_with_cache",show_progress_bar=False)
    model.fit(data["x_train"],data["y_train"])
    raw = []
    for start in range(0,len(data["x_valid"]),cfg["batch_size"]):
        predicted = model.predict(data["x_valid"][start:start+cfg["batch_size"]],output_type=cfg["output_type"],**extra)
        # Quantile output is one array per requested quantile; store rows as (n, quantiles).
        raw.extend((np.column_stack(predicted) if isinstance(predicted,list) else predicted).tolist())
        if start % (cfg["batch_size"]*4) == 0:
            print(json.dumps({"fold":input_path.parent.name,"predicted":min(start+cfg["batch_size"],len(data["x_valid"])),"total":len(data["x_valid"])}),flush=True)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite TabPFN prediction")
    np.save(output,np.asarray(raw),allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input","weights","config","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args()
    run(args.input,args.weights,args.config,args.output)
