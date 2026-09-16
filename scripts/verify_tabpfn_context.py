"""Check saved TabPFN reproducibility and invariance to later query features."""
import json
import os
from pathlib import Path

import numpy as np


def run():
    root = Path("artifacts/tabpfn-experiment-v1")
    cfg = json.loads((root/"config-frozen.json").read_text())
    os.environ["SKB_DATA_DIRECTORY"] = str(Path("artifacts/cache/skrub").resolve())
    os.environ["MPLCONFIGDIR"] = str(Path("artifacts/cache/matplotlib").resolve())
    os.environ["XDG_CACHE_HOME"] = str(Path("artifacts/cache").resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from tabpfn import TabPFNRegressor
    from tabpfn.constants import ModelVersion
    torch.set_num_threads(cfg["torch_threads"])
    result = {}
    for fold in ("apr2026","may2026","jun2026"):
        data = np.load(root/fold/"context-and-features.npz",allow_pickle=False)
        model = TabPFNRegressor.create_default_for_version(ModelVersion.V2,model_path=Path(cfg["weights"]).resolve(),
            n_estimators=cfg["n_estimators"],categorical_features_indices=[0],device=cfg["device"],random_state=cfg["seed"],
            n_preprocessing_jobs=1,fit_mode="fit_with_cache",show_progress_bar=False)
        model.fit(data["x_train"],data["y_train"])
        query = data["x_valid"][:8]
        alone = model.predict(query[:1],output_type=cfg["output_type"])
        batch = model.predict(query,output_type=cfg["output_type"])
        changed = query.copy()
        changed[1:,1:] = 100000
        future_changed = model.predict(changed,output_type=cfg["output_type"])
        saved = np.load(root/fold/"tabpfn-raw.npy",allow_pickle=False)[:8]
        np.testing.assert_allclose(batch,saved,rtol=1e-5,atol=1e-4)
        np.testing.assert_allclose(alone[0],batch[0],rtol=1e-5,atol=1e-4)
        np.testing.assert_allclose(future_changed[0],batch[0],rtol=1e-5,atol=1e-4)
        result[fold] = {"reload_max_abs_difference":float(np.max(np.abs(batch-saved))),
            "batch_size_abs_difference":float(abs(alone[0]-batch[0])),
            "later_query_perturbation_abs_difference":float(abs(future_changed[0]-batch[0]))}
        print(json.dumps({fold:result[fold]}),flush=True)
    output = {"status":"passed","folds":result,"scope":"8 saved predictions reloaded per fold; first query tested against batch/later-feature perturbation"}
    Path("artifacts/tabpfn-context-verification.json").write_text(json.dumps(output,indent=2)+"\n")


if __name__ == "__main__":
    run()
