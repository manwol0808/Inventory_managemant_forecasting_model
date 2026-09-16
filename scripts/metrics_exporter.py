"""Serve aggregate state only. This container never receives customer-level files."""
import argparse
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

METRICS = ("quantity_rmse", "quantity_mae", "quantity_wape", "quantity_bias_mean", "quantity_exact_rate",
           "date_mae_days", "date_bias_days", "date_within_tolerance_rate")


def render(state):
    lines = []

    def gauge(name, value, labels=""):
        if value is not None and math.isfinite(float(value)):
            lines.append(f"{name}{labels} {float(value)}")

    gauge("forecasting_state_present", state is not None)
    if state is None:
        return "\n".join(lines) + "\n"
    status = state["status"]
    gauge("forecasting_run_active", status == "running")
    gauge("forecasting_run_failed", status == "failed")
    gauge("forecasting_run_completed", status == "completed")
    gauge("forecasting_heartbeat_timestamp_seconds", state["updated_at"])
    gauge("forecasting_last_success_timestamp_seconds", state.get("last_success_timestamp_seconds"))
    gauge("forecasting_run_duration_seconds", state["duration_seconds"])
    gauge("forecasting_guardrail_assessed", state["guardrail_status"] != "unassessed")
    # Never synthesize guardrail_pass=0/1 when there has been no assessment.
    for split in ("train", "validation"):
        gauge("forecasting_rows", state.get("rows", {}).get(split), '{split="' + split + '"}')
    coverage = state.get("coverage", {})
    gauge("forecasting_validation_coverage", coverage.get("evaluated_fraction"))
    if status == "completed":
        for metric in METRICS:
            gauge("forecasting_validation_metric", state.get("metrics", {}).get(metric), '{metric="' + metric + '"}')
        for group in ("1", "2", "3-5", "6+"):
            values = state.get("groups", {}).get(group, {})
            for metric in ("n", "quantity_mae", "quantity_rmse", "date_mae_days"):
                gauge("forecasting_history_group_metric", values.get(metric), '{group="' + group + '",metric="' + metric + '"}')
    return "\n".join(lines) + "\n"


def render_xgboost(state):
    if state is None:
        return "forecasting_xgboost_state_present 0\n"
    lines = ["forecasting_xgboost_state_present 1"]

    def gauge(name, value, labels=""):
        if value is not None and math.isfinite(float(value)):
            lines.append(f"forecasting_xgboost_{name}{labels} {float(value)}")

    for status in ("running", "completed", "failed"):
        gauge("run_" + status, state["status"] == status)
    gauge("updated_timestamp_seconds", state["updated_at"])
    gauge("guardrail_assessed", state["guardrail_status"] != "unassessed")
    progress = state.get("training", {})
    if state["status"] == "running" and progress.get("target") in ("quantity", "gap"):
        target = progress["target"]
        gauge("training_round", progress.get("round"), '{target="' + target + '"}')
        if state["status"] == "running":
            for split in ("train_core", "early_stop"):
                for name in ("rmse", "mae"):
                    gauge("training_loss", progress.get("losses", {}).get(split, {}).get(name),
                          '{target="' + target + '",split="' + split + '",metric="' + name + '"}')
    if state["status"] == "completed":
        for version, key in (("continuous", "metrics"), ("cart", "cart_metrics"), ("baseline", "baseline_metrics")):
            for name in METRICS:
                if version == "continuous" and name == "quantity_exact_rate":
                    continue
                gauge("validation_metric", state.get(key, {}).get(name), '{prediction="' + version + '",metric="' + name + '"}')
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--port", type=int, default=9108)
    args = parser.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/metrics":
                self.send_error(404)
                return
            try:
                state = json.loads(args.state.read_text()) if args.state.exists() else None
                xgb_path = args.state.with_name("xgboost-latest.json")
                xgb_state = json.loads(xgb_path.read_text()) if xgb_path.exists() else None
                body = (render(state) + render_xgboost(xgb_state)).encode()
            except (OSError, ValueError, KeyError, TypeError):
                self.send_error(503, "Invalid or unavailable aggregate state")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    ThreadingHTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
