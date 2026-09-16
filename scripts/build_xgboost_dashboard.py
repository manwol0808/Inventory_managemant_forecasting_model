"""Saved training-round curves and paired validation comparison; no synthetic values."""
import argparse
import csv
import io
import json
from datetime import date, timedelta
from pathlib import Path

from scripts.build_features_and_splits import read_csv
from scripts.build_validation_dashboard import SOURCE, aggregate, panel
from scripts.run_baseline import digest, write_json


def learning_panel(number, target, curve):
    metric = curve["metric"]
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["round", "train_core", "early_stop"])
    scores = curve["scores"]
    for i, (train, early) in enumerate(zip(scores["train_core"][metric], scores["early_stop"][metric]), 1):
        writer.writerow([i, train, early])
    # Pin the modern panel schema; otherwise Grafana migrates it as a legacy manual mapping.
    return {"id": number, "type": "xychart", "pluginVersion": "12.1.1", "title": f"{'수량 RMSE' if target == 'quantity' else '간격 MAE'} 학습 곡선 · 선택 {curve['selected_rounds']}회",
            "description": "가로축=실제 boosting 반복 수. 내부 train_core와 early_stop의 후처리 전 오차입니다. 마지막 검증 점수가 아니며 선택된 반복 수로 train_pool에 다시 학습했습니다.",
            "gridPos": {"x": 0 if number == 2 else 12, "y": 7, "w": 12, "h": 9},
            "datasource": SOURCE,
            "targets": [{"refId": "A", "datasource": SOURCE, "scenarioId": "csv_content", "csvContent": output.getvalue(), "dropPercent": 0}],
            "fieldConfig": {"defaults": {"unit": "suffix:개" if target == "quantity" else "suffix:일", "decimals": 2,
                            "min": 0, "color": {"mode": "palette-classic"}, "custom": {"show": "lines", "lineWidth": 2, "pointSize": {"fixed": 2}}},
                            "overrides": [{"matcher": {"id": "byName", "options": "round"}, "properties": [
                                {"id": "unit", "value": "none"}, {"id": "decimals", "value": 0}, {"id": "custom.axisLabel", "value": "학습 반복 수"}]}]},
            "options": {"mapping": "auto", "series": [],
                "legend": {"showLegend": True, "displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "single"}}}


def build(run_dir, baseline_dir, output, uid="forecasting-xgboost", title="자동 장바구니 · XGBoost 학습과 비교"):
    run_dir, baseline_dir, output = map(Path, (run_dir, baseline_dir, output))
    report = json.loads((run_dir / "report.json").read_text())
    if report["version"] != "xgboost-result-v1" or report["final_test_metrics_computed"]:
        raise ValueError("Unexpected run")
    for name in ("validation-predictions.csv", "learning-curves.json"):
        if digest(run_dir / name) != report["files"][name]:
            raise ValueError("Saved artifact hash mismatch")
    if digest(baseline_dir / "report.json") != report["baseline_report_sha256"]:
        raise ValueError("Wrong baseline run")
    base_report = json.loads((baseline_dir / "report.json").read_text())
    if digest(baseline_dir / "validation-predictions.csv") != base_report["files"]["validation-predictions.csv"]:
        raise ValueError("Baseline predictions changed")
    rows, base_rows = read_csv(run_dir / "validation-predictions.csv"), read_csv(baseline_dir / "validation-predictions.csv")
    if {r["origin_event_id"] for r in rows} != {r["origin_event_id"] for r in base_rows}:
        raise ValueError("Mismatched populations")
    curves = json.loads((run_dir / "learning-curves.json").read_text())
    start = (date.fromisoformat(report["split_config"]["train_end"]) + timedelta(days=1)).isoformat()
    end = report["split_config"]["validation_end"]
    daily, base_daily = aggregate(rows, start, end), aggregate(base_rows, start, end)
    for a, b in zip(daily, base_daily):
        if a["origin_on"] != b["origin_on"] or a["n"] != b["n"]:
            raise ValueError("Mismatched daily population")
        a["baseline_quantity_mae"], a["baseline_date_mae"] = b["quantity_mae"], b["date_mae"]
    base, model, cart = report["baseline_metrics"], report["metrics"], report["cart_metrics"]
    text = (f"**{run_dir.name} · 학습 완료 · 검증 {model['n']:,}쌍 · 최종 시험 미평가 · 운영 승인 판단 보류**\n\n"
            "| 지표 | 기준 모델 | XGBoost 연속 수량 | XGBoost 장바구니 정수 수량 |\n|---|---:|---:|---:|\n"
            f"| 수량 RMSE · 주지표 (개) | {base['quantity_rmse']:.4f} | {model['quantity_rmse']:.4f} | {cart['quantity_rmse']:.4f} |\n"
            f"| 수량 MAE (개) | {base['quantity_mae']:.4f} | {model['quantity_mae']:.4f} | {cart['quantity_mae']:.4f} |\n"
            f"| 날짜 MAE (일) | {base['date_mae_days']:.4f} | {model['date_mae_days']:.4f} | {cart['date_mae_days']:.4f} |\n"
            f"| 정수 수량 정확 일치율 | {base['quantity_exact_rate']:.2%} | 해당 없음 | {cart['quantity_exact_rate']:.2%} |\n\n"
            "위쪽 두 선 그래프의 가로축은 **학습 반복 수**, 아래쪽은 **출발 구매일**입니다. 검증 기간 내 재구매가 확인된 21.23%만 평가한 탐색 결과입니다. "
            "선택 편향과 당시 주문 상태의 가용성 미검증이 남아 있습니다. 연속 예측은 최소 1개로 제한하고 장바구니용은 0.5 올림합니다. 이 화면은 저장 결과이며 새로고침이 추가 학습을 뜻하지 않습니다.")
    panels = [{"id": 1, "type": "text", "title": "동일 검증 데이터에서 비교", "gridPos": {"x": 0, "y": 0, "w": 24, "h": 7},
               "options": {"mode": "markdown", "content": text}},
              learning_panel(2, "quantity", curves["quantity"]), learning_panel(3, "gap", curves["gap"]),
              panel(4, "날짜별 수량 MAE · 기준 모델 vs XGBoost", "출발 구매일별 같은 평가 쌍. XGBoost는 연속 수량 기준입니다. 빈 날짜는 평가 없음입니다.",
                    [("baseline_quantity_mae", "기준 모델"), ("quantity_mae", "XGBoost")], daily, 16, "suffix:개"),
              panel(5, "날짜별 날짜 MAE · 기준 모델 vs XGBoost", "출발 구매일별 정수 예측일 오차. 검증 말미의 관측 가능한 재구매 선택 편향에 주의합니다.",
                    [("baseline_date_mae", "기준 모델"), ("date_mae", "XGBoost")], daily, 16, "suffix:일")]
    group_text = "이력 횟수는 예측 시점의 **고객·상품별 관측 구매 횟수**입니다. 단위와 표본 수를 함께 확인하세요.\n\n| 이력 | 평가 쌍 | 기준 수량 RMSE | XGB 수량 RMSE | 기준 날짜 MAE | XGB 날짜 MAE |\n|---|---:|---:|---:|---:|---:|\n"
    for group, item in report["groups"]["history_group"].items():
        b, c = item["baseline_metrics"], item["metrics"]
        group_text += f"| {group}회 | {c['n']} | {b.get('quantity_rmse', 0):.3f}개 | {c.get('quantity_rmse', 0):.3f}개 | {b.get('date_mae_days', 0):.3f}일 | {c.get('date_mae_days', 0):.3f}일 |\n" if c["n"] else f"| {group}회 | 0 | 미평가 | 미평가 | 미평가 | 미평가 |\n"
    panels.append({"id": 6, "type": "text", "title": "이력 집단별 비교 · 통과 판정 아님", "gridPos": {"x": 0, "y": 25, "w": 24, "h": 7}, "options": {"mode": "markdown", "content": group_text}})
    dashboard = {"uid": uid, "title": title, "schemaVersion": 41,
                 "version": 1, "refresh": "", "timezone": "Asia/Seoul", "tags": ["forecasting", "exploratory"],
                 "time": {"from": start + "T00:00:00+09:00", "to": end + "T23:59:59+09:00"},
                 "description": "Saved actual model results; report_sha256=" + digest(run_dir / "report.json"),
                 "links": [{"title": "기준 모델", "type": "link", "url": "/d/forecasting-baseline", "keepTime": False}],
                 "panels": panels}
    write_json(output, dashboard)
    return {"output": str(output), "panels": len(panels), "validation_rows": len(rows)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/baseline-v1"))
    parser.add_argument("--output", type=Path, default=Path("monitoring/dashboards/xgboost.json"))
    args = parser.parse_args()
    print(json.dumps(build(args.run_dir, args.baseline, args.output), ensure_ascii=False))
