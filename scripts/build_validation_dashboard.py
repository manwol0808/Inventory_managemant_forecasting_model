"""Render saved validation predictions as dated aggregate charts, without retraining."""
import argparse
import csv
import io
import json
import math
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from scripts.run_baseline import digest, write_json

SOURCE = {"type": "grafana-testdata-datasource", "uid": "forecasting-saved-results"}


def aggregate(rows, start, end):
    """Origin-day cohorts, not daily sales; undefined errors stay null on empty days."""
    start, end = date.fromisoformat(start), date.fromisoformat(end)
    if start > end:
        raise ValueError("Invalid validation range")
    by_day = defaultdict(list)
    seen = set()
    for row in rows:
        day = date.fromisoformat(row["origin_on"])
        if not start <= day <= end or row["origin_event_id"] in seen:
            raise ValueError("Out-of-range or duplicate prediction")
        seen.add(row["origin_event_id"])
        by_day[day].append(row)
    result = []
    for offset in range((end - start).days + 1):
        day = start + timedelta(days=offset)
        group = by_day[day]
        actual = [float(r["target_quantity"]) for r in group]
        predicted = [float(r["predicted_quantity"]) for r in group]
        if any(not math.isfinite(x) or x <= 0 for x in actual + predicted):
            raise ValueError("Invalid quantity")
        errors = [p - a for p, a in zip(predicted, actual)]
        days = [abs((date.fromisoformat(r["predicted_on"]) - date.fromisoformat(r["target_on"])).days) for r in group]
        n = len(group)
        result.append({"origin_on": day.isoformat(), "n": n,
                       "actual_mean": sum(actual) / n if n else None,
                       "predicted_mean": sum(predicted) / n if n else None,
                       "quantity_mae": sum(map(abs, errors)) / n if n else None,
                       "quantity_rmse": math.sqrt(sum(e * e for e in errors) / n) if n else None,
                       "date_mae": sum(days) / n if n else None})
    return result


def csv_content(daily, fields):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["Time"] + [title for key, title in fields])
    for row in daily:
        writer.writerow([row["origin_on"] + "T00:00:00+09:00"] + [row[key] for key, title in fields])
    return stream.getvalue()


def panel(number, title, description, fields, daily, y, unit, bars=False):
    return {"id": number, "type": "timeseries", "title": title, "description": description,
            "gridPos": {"x": 0 if number % 2 == 0 else 12, "y": y, "w": 12, "h": 9},
            "datasource": SOURCE,
            "targets": [{"refId": "A", "datasource": SOURCE, "scenarioId": "csv_content",
                         "csvContent": csv_content(daily, fields), "dropPercent": 0}],
            "fieldConfig": {"defaults": {"unit": unit, "decimals": 0 if bars else 2,
                           "min": 0, "color": {"mode": "palette-classic"},
                           "custom": {"drawStyle": "bars" if bars else "line", "lineInterpolation": "linear",
                                      "lineWidth": 2, "fillOpacity": 35 if bars else 6,
                                      "showPoints": "never" if bars else "always", "pointSize": 4,
                                      "spanNulls": False, "axisLabel": "평가 쌍" if bars else "일" if unit == "suffix:일" else "개"}},
                            "overrides": []},
            "options": {"tooltip": {"mode": "multi", "sort": "none"},
                        "legend": {"displayMode": "list", "placement": "bottom", "calcs": []}}}


def build(run_dir, output):
    run_dir, output = Path(run_dir), Path(output)
    report = json.loads((run_dir / "report.json").read_text())
    if report["version"] != "baseline-result-v1" or report["final_test_metrics_computed"]:
        raise ValueError("Only saved baseline validation is supported")
    prediction_file = run_dir / "validation-predictions.csv"
    if digest(prediction_file) != report["files"][prediction_file.name]:
        raise ValueError("Prediction hash mismatch")
    with prediction_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    split = report["split_config"]
    start = (date.fromisoformat(split["train_end"]) + timedelta(days=1)).isoformat()
    end = split["validation_end"]
    daily = aggregate(rows, start, end)
    n = sum(r["n"] for r in daily)
    if not n or n != report["metrics"]["n"]:
        raise ValueError("Saved evaluation count mismatch")
    for daily_key, report_key, squared in [("quantity_mae", "quantity_mae", False),
                                          ("quantity_rmse", "quantity_rmse", True),
                                          ("date_mae", "date_mae_days", False)]:
        value = sum((r[daily_key] ** (2 if squared else 1)) * r["n"] for r in daily if r["n"]) / n
        if squared:
            value = math.sqrt(value)
        if not math.isclose(value, report["metrics"][report_key], rel_tol=1e-10):
            raise ValueError("Daily aggregates do not reconcile with saved score")
    explanation = (
        f"**{run_dir.name} · 저장된 검증 결과 · {start} ~ {end} · {n:,}쌍**\n\n"
        "가로축은 **예측의 출발 구매일(한국 날짜)**입니다. 평가 쌍은 같은 고객·상품의 출발 구매와 다음 구매 한 쌍입니다. "
        "당일 전체 이력을 다음 날 자정에 알았다고 가정해 나중에 계산한 결과이며, 당시 실시간 예측 기록은 아닙니다. "
        "점 하나는 그날 출발한 평가 쌍들의 집계입니다. "
        "실제·예측 수량은 해당 쌍의 **다음 구매 수량**이며, 그 날짜의 전체 판매량이 아닙니다. 학습 진행 곡선도 아닙니다.\n\n"
        "검증 마감 전 재구매가 확인된 사례만 포함합니다. 뒤쪽 날짜일수록 긴 간격 사례가 빠지므로 오차 감소를 성능 향상으로 단정하지 마세요. "
        "평가 0건은 오차를 비워 두고 선을 연결하지 않습니다. 수량 평균 두 선이 겹쳐도 개별 오차는 상쇄될 수 있습니다. "
        "표본 수를 함께 보세요. 최신 상태 스냅샷 기반 탐색 결과이며 운영 승인은 미정입니다.\n\n"
        "[실행 상태·전체 점수 화면으로 돌아가기](/d/forecasting-baseline)"
    )
    panels = [{"id": 1, "type": "text", "title": "검증 추이 읽는 법", "gridPos": {"x": 0, "y": 0, "w": 24, "h": 6},
               "options": {"mode": "markdown", "content": explanation}}]
    panels += [
        panel(2, "출발 구매일별 다음 주문 수량 · 실제와 예측 평균", "같은 평가 쌍의 다음 구매 개수 평균. 전체 판매량이 아니며 평균 차이는 MAE와 다릅니다.",
              [("actual_mean", "실제 다음 수량 평균"), ("predicted_mean", "예측 다음 수량 평균")], daily, 6, "suffix:개"),
        panel(3, "출발 구매일별 수량 오차 · 낮을수록 좋음", "각 예측의 오차를 계산한 뒤 날짜별 집계. RMSE는 큰 오류의 영향을 더 크게 받습니다.",
              [("quantity_mae", "수량 MAE"), ("quantity_rmse", "수량 RMSE")], daily, 6, "suffix:개"),
        panel(4, "출발 구매일별 날짜 오차 · 낮을수록 좋음", "예측한 다음 구매일과 실제 다음 구매일 사이 절대 차이의 평균입니다.",
              [("date_mae", "날짜 MAE")], daily, 15, "suffix:일"),
        panel(5, "출발 구매일별 평가 건수 · 사람 수가 아님", "검증 마감까지 재구매가 확인된 고객·상품 예측 쌍의 수. 평가 건수가 적은 날은 값이 크게 흔들릴 수 있습니다.",
              [("n", "평가 쌍 수")], daily, 15, "none", bars=True),
    ]
    dashboard = {"uid": "forecasting-validation-trends", "title": "자동 장바구니 · 출발 구매일별 검증 추이",
                 "schemaVersion": 41, "version": 1, "refresh": "", "timezone": "Asia/Seoul",
                 "time": {"from": start + "T00:00:00+09:00", "to": end + "T23:59:59+09:00"},
                 "tags": ["forecasting", "exploratory", "saved-validation"], "panels": panels,
                 "links": [{"title": "실행 상태·전체 점수", "type": "link", "url": "/d/forecasting-baseline", "keepTime": False}],
                 "description": f"Actual saved validation aggregates; run={run_dir.name}; prediction_sha256={digest(prediction_file)}; report_sha256={digest(run_dir / 'report.json')}"}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, dashboard)
    return {"days": len(daily), "evaluation_rows": n, "empty_days": [r["origin_on"] for r in daily if not r["n"]],
            "output": str(output), "sha256": digest(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("monitoring/dashboards/validation-trends.json"))
    args = parser.parse_args()
    print(json.dumps(build(args.run_dir, args.output), ensure_ascii=False, indent=2))
