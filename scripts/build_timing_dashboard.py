"""Aggregate-only Grafana view of completed timing and probability trials."""
import json
from pathlib import Path

from scripts.build_validation_dashboard import panel
from scripts.run_baseline import digest, write_json


def text_panel(i,title,content,y,h):
    return {"id":i,"type":"text","title":title,"gridPos":{"x":0,"y":y,"w":24,"h":h},
            "options":{"mode":"markdown","content":content}}


def build():
    names = ("timing-experiment-v2","survival-experiment-v1","tabpfn-experiment-v1")
    reports = {name:json.loads((Path("artifacts")/name/"report.json").read_text()) for name in names}
    verification = json.loads(Path("artifacts/timing-verification-v1.json").read_text())
    for name in names:
        if verification["reports"][name]["sha256"] != digest(Path("artifacts")/name/"report.json"):
            raise ValueError("Dashboard requires verified report version")
    timing,survival,tab = [reports[n] for n in names]
    best = timing["selected_development_candidate"]
    s_best = survival["selected_development_candidate"]
    km = s_best.split("_")[0]+"_km"
    point = [("개인 주기·직전 수량",timing["candidates"]["w180_pair_median_qlast"]),
             ("XGBoost · 전체 학습 사례",timing["candidates"][best]),
             ("TabPFN v2 · 1,000 사례",tab["candidates"]["tabpfn_v2_1000"]),
             ("XGBoost · 동일 1,000 사례",tab["candidates"]["xgboost_matched_1000"])]
    summary = "**모든 비교 실행 완료 · 과거 개발 비교 · 자동 장바구니 적용 전**\n\n"
    summary += "출발 구매 2026년 4·5·6월, 각 구매 뒤 60일 관찰. 18,136건 중 재구매 7,882건, 60일 미구매 10,248건, 보류로 불명 6건. "
    summary += "아래 날짜·수량 점수는 재구매 7,882건에 한정합니다. 과거의 짧은 관찰 기간 MAE와 직접 비교하지 마세요.\n\n"
    summary += "|후보|날짜 MAE|고객 평균 날짜 MAE|날짜 ±7일|수량 ±1개|수량 ±2개|\n|---|---:|---:|---:|---:|---:|\n"
    for title,item in point:
        m = item["overall"]
        summary += f"|{title}|{m['date_mae']:.2f}일|{m['customer_macro_date_mae']:.2f}일|{m['date_within_7_rate']:.1%}|{m['quantity_within_1_rate']:.1%}|{m['quantity_within_2_rate']:.1%}|\n"
    summary += "\nTabPFN은 최신 3.5가 아닌 v2의 제한된 로컬 비교입니다. XGBoost 전체 사례 후보는 32구성 중 고객 평균 날짜 MAE로 선택했습니다. "
    summary += "AI 후보의 수량은 모두 최근 3번 주문 수량의 중앙값 규칙이며 별도 수량 AI 점수가 아닙니다. "
    summary += "MAE 개선이 ±7일 적중률 개선을 보장하지 않습니다. 최신 상태 스냅샷의 과거 가용성은 미검증이며 기존 최종 시험은 사용 완료했습니다."
    panels = [text_panel(1,"후보 비교와 읽는 법",summary,0,11)]
    monthly = []
    for fold in timing["config"]["folds"]:
        row = {"origin_on":fold["origin_start"]}
        for i,(_,item) in enumerate(point):
            for metric in ("date_mae","date_within_7_rate","quantity_within_1_rate"):
                row[f"m{i}_{metric}"] = item["folds"][fold["id"]][metric]
        for key,candidate in [("aft",s_best),("km",km)]:
            row[key] = survival["candidates"][candidate]["folds"][fold["id"]]["landmark_customer_macro_brier"]
        monthly.append(row)
    for i,(metric,title,unit,y) in enumerate([("date_mae","출발 구매월별 날짜 MAE","suffix:일",11),
        ("date_within_7_rate","출발 구매월별 날짜 ±7일 비율","percentunit",11),
        ("quantity_within_1_rate","출발 구매월별 수량 ±1개 비율","percentunit",20)],2):
        p = panel(i,title,"각 점은 한 달의 평가 집계이며 학습 진행 곡선이 아닙니다.",
            [(f"m{j}_{metric}",name) for j,(name,_) in enumerate(point)],monthly,y,unit)
        p["fieldConfig"]["defaults"]["custom"]["axisLabel"] = "비율" if unit == "percentunit" else "일"
        panels.append(p)
    p = panel(5,"출발 구매월별 재구매 확률 Brier","각 구매·고객을 차례로 평균합니다. 낮을수록 좋으며 날짜 MAE와 단위가 다릅니다.",
              [("km","단순 확률 기준"),("aft","생존분석")],monthly,20,"none")
    p["fieldConfig"]["defaults"]["decimals"] = 4
    p["fieldConfig"]["defaults"]["custom"]["axisLabel"] = "확률 오차"
    panels.append(p)
    probability = survival["candidates"][s_best]["probability_metrics"]
    base = survival["candidates"][km]["probability_metrics"]
    content = f"**{s_best} · 고객 평균 Brier {base['landmark_customer_macro_brier']:.5f} → {probability['landmark_customer_macro_brier']:.5f}**\n\n"
    content += "아직 재구매하지 않은 주간 시점에서 향후 7일 구매 확률을 평가합니다. 추천 기회 110,984건, 실제 구매 양성 7,711건입니다. "
    content += "같은 출발 구매의 여러 주차가 포함되므로 아래 값은 실제 장바구니 중복을 제거한 성능이 아닙니다.\n\n"
    content += "|확률 문턱|추천 중 실제 구매 비율 (정밀도)|실제 구매 중 추천한 비율 (재현율)|전체 기회 중 추천 비율|\n|---|---:|---:|---:|\n"
    for threshold,item in probability["thresholds"].items():
        content += f"|{float(threshold):.0%}|{item['precision']:.1%}|{item['recall']:.1%}|{item['alert_rate']:.1%}|\n"
    content += "\n문턱을 높이면 정확한 추천의 비율은 높아지지만 구매를 많이 놓칩니다. 문턱은 진단값이며 운영 기준으로 확정하지 않았습니다. "
    content += "생존분석의 무조건부 날짜 중앙값은 별도 목표이며 날짜 회귀 모델을 그대로 대체하지 않습니다."
    panels.append(text_panel(6,"재구매 확률과 추천 문턱의 관계",content,29,10))
    dashboard = {"uid":"forecasting-timing-models","title":"자동 장바구니 · 주기와 재구매 확률 비교", "schemaVersion":41,"version":1,
        "refresh":"","timezone":"Asia/Seoul","time":{"from":"2026-04-01T00:00:00+09:00","to":"2026-06-30T23:59:59+09:00"},
        "tags":["forecasting","development-backtest"],"panels":panels,
        "description":"Verified saved trial aggregates; no customer records; verification="+digest(Path("artifacts/timing-verification-v1.json")),
        "links":[{"title":"이전 최종 시험 (사용 완료)","type":"link","url":"/d/forecasting-final-test","keepTime":False}]}
    write_json(Path("monitoring/dashboards/timing-models.json"),dashboard)
    print("http://127.0.0.1:3008/d/forecasting-timing-models")


if __name__ == "__main__":
    build()
