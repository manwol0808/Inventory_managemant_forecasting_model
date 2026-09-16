"""Saved final-test metrics, explicitly separate from development validation."""
import json
from datetime import date, timedelta
from pathlib import Path

from scripts.build_features_and_splits import read_csv
from scripts.build_validation_dashboard import aggregate, panel
from scripts.run_baseline import digest, write_json


def build():
    root=Path("artifacts/final-test-v1")
    report=json.loads((root/"report.json").read_text())
    for name in ("predictions.csv","baseline-predictions.csv"):
        if digest(root/name)!=report["files"][name]:
            raise ValueError("Prediction file changed")
    rows=read_csv(root/"predictions.csv")
    base=read_csv(root/"baseline-predictions.csv")
    for r in base:
        r["predicted_on"]=(date.fromisoformat(r["origin_on"])+timedelta(days=int(r["predicted_gap_days"]))).isoformat()
    carts=[{**r,"predicted_quantity":r["cart_quantity"]} for r in rows]
    start,end="2026-08-15","2026-09-15"
    daily,base_daily=aggregate(carts,start,end),aggregate(base,start,end)
    for a,b in zip(daily,base_daily):
        if a["n"]!=b["n"] or a["origin_on"]!=b["origin_on"]:
            raise ValueError("Different test populations")
        a["baseline_quantity_mae"]=b["quantity_mae"]
        a["baseline_date_mae"]=b["date_mae"]
    b,m,c=report["baseline_metrics"],report["metrics"],report["cart_metrics"]
    text="**xgboost-short-v2 · 후보 고정 후 최종 시험 786건 · 자동 장바구니 운영 승인 보류**\n\n"
    text+="|지표|기준 모델|XGBoost 연속 수량|XGBoost 장바구니 정수|\n|---|---:|---:|---:|\n"
    for label,key in (("수량 RMSE (개)","quantity_rmse"),("수량 MAE (개)","quantity_mae"),("날짜 MAE (일)","date_mae_days")):
        text+=f"|{label}|{b[key]:.4f}|{m[key]:.4f}|{c[key]:.4f}|\n"
    text+=f"|정수 수량 정확 일치율|{b['quantity_exact_rate']:.2%}|해당 없음|{c['quantity_exact_rate']:.2%}|\n"
    text+=f"|날짜 ±7일 비율 (설명용)|{b['date_within_tolerance_rate']:.2%}|{m['date_within_tolerance_rate']:.2%}|{c['date_within_tolerance_rate']:.2%}|\n\n"
    text+="학습 마감 2026-07-13, 시험 출발일 2026-08-15~09-15. 시험 마감까지 재구매가 확인된 786 / 4,956건(15.86%)만 평가했습니다. "
    text+="짧은 간격의 관측된 재구매에 치우친 결과이며, 당시 주문 상태 가용성은 미검증입니다. 재고 소진 실측·실시간 운영 성능이 아닙니다. "
    text+="이 시험은 사용 완료했습니다. 모델을 다시 조정하면 새로운 미래 시험 구간이 필요합니다."
    panels=[{"id":1,"type":"text","title":"최종 시간 구간 평가","gridPos":{"x":0,"y":0,"w":24,"h":10},"options":{"mode":"markdown","content":text}},
            panel(2,"출발 구매일별 정수 수량 MAE","장바구니에 넣을 정수 개수 기준. 평가 없음은 빈 값.",[("baseline_quantity_mae","기준 모델"),("quantity_mae","XGBoost 정수")],daily,10,"suffix:개"),
            panel(3,"출발 구매일별 날짜 MAE","늦은 출발일의 낮은 오차는 관측 종료 편향일 수 있습니다.",[("baseline_date_mae","기준 모델"),("date_mae","XGBoost")],daily,10,"suffix:일")]
    groups="|구매 이력|시험 건|기준 수량 MAE|XGB 정수 수량 MAE|기준 날짜 MAE|XGB 날짜 MAE|\n|---|---:|---:|---:|---:|---:|\n"
    for group,vals in report["history_groups"].items():
        b,c=vals["baseline"],vals["cart"]
        groups+=f"|{group}회|{c['n']}|{b['quantity_mae']:.3f}개|{c['quantity_mae']:.3f}개|{b['date_mae_days']:.3f}일|{c['date_mae_days']:.3f}일|\n"
    groups+="\n1·2회 이력의 정수 수량 MAE와 6회 이상 이력의 날짜 MAE는 기준 모델보다 나쁩니다. 전체 평균만으로 모든 고객의 자동 적용을 승인하지 않습니다."
    panels.append({"id":4,"type":"text","title":"집단별 남은 약점","gridPos":{"x":0,"y":19,"w":24,"h":8},"options":{"mode":"markdown","content":groups}})
    p=panel(5,"출발 구매일별 평가 건수","사람 수가 아닌 고객·상품 예측 건수입니다.",[("n","평가 건수")],daily,27,"none",True)
    p["gridPos"].update(x=0,w=24)
    panels.append(p)
    dashboard={"uid":"forecasting-final-test","title":"자동 장바구니 · 최종 시험","schemaVersion":41,"version":1,"refresh":"","timezone":"Asia/Seoul","time":{"from":start+"T00:00:00+09:00","to":end+"T23:59:59+09:00"},"tags":["forecasting","retrospective-test"],"links":[{"title":"이력 확장 검증 비교","type":"link","url":"/d/forecasting-history-comparison","keepTime":False}],"panels":panels,"description":"Saved frozen-candidate holdout; sha256="+digest(root/"report.json")}
    write_json(Path("monitoring/dashboards/final-test.json"),dashboard)
    return {"rows":len(rows),"dashboard":"http://127.0.0.1:3008/d/forecasting-final-test"}


if __name__=="__main__":
    print(json.dumps(build(),ensure_ascii=False))
