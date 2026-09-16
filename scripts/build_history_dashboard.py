"""Publish aggregate comparison panels from verified saved runs, without customer data."""
import json
from pathlib import Path

from scripts.build_features_and_splits import read_csv
from scripts.build_validation_dashboard import aggregate, panel
from scripts.build_xgboost_dashboard import build as detail_dashboard
from scripts.compare_history_models import RUNS
from scripts.run_baseline import digest, write_json


def build():
    source = Path("artifacts/history-comparison-v1.json")
    report = json.loads(source.read_text())
    names = {"short":"최근 이력", "full":"전체 이력", "recency180":"전체 + 최근 가중치"}
    table = "**동일 최신 스냅샷 · 동일 검증 1,196건 · 학습 완료 · 최종 시험은 별도 화면**\n\n"
    table += "|모델|수량 RMSE (개)|연속 수량 MAE (개)|장바구니 수량 MAE (개)|정수 정확 일치율|날짜 MAE (일)|\n|---|---:|---:|---:|---:|---:|\n"
    for key,item in report["runs"].items():
        m,c=item["metrics"],item["cart_metrics"]
        table+=f"|{names[key]}|{m['quantity_rmse']:.4f}|{m['quantity_mae']:.4f}|{c['quantity_mae']:.4f}|{c['quantity_exact_rate']:.2%}|{m['date_mae_days']:.4f}|\n"
    table += "\n최근 이력 후보를 유지합니다. 전체 이력과 최근 가중치 후보는 대체 모델로 채택하지 않습니다. 최종 시험·운영 승인 결과가 아닙니다.\n\n"
    table += "원본 138,427행 → 식별 불가 고객의 이전 이력 1,148행 별도 보관 → 정책 판정 137,279행. 학습 사례: 최근 6,579 / 전체 43,435. "
    table += "최근=2026-04-09부터, 전체 정제 이력=2024-10-16부터. 학습 마감은 둘 다 2026-07-13입니다. 검증 기간 내 재구매가 관측된 21.23%만 평가했고, 과거 상태 가용성은 미검증입니다."
    panels=[{"id":1,"type":"text","title":"이력을 늘리면 더 정확해지는가?","gridPos":{"x":0,"y":0,"w":24,"h":10},"options":{"mode":"markdown","content":table}}]
    daily={}
    for key,(run,_) in RUNS.items():
        rows=read_csv(Path("artifacts")/run/"validation-predictions.csv")
        daily[key]=aggregate(rows,"2026-07-14","2026-08-14")
    merged=[]
    for i,day in enumerate(daily["short"]):
        row={"origin_on":day["origin_on"],"n":day["n"]}
        for key in RUNS:
            if daily[key][i]["origin_on"]!=day["origin_on"] or daily[key][i]["n"]!=day["n"]:
                raise ValueError("Daily populations differ")
            for metric in ("quantity_mae","date_mae"):
                row[key+"_"+metric]=daily[key][i][metric]
        merged.append(row)
    for number,metric,title,unit in ((2,"quantity_mae","출발 구매일별 수량 MAE","suffix:개"),(3,"date_mae","출발 구매일별 날짜 MAE","suffix:일")):
        p=panel(number,title,"같은 고객·상품·출발일과 정답을 비교한 저장 결과. 빈 날짜는 평가 없음.",[(k+"_"+metric,names[k]) for k in RUNS],merged,10,unit)
        p["gridPos"]["x"]=0 if number==2 else 12
        panels.append(p)
    group_table="**집단 구성은 최근 이력 기준으로 고정했습니다.** 전체 이력으로 바뀐 집단끼리 비교하지 않습니다.\n\n|이력|평가 건|최근 수량 RMSE|전체 수량 RMSE|가중치 수량 RMSE|최근 날짜 MAE|전체 날짜 MAE|가중치 날짜 MAE|\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    for group,items in report["fixed_short_history_groups"].items():
        group_table+=f"|{group}회|{items['short']['n']}|"+"|".join(f"{items[k][m]:.3f}" for m in ("quantity_rmse","date_mae_days") for k in RUNS)+"|\n"
    panels.append({"id":4,"type":"text","title":"같은 이력 집단의 성능 비교","gridPos":{"x":0,"y":19,"w":24,"h":8},"options":{"mode":"markdown","content":group_table}})
    links=[{"title":"선택 후보 최종 시험","type":"link","url":"/d/forecasting-final-test","keepTime":False}]
    for key,(run,_) in RUNS.items():
        uid="forecasting-history-"+key
        base="baseline-short-v2" if key=="short" else "baseline-full-v1"
        detail_dashboard(Path("artifacts")/run,Path("artifacts")/base,Path("monitoring/dashboards")/("history-"+key+".json"),uid,"자동 장바구니 · "+names[key]+" 학습곡선")
        links.append({"title":names[key]+" 학습곡선","type":"link","url":"/d/"+uid,"keepTime":False})
    dashboard={"uid":"forecasting-history-comparison","title":"자동 장바구니 · 이력 확장 비교","schemaVersion":41,"version":1,"refresh":"","timezone":"Asia/Seoul","tags":["forecasting","exploratory"],"time":{"from":"2026-07-14T00:00:00+09:00","to":"2026-08-14T23:59:59+09:00"},"description":"Verified saved comparison; sha256="+digest(source),"links":links,"panels":panels}
    write_json(Path("monitoring/dashboards/history-comparison.json"),dashboard)
    return {"dashboard":"http://127.0.0.1:3008/d/forecasting-history-comparison","runs":len(RUNS)}


if __name__=="__main__":
    print(json.dumps(build(),ensure_ascii=False))
