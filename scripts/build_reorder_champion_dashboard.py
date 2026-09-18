"""재구매 장바구니 챔피언(reorder-champion-v4) 화면. 저장된 집계만 싣고 외부 조회는 하지 않는다."""
import csv
import io
import json
from pathlib import Path

SOURCE = {"type": "grafana-testdata-datasource", "uid": "forecasting-saved-results"}
METRICS = Path("artifacts/reorder-champion-v1/metrics.json")
OUTPUT = Path("monitoring/dashboards/reorder-champion.json")


def csv_data(headers, rows):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(headers)
    writer.writerows(rows)
    return stream.getvalue()


def text_panel(pid, title, content, y, h=6, x=0, w=24):
    return {"id": pid, "type": "text", "title": title, "gridPos": {"x": x, "y": y, "w": w, "h": h},
            "options": {"mode": "markdown", "content": content}}


def bars(pid, title, description, headers, rows, y, x=0, w=12, h=10, unit="percentunit", maximum=1.0):
    defaults = {"unit": unit, "decimals": 1 if unit == "percentunit" else 0, "min": 0,
                "color": {"mode": "palette-classic"},
                "custom": {"axisPlacement": "auto", "fillOpacity": 85, "lineWidth": 0, "gradientMode": "none"}}
    if maximum is not None:
        defaults["max"] = maximum
    return {"id": pid, "type": "barchart", "title": title, "description": description,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "datasource": SOURCE,
            "targets": [{"refId": "A", "datasource": SOURCE, "scenarioId": "csv_content",
                         "csvContent": csv_data(headers, rows)}],
            "fieldConfig": {"defaults": defaults, "overrides": []},
            "options": {"orientation": "horizontal", "xField": headers[0], "showValue": "always",
                        "stacking": "none", "groupWidth": 0.78, "barWidth": 0.9, "barRadius": 0.12,
                        "xTickLabelRotation": 0, "legend": {"displayMode": "list", "placement": "bottom"},
                        "tooltip": {"mode": "single"}}}


def table(pid, title, description, headers, rows, y, x=0, w=24, h=9):
    return {"id": pid, "type": "table", "title": title, "description": description,
            "gridPos": {"x": x, "y": y, "w": w, "h": h}, "datasource": SOURCE,
            "targets": [{"refId": "A", "datasource": SOURCE, "scenarioId": "csv_content",
                         "csvContent": csv_data(headers, rows)}],
            "fieldConfig": {"defaults": {"custom": {"align": "auto", "cellOptions": {"type": "auto"}}},
                            "overrides": []},
            "options": {"showHeader": True, "cellHeight": "sm", "footer": {"show": False}}}


def build(metrics=METRICS, output=OUTPUT):
    m = json.loads(Path(metrics).read_text(encoding="utf-8"))
    naive, cur, ch, noadj = m["naive"], m["current"], m["champion"], m["no_adjust"]
    intro = (
        f"**{m['version']} · 등록 {m['as_of']} · 월별 폴드 {m['fold_range']} · {m['rows']:,}건**\n\n"
        "첫 구매를 제외하고 90일 안에 재구매가 관측된 고객×품목 건만 채점했습니다. 학습에 쓰지 않은 기간입니다.\n\n"
        "**앱 연동은 없고 실제로 장바구니에 담은 적도 없습니다.** 과거 주문 이력에 우리 규칙이 정했을 "
        "가상의 담는 날을 계산해, 이미 알고 있는 다음 구매일과 비교한 사후 대조입니다. "
        "담는 날이 실제 구매일보다 뒤면 **늦음**입니다. 따라서 모든 수치는 사장님의 구매 행동이 "
        "장바구니 제안에 영향받지 않는다는 가정 위에 있습니다. 운영 성능은 연동 후 새 로그로만 확인됩니다.\n\n"
        "**담는 날은 기존 규칙(예측 재구매일 7일 전 이전 월요일)을 그대로 씁니다.** 바뀐 것은 세 가지입니다 — "
        "① 수량을 매장 단위 장바구니 모델 값으로 교체, ② 매장 주문 리듬 예측이 더 이르면 최대 7일까지 앞당김, "
        "③ 그 고객×품목의 과거 예측 오차만큼 담는 날을 당기는 짝별 보정"
        f"(앞당김만 허용, 한도 7일, {m['adjusted_share']:.0%}에 적용).\n\n"
        "**간격 예측 모델은 그대로 둡니다.** 모델을 바꾸는 방향 다섯 가지를 시험해 모두 기각했습니다.\n\n"
        "**순진한 규칙**은 예측 없이 구매 다음 월요일에 무조건 담는 방식입니다. 모델이 이것보다 나은지가 판단 기준입니다. "
        "**대기**는 담긴 뒤 실제 구매까지의 일수이고, 짧을수록 제때 담았다는 뜻입니다. "
        "**늦음**은 담기 전에 이미 산 경우로 그 주기는 0점입니다."
    )
    hit = [["늦음 (낮을수록 좋음)", naive["late"], cur["late"], ch["late"]],
           ["담긴 뒤 7일 안에 구매", naive["hit7"], cur["hit7"], ch["hit7"]],
           ["담긴 뒤 14일 안에 구매", naive["hit14"], cur["hit14"], ch["hit14"]],
           ["담긴 뒤 21일 안에 구매", naive["hit21"], cur["hit21"], ch["hit21"]],
           ["수량 정확", naive["qty_exact"], cur["qty_exact"], ch["qty_exact"]],
           ["수량 ±1개", naive["qty_pm1"], cur["qty_pm1"], ch["qty_pm1"]]]
    wait = [["대기 중앙값", naive["wait_median"], cur["wait_median"], ch["wait_median"]],
            ["대기 상위 10% 경계", naive["wait_p90"], cur["wait_p90"], ch["wait_p90"]]]
    gap_rows = [[g["label"], g["n"], f"구매 후 {g['cart_median']:.0f}일", f"{g['wait_median']:.0f}일", g["present"]]
                for g in m["by_gap"]]
    ex_rows = [[f"{e['gap']}일", e["n"], f"구매 후 {e['cart_median']:.0f}일 ({e['cart_min']}~{e['cart_max']}일)",
                f"{e['wait_median']:.0f}일", e["present"]] for e in m["examples"]]
    panels = [
        text_panel(1, "이 화면 읽는 법", intro, 0),
        bars(2, "적중과 수량 · 순진한 규칙 / 현행 / 챔피언", "같은 검증 건을 같은 분모로 비교합니다.",
             ["지표", "순진한 규칙", "현행", "챔피언"], hit, 6),
        bars(3, "담긴 뒤 구매까지 대기 일수 · 짧을수록 좋음",
             "장바구니에 얼마나 오래 머물렀는지. 순진한 규칙은 상위 10%가 57일까지 갑니다.",
             ["지표", "순진한 규칙", "현행", "챔피언"], wait, 6, x=12, unit="suffix:일", maximum=None),
        table(4, "예측 간격 구간별 담기는 시점", "챔피언 기준. 예측 간격이 길수록 늦게 담깁니다.",
              ["예측 간격", "건수", "담는 날", "대기 중앙", "구매 시점에 담겨있음"], gap_rows, 16),
        table(5, "예측 간격별 사례", "라우터가 해당 간격으로 예측한 짝이 실제로 언제 담겼는지입니다.",
              ["예측 간격", "건수", "담는 날", "대기 중앙", "구매 시점에 담겨있음"], ex_rows, 25),
        bars(6, "짝별 보정의 효과", "같은 구성에서 ③번 보정만 켜고 끈 차이입니다. 늦음만 줄이고 적중은 유지합니다.",
             ["지표", "보정 없음", "보정 적용"],
             [["늦음", noadj["late"], ch["late"]],
              ["담긴 뒤 14일 안에 구매", noadj["hit14"], ch["hit14"]],
              ["담긴 뒤 21일 안에 구매", noadj["hit21"], ch["hit21"]]], 34, w=12, h=9),
        bars(9, "월별 폴드 안정성 · 현행 대비 21일 안 적중",
             "폴드 12개 중 10개에서 챔피언이 앞섭니다. 뒤진 두 폴드도 0.3%p 이내입니다.",
             ["폴드", "현행", "챔피언"],
             [[f["fold"], f["current_hit21"], f["champion_hit21"]] for f in m["by_fold"]], 52, w=24, h=11),
        text_panel(7, "모델이 실제로 버는 것", (
            "순진한 규칙(구매 다음 월요일에 무조건 담기) 대비 챔피언의 차이입니다.\n\n"
            f"- 담긴 뒤 14일 안에 구매: **{naive['hit14']:.1%} → {ch['hit14']:.1%}**\n"
            f"- 담긴 뒤 21일 안에 구매: **{naive['hit21']:.1%} → {ch['hit21']:.1%}**\n"
            f"- 대기 상위 10% 경계: **{naive['wait_p90']}일 → {ch['wait_p90']}일**\n"
            f"- 대가는 늦음 {naive['late']:.1%} → {ch['late']:.1%}입니다.\n\n"
            "**약 8%p가 예측 모델의 값입니다.** 특히 꼬리에서 큽니다 — 순진한 규칙은 열 건 중 한 건이 두 달 가까이 "
            "장바구니에 박혀 있는데 챔피언은 44일로 줄입니다. 담는 날을 앞당기는 규칙은 모두 순진한 쪽으로 가는 방향이라 "
            "채택하지 않았습니다."), 34, x=12, w=12, h=9),
        text_panel(8, "확인하지 않은 것", (
            "- 수량은 **기존 담는 날 기준**으로 계산한 값을 재사용했습니다. ②번이 날짜를 당긴 건에 대해서는 "
            "그 시점 피처로 다시 뽑아야 최종값입니다.\n"
            "- 간격 예측 모델을 바꾸는 다섯 가지(분위 0.7, enriched 64피처 교체, 학습 창 확대, 학습 모집단 제한, "
            "매장 리듬을 피처로 투입)를 월별 워크포워드로 시험해 모두 기각했습니다. 분위 0.7은 늦음을 9.5%에서 19.3%로 "
            "두 배로 만들었고, enriched 교체는 12폴드 전체에서 61.9% → 61.6%였습니다.\n"
            "- 매장 주문 리듬 모델의 장바구니 임계값 0.2는 2026-07~08 실험에서 정해져 약한 누수가 남아 있습니다.\n"
            "- 예측 간격 22-30일 구간이 가장 약합니다. 실제 간격 표준편차가 20.5일이며 담는 날 규칙 25가지와 재학습으로도 "
            "개선되지 않았습니다.\n"
            "- 예측을 한 번만 하고 담은 뒤에는 고치지 않습니다. 담기 전 주 단위 재예측은 아직 측정하지 않았습니다.\n"
            "- 담기 전 주 단위 재예측을 세 가지로 구현해 모두 실패했습니다. 피처가 전부 구매 시점 고정값이라 "
            "구매와 구매 사이에 새로 도착하는 정보가 없고, 챔피언이 담는 날에 다시 물어도 모델 답이 같습니다.\n"
            "- 운영 적용과 앱 연동은 하지 않았습니다. 장바구니 삭제 로그가 없어 오래 담겨 있는 비용은 측정하지 못했습니다."),
            43, h=9)]
    dashboard = {"uid": "reorder-champion", "title": "재구매 장바구니 챔피언 · reorder-champion-v4",
                 "tags": ["forecasting", "champion"], "timezone": "Asia/Seoul", "schemaVersion": 39,
                 "version": 4, "editable": False, "refresh": "", "panels": panels,
                 "time": {"from": "now-6M", "to": "now"}}
    Path(output).write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    print(build())
