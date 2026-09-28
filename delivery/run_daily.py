"""매일 1회 배치: 주문 미러 → 전처리 → 재구매 알림 제안 표.

  python -m delivery.run_daily --as-of 2026-10-08T06:00:00+09:00 --work /tmp/run --state /data/state \
      [--export-csv <이미 받은 주문 CSV>]          # 없으면 BigQuery에서 직접 뽑는다
      [--firestore-collection cart_suggestions]   # 또는 --bigquery-table proj.dataset.cart_suggestions
      [--store-rhythm <매장 리듬 CSV>]

결과는 항상 <work>/schedule.csv 와 <work>/summary.json 에 남고, 저장소 옵션이 있으면 거기에도 쓴다.
<state> 폴더는 실행 사이에 유지해야 한다(래칫 보정 sqlite, 직전 실행의 schedule.csv).
"""
import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.apply_training_policy import apply_policy                       # noqa: E402
from scripts.build_features_and_splits import build as build_features         # noqa: E402
from scripts.build_features_and_splits import read_csv                          # noqa: E402
from scripts.build_purchase_events import build as build_events, write_csv       # noqa: E402
from scripts.pair_adjustments import PairAdjustments, RATCHET_CAP_DAYS         # noqa: E402
from scripts.prepare_full_history import prepare                              # noqa: E402
from scripts.purchase_segment_features import segment_features                # noqa: E402

KST = timezone(timedelta(hours=9))
HISTORY_START = "2024-10-02"       # 주문 미러에서 뽑는 첫날 (sql/extract_order_items_full_v1.sql 과 같음)
STATIC = ("config/champion.json", "artifacts/router-champion-v1", "artifacts/xgboost-short-v2")
PRODUCT_EXCLUSIONS = REPO/"config/product-exclusions-full-v1.json"
CUSTOMER_EXCLUSIONS = REPO/"data/test-account-exclusions-full-v1.json"
SEGMENT_CONFIG = REPO/"config/purchase-segments-v1.json"
SPLIT_TEMPLATE = REPO/"config/time-split-v1.json"
OUTPUT_FIELDS = ("suggestion_key", "customer_id", "product_id", "cart_at", "cart_quantity",
                 "predicted_repurchase_on", "status", "customer_group", "purchase_stage",
                 "model_version", "ab_group", "batch_run_at")


DEFAULT_APP_ORDERS_TABLE = "coffee-bean-setting.omcheck_raw.app_orders"


def extract_sql(last_day, app_orders_table=None):
    """추출 SQL. 앱 주문 표를 주면 웹+앱 UNION 버전(sql/extract_order_items_with_app_v1.sql)을 쓴다."""
    name = "extract_order_items_with_app_v1.sql" if app_orders_table else "extract_order_items_full_v1.sql"
    sql = (REPO/"sql"/name).read_text()
    if app_orders_table:
        sql = sql.replace(f"`{DEFAULT_APP_ORDERS_TABLE}`", f"`{app_orders_table}`")
    return re.sub(r"BETWEEN DATE '\d{4}-\d{2}-\d{2}' AND DATE '\d{4}-\d{2}-\d{2}'",
                  f"BETWEEN DATE '{HISTORY_START}' AND DATE '{last_day}'", sql)


def extract_bigquery(last_day, out_csv, out_meta, app_orders_table=None):
    """추출 SQL 을 날짜만 바꿔 실행하고 bq CSV 와 같은 모양으로 저장한다."""
    from google.cloud import bigquery
    from google.api_core.exceptions import NotFound
    client = bigquery.Client()
    try:
        rows = list(client.query(extract_sql(last_day, app_orders_table)).result())
    except NotFound:
        if not app_orders_table:
            raise
        print(json.dumps({"warning": "app_orders table missing, web orders only", "table": app_orders_table}))
        rows = list(client.query(extract_sql(last_day)).result())
    fields = list(rows[0].keys())
    def cell(v):
        if isinstance(v, datetime):
            return v.astimezone(timezone.utc).replace(tzinfo=None).isoformat(sep=" ")
        return "" if v is None else str(v)
    write_csv(out_csv, fields, [{k: cell(r[k]) for k in fields} for r in rows])
    out_meta.write_text(json.dumps({"numRows": len(rows), "extracted_for": last_day}))


def split_config(events_dir, path):
    """피처 생성기가 요구하는 달력 설정. 날짜 구간은 메타데이터 열에만 쓰이고 피처 값에는 영향 없다."""
    report = json.loads((events_dir/"report.json").read_text())
    start, end = report["first_ordered_on"], report["last_ordered_on"]
    d = lambda s, n: (datetime.fromisoformat(s) + timedelta(days=n)).date().isoformat()
    cfg = {**json.loads(SPLIT_TEMPLATE.read_text()), "start": start, "train_core_end": start,
           "train_end": d(start, 1), "validation_end": d(start, 2), "test_end": end}
    path.write_text(json.dumps(cfg, indent=2))
    return path


def preprocess(export_csv, export_meta, work, base_start):
    full = work/"data/full-history-v1"
    prepare(export_csv, export_meta, full, train_cutoff="9999-12-31")   # 식별 불가 상품은 격리만, 중단하지 않음
    orders = read_csv(full/"orders.csv")
    write_csv(full/"short-orders.csv", list(orders[0]), [r for r in orders if r["ordered_on"] >= base_start])
    for name, source in (("full-v1", full/"orders.csv"), ("short-v2", full/"short-orders.csv")):
        policy, events, feats = (work/f"data/purchase-policy-{name}", work/f"data/purchase-events-{name}",
                                 work/f"data/features-splits-{name}")
        apply_policy(source, policy, PRODUCT_EXCLUSIONS, CUSTOMER_EXCLUSIONS)
        build_events(policy/"rows.csv", policy/"report.json", events)
        build_features(events, split_config(events, work/f"data/time-split-{name}.json"), feats)
    rows = read_csv(work/"data/features-splits-full-v1/all-origins-audit.csv")
    additions = segment_features(read_csv(work/"data/purchase-events-full-v1/daily-events.csv"),
                                 json.loads(SEGMENT_CONFIG.read_text()))
    if set(additions) != {r["origin_event_id"] for r in rows}:
        raise ValueError("Preprocessing changed eligible origin population")
    enriched = work/"artifacts/purchase-segments-v1/enriched-origins.csv"
    enriched.parent.mkdir(parents=True)
    write_csv(enriched, list({**rows[0], **additions[rows[0]["origin_event_id"]]}),
              [{**r, **additions[r["origin_event_id"]]} for r in rows])


RATCHET_DB = "pair-adjustments.sqlite"


def ratchet_db(state, work):
    """래칫 sqlite 는 작업 폴더의 사본으로 쓴다. Cloud Run 의 /state 는 Cloud Storage FUSE 라 랜덤 쓰기가
    안 된다(sqlite 저널이 OutOfOrderError, 2026-09-28 첫 실행). 끝나면 파일 통째로 복사해 돌려놓는다."""
    local = work/RATCHET_DB
    if local.exists():                      # 같은 실행 안에서는 한 번만 복사 (record → schedule 순서)
        return local
    if (state/RATCHET_DB).exists():
        shutil.copy(state/RATCHET_DB, local)
        import sqlite3
        try:
            ok = sqlite3.connect(local).execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        except sqlite3.DatabaseError:
            ok = False
        if not ok:                          # FUSE 시절 손상본 — 보정 없이 새로 시작 (다음 실행부터 다시 쌓인다)
            print(json.dumps({"warning": "ratchet db corrupt, starting fresh"}))
            local.unlink()
    return local


def record_outcomes(state, work):
    """직전 실행의 제안 중 이번 데이터에서 재구매가 관측된 것을 래칫 저장소에 쌓는다."""
    previous = state/"last-schedule.csv"
    if not previous.exists():
        return 0
    labels = {r["origin_event_id"]: r for r in read_csv(work/"data/purchase-events-full-v1/next-purchase-labels.csv")}
    observed = [{**r, "target_gap_days": labels[r["origin_event_id"]]["target_gap_days"]}
                for r in read_csv(previous) if r["status"] in ("due", "scheduled")
                and r["origin_event_id"] in labels and labels[r["origin_event_id"]]["label_state"] == "observed"]
    with PairAdjustments(ratchet_db(state, work), RATCHET_CAP_DAYS, "ratchet") as store:
        return store.record(observed)


def schedule(work, state, as_of, store_rhythm):
    import os
    for rel in STATIC:
        (work/rel).parent.mkdir(parents=True, exist_ok=True)
        src, dst = REPO/rel, work/rel
        shutil.copytree(src, dst) if src.is_dir() else shutil.copy(src, dst)
    cwd = os.getcwd()
    os.chdir(work)
    try:
        from scripts.router_champion import RouterChampion
        champion = RouterChampion(store_rhythm=store_rhythm, adjustments_db=ratchet_db(state, work),
                                  adjustment_cap=RATCHET_CAP_DAYS, adjustment_mode="ratchet",
                                  lead_policy="proportional", first_purchase="item_median",
                                  second_purchase="last_gap", schedule="days_before", router=False)
        return champion.schedule(as_of, work/"out")
    finally:
        os.chdir(cwd)


def publish(rows, args):
    if args.firestore_collection:
        from google.cloud import firestore
        db, batch, n = firestore.Client(), None, 0
        for r in rows:
            batch = batch or db.batch()
            batch.set(db.collection(args.firestore_collection).document(r["suggestion_key"]), r)
            n += 1
            if n % 400 == 0:
                batch.commit(); batch = None
        if batch:
            batch.commit()
    if args.bigquery_table:
        from google.cloud import bigquery
        client = bigquery.Client()
        job = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND", autodetect=True,
                                     source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON)
        client.load_table_from_json(rows, args.bigquery_table, job_config=job).result()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--as-of", help="KST ISO 시각. 예 2026-10-08T06:00:00+09:00. 생략하면 지금")
    p.add_argument("--work", type=Path, required=True, help="이번 실행 작업 폴더 (비어 있어야 함)")
    p.add_argument("--state", type=Path, required=True, help="실행 사이에 유지하는 폴더")
    p.add_argument("--export-csv", type=Path, help="BigQuery 대신 쓸 주문 CSV (bq 형식)")
    p.add_argument("--store-rhythm", type=Path)
    p.add_argument("--app-orders-table", nargs="?", const=DEFAULT_APP_ORDERS_TABLE,
                   help="만월체크 앱 결제 주문 BigQuery 표. 값 없이 주면 기본 표. 생략하면 웹 주문만")
    p.add_argument("--firestore-collection")
    p.add_argument("--bigquery-table")
    args = p.parse_args()
    as_of = datetime.fromisoformat(args.as_of) if args.as_of else datetime.now(KST)
    if as_of.tzinfo is None:
        raise ValueError("Timezone required")
    as_of = as_of.astimezone(KST)
    work, state = args.work, args.state
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(work)
    work.mkdir(parents=True, exist_ok=True); state.mkdir(parents=True, exist_ok=True)
    export_csv, export_meta = work/"data/export.csv", work/"data/export-meta.json"
    export_csv.parent.mkdir(parents=True)
    if args.export_csv:
        shutil.copy(args.export_csv, export_csv)
        export_meta.write_text(json.dumps({"numRows": len(read_csv(args.export_csv))}))
    else:
        extract_bigquery((as_of - timedelta(days=1)).date().isoformat(), export_csv, export_meta,
                         args.app_orders_table)  # 어제까지 확정분
    base_start = json.loads((REPO/"artifacts/xgboost-short-v2/report.json").read_text())["split_config"]["start"]
    preprocess(export_csv, export_meta, work, base_start)
    recorded = record_outcomes(state, work)
    summary = schedule(work, state, as_of, args.store_rhythm)
    rows = read_csv(work/"out/schedule.csv")
    shutil.copy(work/"out/schedule.csv", work/"schedule.csv")
    shutil.copy(work/"out/schedule.csv", state/"last-schedule.csv")
    if (work/RATCHET_DB).exists():
        shutil.copy(work/RATCHET_DB, state/RATCHET_DB)   # 통째로 순차 쓰기 — FUSE 에서도 안전
    run_at = datetime.now(KST).isoformat(timespec="seconds")
    published = [{k: (int(r[k]) if k == "cart_quantity" else r[k]) for k in OUTPUT_FIELDS if k != "batch_run_at"}
                 | {"batch_run_at": run_at} for r in rows if r["status"] != "closed"]
    publish(published, args)
    summary |= {"batch_run_at": run_at, "published_rows": len(published), "ratchet_recorded": recorded}
    (work/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: summary[k] for k in ("as_of", "rows", "status", "published_rows", "ratchet_recorded")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
