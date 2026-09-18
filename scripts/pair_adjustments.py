"""고객×품목별 담는 날 보정 저장소.

짝마다 과거 예측 오차(실제 간격 - 예측 간격)를 쌓고, 그 중앙값만큼 담는 날을 당긴다.
앞당김만 적용한다. 늦게 담으면 그 주기는 통째로 0점이지만, 조금 일찍 담는 것은
장바구니에 남아 있으므로 회복 가능하기 때문이다.

배치는 무상태이므로 이 저장소가 실행 사이의 기억을 맡는다.

  python -m scripts.pair_adjustments record <예측CSV> [--db ...]   관측된 결과를 쌓는다
  python -m scripts.pair_adjustments show <매장ID> <품목ID>         한 짝의 보정값을 본다
"""
import argparse
import sqlite3
import statistics
from pathlib import Path

DEFAULT_DB = Path("data/pair-adjustments-v1.sqlite")
DEFAULT_CAP_DAYS = 7
SCHEMA = """
CREATE TABLE IF NOT EXISTS pair_error (
    origin_event_id TEXT PRIMARY KEY,
    customer_id     TEXT NOT NULL,
    product_id      TEXT NOT NULL,
    origin_on       TEXT NOT NULL,
    predicted_gap   INTEGER NOT NULL,
    actual_gap      INTEGER NOT NULL,
    error_days      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS pair_error_pair ON pair_error (customer_id, product_id);
"""


class PairAdjustments:
    """오차 이력을 보관하고 짝별 보정값을 낸다. 보정값은 0 이하(앞당김)뿐이다."""

    def __init__(self, path=DEFAULT_DB, cap_days=DEFAULT_CAP_DAYS):
        if cap_days < 0:
            raise ValueError("Cap must not be negative")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.cap_days = cap_days
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)
        self.db.commit()

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def record(self, rows):
        """관측된 결과를 쌓는다. origin_event_id 로 멱등하므로 재실행해도 중복되지 않는다.

        rows 의 각 항목은 origin_event_id, customer_id, product_id, origin_on,
        predicted_gap_days, target_gap_days 를 가진다. 재구매가 관측되지 않은 건은 건너뛴다.
        """
        payload = []
        for r in rows:
            if not r.get("target_gap_days") or not r.get("predicted_gap_days"):
                continue
            actual, predicted = int(float(r["target_gap_days"])), int(float(r["predicted_gap_days"]))
            if actual <= 0:
                continue                      # 같은 날 재구매는 주기가 아니다
            payload.append((r["origin_event_id"], r["customer_id"], r["product_id"], r["origin_on"],
                            predicted, actual, actual - predicted))
        self.db.executemany("INSERT OR REPLACE INTO pair_error VALUES (?,?,?,?,?,?,?)", payload)
        self.db.commit()
        return len(payload)

    def adjustment(self, customer_id, product_id, before_on=None):
        """이 짝의 보정 일수. 0 이하이며 -cap_days 아래로 내려가지 않는다.

        before_on 을 주면 그 날짜보다 앞선 관측만 쓴다. 과거 시점을 재현할 때 필요하다.
        """
        sql = "SELECT error_days FROM pair_error WHERE customer_id=? AND product_id=?"
        args = [customer_id, product_id]
        if before_on:
            sql += " AND origin_on < ?"
            args.append(before_on)
        errors = [row[0] for row in self.db.execute(sql, args)]
        if not errors:
            return 0
        return int(max(-self.cap_days, min(0, statistics.median(errors))))

    def all_adjustments(self):
        """짝 전체의 보정값. 배치에서 한 번에 조회할 때 쓴다."""
        out = {}
        for customer_id, product_id in self.db.execute(
                "SELECT DISTINCT customer_id, product_id FROM pair_error"):
            value = self.adjustment(customer_id, product_id)
            if value:
                out[(customer_id, product_id)] = value
        return out

    def stats(self):
        pairs = self.db.execute("SELECT COUNT(DISTINCT customer_id || '\\x00' || product_id) FROM pair_error")
        return {"errors": self.db.execute("SELECT COUNT(*) FROM pair_error").fetchone()[0],
                "pairs": pairs.fetchone()[0], "adjusted_pairs": len(self.all_adjustments()),
                "cap_days": self.cap_days}


def demo():
    """저장소가 실제로 앞당기기만 하는지 확인한다."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        with PairAdjustments(Path(tmp) / "t.sqlite") as store:
            def ev(n, predicted, actual, customer="A", product="P"):
                return {"origin_event_id": f"{customer}{product}{n}", "customer_id": customer,
                        "product_id": product, "origin_on": f"2026-01-{n:02d}",
                        "predicted_gap_days": predicted, "target_gap_days": actual}

            assert store.adjustment("A", "P") == 0, "이력이 없으면 보정하지 않는다"

            # 모델이 계속 길게 본다 -> 담기가 늦는다 -> 앞당겨야 한다
            store.record([ev(1, 25, 18), ev(2, 24, 17), ev(3, 26, 19)])
            assert store.adjustment("A", "P") == -7, store.adjustment("A", "P")

            # 한도를 넘지 않는다
            store.record([ev(4, 40, 10), ev(5, 40, 10), ev(6, 40, 10)])
            assert store.adjustment("A", "P") == -7

            # 모델이 짧게 본다 -> 이미 일찍 담고 있다 -> 더 당기지 않는다
            store.record([ev(1, 10, 30, "B"), ev(2, 10, 31, "B"), ev(3, 10, 29, "B")])
            assert store.adjustment("B", "P") == 0

            # 같은 건을 다시 넣어도 이력이 늘지 않는다
            before = store.stats()["errors"]
            store.record([ev(1, 25, 18)])
            assert store.stats()["errors"] == before

            # 관측 시점 제한
            assert store.adjustment("A", "P", before_on="2026-01-01") == 0

            # 재구매가 없거나 같은 날인 건은 쌓지 않는다
            assert store.record([{"origin_event_id": "x", "customer_id": "C", "product_id": "P",
                                  "origin_on": "2026-01-01", "predicted_gap_days": 10,
                                  "target_gap_days": ""}]) == 0
    print("pair_adjustments demo OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("record")
    r.add_argument("predictions", type=Path, help="origin_event_id/customer_id/product_id/origin_on/"
                                                  "predicted_gap_days/target_gap_days 를 가진 CSV")
    r.add_argument("--db", type=Path, default=DEFAULT_DB)
    r.add_argument("--cap-days", type=int, default=DEFAULT_CAP_DAYS)
    s = sub.add_parser("show")
    s.add_argument("customer_id")
    s.add_argument("product_id")
    s.add_argument("--db", type=Path, default=DEFAULT_DB)
    s.add_argument("--cap-days", type=int, default=DEFAULT_CAP_DAYS)
    sub.add_parser("demo")
    args = parser.parse_args()
    if args.command == "demo":
        demo()
    elif args.command == "record":
        import csv
        with PairAdjustments(args.db, args.cap_days) as store:
            with args.predictions.open(newline="", encoding="utf-8") as stream:
                added = store.record(csv.DictReader(stream))
            print({"recorded": added, **store.stats()})
    else:
        with PairAdjustments(args.db, args.cap_days) as store:
            print({"adjustment_days": store.adjustment(args.customer_id, args.product_id)})
