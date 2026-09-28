# 데이터 계약: 제안 표 1개 · 이벤트 6개 · 규칙 3개

저장소는 Firestore(`coffee-bean-setting`)입니다. 시각은 전부 **KST, 시간대 포함 ISO 8601** (`2026-10-08T09:00:00+09:00`).

**앱은 `cart_suggestions` 를 직접 읽지 않습니다.** 이 표는 아임웹 `member_code` 키라 Firebase uid 와 대조할 수 없어 클라이언트 규칙이 없습니다. 앱 쪽 Cloud Function `fanoutReorderSuggestions`(매일 10:00 KST)가 OTP 검증된 연결(`manwolConnections`)로 uid 를 찾아 `shopReorderSuggestions/{ownerUid}/items/{suggestion_key}` 에 `status=due` 인 것만 복사하고(holdout 제외), 새 제안이 생긴 매장에 푸시·알림벨을 보냅니다. 앱 코드는 `feat/reorder-suggestions` 브랜치에 있습니다.

## 1. 제안 표 `cart_suggestions` (배치가 씀, CF 만 읽음)

문서 ID / 기본 키 = `suggestion_key`. 배치가 매일 전체를 다시 쓰므로 **같은 키가 매일 갱신**됩니다.

| 필드 | 타입 | 설명 |
|---|---|---|
| `suggestion_key` | string | 매장·품목·출발 구매 이벤트로 만든 해시. 모델 버전이 바뀌어도 같은 구매에 대해서는 같은 키 |
| `customer_id` | string | 매장ID. 주문 DB `member_code`와 같은 값 |
| `product_id` | string | 품목ID. 옵션 선택은 앱이 처리 |
| `cart_at` | timestamp | 알림/담기 시각. **이 값으로 판단** (`cart_on` 날짜만 보면 안 됨) |
| `cart_quantity` | int | 담을 수량(옵션 단위 개수) |
| `predicted_repurchase_on` | date | 표시용 예측 재구매일 |
| `status` | string | 아래 표 |
| `customer_group` | string | 매장 유형 6종. 신뢰도 표시용 (`README.md` 4번) |
| `purchase_stage` | string | `first` / `second` / `later`. 몇 번째 구매인지 |
| `model_version` | string | 예: `reorder-champion-v8-norouter`. 나중에 성능 비교 조인 키 |
| `ab_group` | string | `control` / `treatment` / `holdout` |
| `batch_run_at` | timestamp | 이 행을 만든 배치 실행 시각 |

`status` 값:

| 값 | 뜻 | 앱 동작 |
|---|---|---|
| `scheduled` | 알림 시각이 아직 안 옴 | 아무것도 안 함 |
| `due` | 알림 시각이 지남, 아직 유효 | **알림/담기** (규칙 1·2·3 확인 후) |
| `closed` | 감시 기간 지남 | 아무것도 안 함. 담긴 게 있으면 그대로 둠 |
| `no_suggestion_*` | 제안 안 함 (예: 규칙상 제외) | 아무것도 안 함 |

예시 행:

```json
{
  "suggestion_key": "c911d07fb1a4b7de759aa3fc3f86668e7aa60078db0a77dc18237b0ac13e02d3",
  "customer_id": "m202603177c47a6158d565",
  "product_id": "476",
  "cart_at": "2026-10-08T09:00:00+09:00",
  "cart_quantity": 2,
  "predicted_repurchase_on": "2026-10-15",
  "status": "due",
  "customer_group": "stable",
  "purchase_stage": "later",
  "model_version": "reorder-champion-v8-norouter",
  "ab_group": "treatment",
  "batch_run_at": "2026-10-08T06:02:11+09:00"
}
```

## 2. 이벤트 표 `cart_events` (앱·CF 가 씀, 분석이 읽음)

한 행 = 한 사건. 추가만 하고 수정·삭제하지 않습니다. 앱은 `member_code` 를 모르므로 **`userId`(Firebase uid)와 `ownerUid`(매장 사장 uid)** 로 남기고, `customer_id` 는 CF 가 쓰는 `cart_suggested` 에만 있습니다. 분석 시 `manwolConnections`(ownerUid → member_code)로 조인합니다. `source` 는 `app` / `server`.

공통 필드 (모든 이벤트):

| 필드 | 타입 | 설명 |
|---|---|---|
| `event_type` | string | 아래 6종 중 하나 |
| `customer_id` | string | 매장ID (`member_code`) |
| `product_id` | string | 품목ID |
| `occurred_at` | timestamp | 사건 시각 KST |
| `suggestion_key` | string or null | 우리 제안과 관련 있으면 그 키. 직접 담은 품목이면 null |

이벤트별 추가 필드:

| `event_type` | 추가 필드 | 언제 |
|---|---|---|
| `cart_suggested` | `cart_quantity`, `predicted_repurchase_on`, `model_version`, `ab_group`, `customer_id` | **CF 가 씀.** 매장 복사본에 새 키가 생겨 푸시·알림벨을 보낸 순간 |
| `cart_item_removed` | `undone`, `stage` (`suggestion`=카드에서 "안 살래요" / `cart`=담은 뒤 삭제) | 사장님이 우리 제안 품목을 지움 |
| `cart_quantity_edited` | `quantity_before`, `quantity_after` | 사장님이 수량 변경 |
| `cart_checkout` | `order_id`, `source` (`suggested` / `manual`), `quantity` | 결제 확정. **품목마다 한 행.** 제안이 떠 있던 상품은 어디서 담았든 `suggested` |
| `cart_item_added_manually` | 없음 | 사장님이 직접 담음 (우리 제안 아님) |
| `cart_viewed` | `cart_item_count` | 장바구니 화면 열람 |

예시:

```json
{"event_type":"cart_item_removed","customer_id":"m2026...","product_id":"476",
 "occurred_at":"2026-10-08T10:12:44+09:00","suggestion_key":"c911d0...","undone":false}

{"event_type":"cart_checkout","customer_id":"m2026...","product_id":"476",
 "occurred_at":"2026-10-09T21:30:00+09:00","suggestion_key":"c911d0...",
 "order_id":"20261009-000123","source":"suggested"}
```

## 3. 앱이 지킬 규칙

1. **멱등**: `suggestion_key` 하나에 `cart_suggested`는 최대 한 번. 앱 저장소의 고유 키/트랜잭션으로 보장.
2. **사용자 우선**: 그 키에 `cart_item_removed` 또는 `cart_quantity_edited`가 있으면 다시 담거나 수량을 되돌리지 않음.
3. **홀드아웃 차단**: `ab_group = 'holdout'`이면 `status`와 무관하게 알림/담기 안 함. 단 `cart_checkout`, `cart_item_added_manually`, `cart_viewed`는 홀드아웃 매장도 기록(기준선용).

추가로 앱 쪽에서 정할 것 (우리는 의견만):

- 이미 장바구니에 같은 품목이 있을 때 수량을 더할지, 사용자 수량을 유지할지. 권장: 사용자 수량 유지, 알림만.
- 그 품목에 새 주문·취소·반품이 들어오면 기존 제안은 무시. 다음 날 배치가 새 제안을 만듭니다.
- 단종·품절·옵션 미선택 품목은 앱이 걸러냄. 배치는 모름.
## 4. 앱 쪽 구현 위치 (`feat/reorder-suggestions`)

- CF `functions/cart-suggestions.js` + 테스트 `functions/test/cart-suggestions.test.js`
- 규칙 `firestore.rules` (`shopReorderSuggestions` 매장 구성원 read, `cart_events` 본인 create)
- Flutter `lib/models/reorder_suggestion_model.dart`, `lib/providers/reorder_suggestion_provider.dart`, 카드 `lib/screens/shop/shop_home_cart.dart` `_buildReorderSuggestions`
- BigQuery 내보내기 `scripts/bq_collections.txt` 에 세 컬렉션 추가
