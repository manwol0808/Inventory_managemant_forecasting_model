# 시작 전 확인 질문

앱 저장소를 직접 보고 1·3·4·6·7 은 코드로 답했습니다(`feat/reorder-suggestions`). 남은 것만 확인하면 됩니다.

1. ~~백엔드 저장소~~ → Firestore, 프로젝트 `coffee-bean-setting`(asia-northeast3). 예측 배치 서비스 계정에 이 프로젝트의 Firestore 쓰기 + `manwol-core` BigQuery 읽기 권한이 필요합니다. **누가 발급하나요?**
2. ~~매장 ID~~ → 앱은 uid 만 쓰고 `manwolConnections`+OTP 관문으로 member_code 를 찾습니다(CF 구현). **앱 결제 주문은 예측 데이터에 아예 없습니다** (2026-09-28 코드·온톨로지·BigQuery 확인):
   - 앱 주문은 아임웹 API 를 타지 않고 앱 DB `omcheck_shop.app_orders` 에만 있습니다(`20260903_하루_흐름_결정사항_검증라운드.md`). `manwol-core` 에도, BigQuery `manwol_core_mirror` 에도 없습니다(`bq ls` 확인: `public_imweb_orders` 뿐). `coffee-bean-setting.omcheck_raw.shopOrders` 는 09-23 에 폐기된 Firestore 미러라 비어 있습니다.
   - `app_orders.member_code` 는 아임웹 uid(이메일). m코드 대응은 `imweb_members(uid, member_code)` 로 가능.
   - 현재 규모: plm 발주서 기준 앱 채널 3건(09-07~09-16, 결제 파이프라인 08-21 개통). 지금은 무시해도 되지만 "만월체크 구매 활성화" OKR 대로 앱 주문이 늘면 **앱에서 산 품목에 또 알림이 가고, 앱 구매 이력이 예측에서 빠집니다.**
   - **조치 완료(코드, 09-28)**: 양쪽 다 만들어 둠. ① 앱 `feat/reorder-suggestions` 브랜치 `scripts/bq_daily_sync.sh` [2b] 단계 — Cloud SQL `app_orders` 를 결제분·비테스트만, `imweb_members` 로 m코드 변환해 `omcheck_raw.app_orders` 로 매일 적재(배송지·연락처·이메일 제외). ② 예측 배치 `--app-orders-table` — `sql/extract_order_items_with_app_v1.sql` 로 웹+앱 UNION, 표 없으면 웹만. 검증: BigQuery 실행(웹 130,648 + 앱 3행, 열 동일), 동결 데이터에 앱 주문 1건 넣어 마지막 구매·알림 시각이 갱신됨 확인.
   - **앱 개발자 확인 필요 2개**: (a) `bq_daily_sync` 컨테이너 SA(`603340940588-compute@`)에 `cloudsql.instances.export` 권한, Cloud SQL 인스턴스 SA 에 `gs://omcheck-bq-export` 쓰기 권한 — 없으면 [2b] 는 경고만 남기고 넘어감. (b) 부분 환불은 주문 상태로만 봄(품목 단위 미구분) — 앱 환불이 품목 단위면 알려주세요.
3. ~~품목 ID~~ → `shopProducts` 문서 ID = 아임웹 `prod_no` = 예측의 `product_id`. 옵션은 제안에 없어 옵션 상품은 담기 때 옵션 시트를 띄웁니다.
4. ~~이벤트 기록~~ → 6종 모두 구현. 검토 포인트: 결제 확정 훅(`shop_home_checkout.dart` onSettled)에서만 남기므로 `/shop/checkout`(상세에서 바로 구매)·주문상세 재결제 경로는 아직 안 남깁니다. 필요하면 같은 한 줄을 그 두 곳에 추가.
5. **실행 주체**: Cloud Run Job `manwol-reorder` 를 기존 `omcheck-bq-sync` 옆에 등록합니다(`RUNTIME.md`). 그쪽에서 등록할지, 저희에게 권한을 줄지?
6. ~~알림 형태~~ → 푸시 + 알림벨 + 식자재몰 홈 "재주문 시기" 카드(담기/안 살래요). 자동 담기는 안 합니다(09-17 에 "최근주문 빠른담기"를 사용자 요청으로 뺀 이력 존중). **UX 검토 부탁.**
7. ~~알림 시각~~ → 배치 09:30, CF 팬아웃·푸시 10:00 KST. 바꾸려면 Scheduler 둘만.
8. **장애 알림**: 배치 실패 시 어디로 알려드리면 되나요(Slack 채널, 이메일)?
