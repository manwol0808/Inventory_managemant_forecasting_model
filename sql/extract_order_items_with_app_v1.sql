-- 웹(아임웹) 주문 + 만월체크 앱 결제 주문. 출력 열은 extract_order_items_full_v1.sql 과 동일.
-- 앱 주문 원천: coffee-bean-setting.omcheck_raw.app_orders (omcheck scripts/bq_daily_sync.sh 가 매일 적재,
-- member_code 는 앱 DB imweb_members 로 이미 m코드 변환됨, 결제분·비테스트만). 이 표가 없으면 이 SQL 은 실패한다 —
-- delivery/run_daily.py 는 --app-orders-table 을 줄 때만 이 파일을 쓴다.
-- 앱 주문 상태 → 아임웹 섹션 상태 대응: paid/preparing → PRODUCT_PREPARATION, shipping → SHIPPING,
-- delivered → SHIPPING_COMPLETE, cancelled → CANCEL_COMPLETE, refunded → RETURN_COMPLETE (주문 단위; 부분 환불은 미구분).
-- 앱 order_id 는 'APP-' 접두로 웹 주문번호와 충돌하지 않는다.
WITH ranked_orders AS (
  SELECT
    order_no, member_code, wtime, updated_at, total_price,
    total_payment_price, sections,
    COUNT(*) OVER (PARTITION BY order_no, updated_at) AS same_update_rows,
    ROW_NUMBER() OVER (
      PARTITION BY order_no
      ORDER BY updated_at DESC, TO_JSON_STRING(sections) DESC,
        CAST(member_code AS STRING) DESC, wtime DESC,
        total_price DESC, total_payment_price DESC
    ) AS version_rank
  FROM `manwol-core.manwol_core_mirror.public_imweb_orders`
  WHERE channel = 'b2s'
), latest_orders AS (
  SELECT * FROM ranked_orders
  WHERE version_rank = 1
    AND member_code IS NOT NULL
    AND DATE(wtime, 'Asia/Seoul') BETWEEN DATE '2024-10-02' AND DATE '2026-09-15'
),
web_items AS (
SELECT
  CURRENT_TIMESTAMP() AS extracted_at,
  CAST(o.member_code AS STRING) AS customer_id,
  CAST(o.order_no AS STRING) AS order_id,
  o.wtime AS ordered_at,
  DATE(o.wtime, 'Asia/Seoul') AS ordered_on,
  o.updated_at AS source_updated_at,
  o.same_update_rows AS latest_timestamp_row_count,
  o.total_price AS order_total,
  o.total_payment_price AS paid_total,
  section_position,
  item_position,
  JSON_VALUE(s, '$.orderSectionCode') AS order_section_code,
  JSON_VALUE(s, '$.orderSectionNo') AS order_section_no,
  JSON_VALUE(s, '$.orderSectionStatus') AS section_status,
  JSON_VALUE(i, '$.orderItemCode') AS order_item_code,
  JSON_VALUE(i, '$.orderSectionItemNo') AS order_section_item_no,
  JSON_VALUE(i, '$.productInfo.prodNo') AS product_id,
  JSON_VALUE(i, '$.productInfo.prodName') AS product_name,
  JSON_VALUE(i, '$.qty') AS quantity_raw,
  SAFE_CAST(JSON_VALUE(i, '$.qty') AS INT64) AS quantity,
  TO_HEX(SHA256(TO_JSON_STRING(i))) AS item_json_sha256
FROM latest_orders o
CROSS JOIN UNNEST(JSON_QUERY_ARRAY(o.sections)) AS s WITH OFFSET section_position
CROSS JOIN UNNEST(JSON_QUERY_ARRAY(s, '$.sectionItems')) AS i WITH OFFSET item_position
),
app_items AS (
  SELECT
    CURRENT_TIMESTAMP() AS extracted_at,
    CAST(a.member_code AS STRING) AS customer_id,
    CONCAT('APP-', a.order_id) AS order_id,
    a.paid_at AS ordered_at,
    DATE(a.paid_at, 'Asia/Seoul') AS ordered_on,
    a.updated_at AS source_updated_at,
    1 AS latest_timestamp_row_count,
    CAST(NULL AS INT64) AS order_total,
    CAST(NULL AS INT64) AS paid_total,
    0 AS section_position,
    item_position,
    'APP' AS order_section_code,
    '0' AS order_section_no,
    CASE a.status
      WHEN 'paid' THEN 'PRODUCT_PREPARATION'
      WHEN 'preparing' THEN 'PRODUCT_PREPARATION'
      WHEN 'shipping' THEN 'SHIPPING'
      WHEN 'delivered' THEN 'SHIPPING_COMPLETE'
      WHEN 'cancelled' THEN 'CANCEL_COMPLETE'
      WHEN 'refunded' THEN 'RETURN_COMPLETE'
      ELSE CONCAT('APP_', UPPER(a.status))
    END AS section_status,
    CONCAT(a.order_id, '-', CAST(item_position AS STRING)) AS order_item_code,
    CAST(item_position AS STRING) AS order_section_item_no,
    JSON_VALUE(i, '$.productId') AS product_id,
    JSON_VALUE(i, '$.productName') AS product_name,
    JSON_VALUE(i, '$.quantity') AS quantity_raw,
    SAFE_CAST(JSON_VALUE(i, '$.quantity') AS INT64) AS quantity,
    TO_HEX(SHA256(TO_JSON_STRING(i))) AS item_json_sha256
  FROM `coffee-bean-setting.omcheck_raw.app_orders` a
  CROSS JOIN UNNEST(JSON_QUERY_ARRAY(a.items)) AS i WITH OFFSET item_position
  WHERE a.member_code IS NOT NULL AND a.paid_at IS NOT NULL
    AND DATE(a.paid_at, 'Asia/Seoul') BETWEEN DATE '2024-10-02' AND DATE '2026-09-15'
)
SELECT * FROM web_items
UNION ALL
SELECT * FROM app_items
ORDER BY ordered_on, order_id, section_position, item_position;
