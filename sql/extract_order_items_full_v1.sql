-- Read-only full-history snapshot. Same 21 columns as v2.
-- Fixed cutoff preserves the original validation and final-test periods.
-- CLI export must reconcile the returned row count; console previews are not complete exports.
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
)
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
ORDER BY ordered_on, order_id, section_position, item_position;
