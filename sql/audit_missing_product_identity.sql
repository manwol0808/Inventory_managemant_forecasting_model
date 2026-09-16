-- Inspect product metadata only; no customer/contact fields.
WITH latest AS (
  SELECT order_no, wtime, sections
  FROM `manwol-core.manwol_core_mirror.public_imweb_orders`
  WHERE channel = 'b2s'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY order_no ORDER BY updated_at DESC) = 1
)
SELECT TO_JSON_STRING(JSON_QUERY(i, '$.productInfo')) AS product_info,
       TO_JSON_STRING(JSON_KEYS(i)) AS item_keys,
       COUNT(*) AS item_rows
FROM latest,
UNNEST(JSON_QUERY_ARRAY(sections)) s,
UNNEST(JSON_QUERY_ARRAY(s, '$.sectionItems')) i
WHERE DATE(wtime, 'Asia/Seoul') BETWEEN DATE '2024-10-02' AND DATE '2026-09-15'
  AND JSON_VALUE(i, '$.productInfo.prodNo') IS NULL
GROUP BY product_info, item_keys
ORDER BY item_rows DESC
LIMIT 5;
