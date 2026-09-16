-- Read-only: exact test name as identified by the user; export identifiers only.
WITH latest_members AS (
  SELECT member_code, name
  FROM `manwol-core.manwol_core_mirror.public_imweb_members`
  WHERE channel = 'b2s'
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY member_code ORDER BY updated_at DESC, name DESC
  ) = 1
)
SELECT CAST(member_code AS STRING) AS customer_id
FROM latest_members
WHERE TRIM(name) = '테스트'
ORDER BY customer_id;
