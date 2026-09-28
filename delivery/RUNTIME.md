# 실행 환경

## 한눈에

| 항목 | 값 |
|---|---|
| 형태 | Docker 컨테이너 1개, 매일 1회 실행 후 종료 (상시 서버 아님) |
| 시각 | **09:30 KST** (알림 시각 09:00 이 지난 제안을 due 로 판정) → 앱 CF `fanoutReorderSuggestions` 가 10:00 에 팬아웃·푸시 |
| 권장 | Cloud Run Job + Cloud Scheduler (Firebase와 같은 GCP 프로젝트) |
| 소요 | 약 1~2분 (로컬 M1 기준 40초, 주문 13.8만 행) |
| 메모리 | 2 GB면 충분 |
| 입력 | BigQuery `manwol-core.manwol_core_mirror.public_imweb_orders` 읽기 |
| 출력 | Firestore `cart_suggestions/{suggestion_key}` (프로젝트 `coffee-bean-setting`) + 컨테이너 안 `/tmp/run/schedule.csv` |
| 상태 | `/state` = Cloud Storage FUSE `gs://omcheck-reorder-state` (수 MB). 래칫 보정 sqlite + 직전 실행 결과. **실행 사이에 유지 필수.** sqlite 는 FUSE 에서 직접 못 쓰므로(랜덤 쓰기 불가) 작업 폴더 사본으로 쓰고 끝에 통째로 복사한다 |

## 빌드

저장소 루트에서. `artifacts/`·`data/`는 Git에 없으므로 모델 파일이 있는 체크아웃에서 빌드해야 한다(우리가 빌드해서 이미지로 전달).

```bash
docker buildx build --platform linux/amd64 -f delivery/Dockerfile -t asia-northeast3-docker.pkg.dev/coffee-bean-setting/omcheck-jobs/reorder:v8 --load .
docker push asia-northeast3-docker.pkg.dev/coffee-bean-setting/omcheck-jobs/reorder:v8
```

## 실행 인자

```
--as-of 2026-10-08T06:00:00+09:00     필수. 실행 시각(KST). "어제까지" 주문을 읽는다
--firestore-collection cart_suggestions   또는
--bigquery-table <project>.<dataset>.cart_suggestions
--store-rhythm /state/store-rhythm.csv    선택. 매장 리듬 모델 CSV (customer_id,product_id,cart_on,cart_quantity)
--app-orders-table                        만월체크 앱 결제 주문(BigQuery coffee-bean-setting.omcheck_raw.app_orders)도 읽는다.
                                          표가 아직 없으면 경고만 남기고 웹 주문만으로 돈다
--export-csv <파일>                       선택. BigQuery 대신 이미 받은 주문 CSV 사용 (테스트용)
```

`--work /tmp/run --state /state`는 ENTRYPOINT에 고정. Cloud Run Job에서는 `/state`에 Cloud Storage 볼륨(FUSE) 또는 Filestore를 마운트한다.

## Cloud Run Job 예시

```bash
gcloud run jobs create manwol-reorder \
  --image asia-northeast3-docker.pkg.dev/coffee-bean-setting/omcheck-jobs/reorder:v8 \
  --region asia-northeast3 --memory 2Gi --cpu 1 --task-timeout 20m --max-retries 1 \
  --service-account manwol-reorder-batch@coffee-bean-setting.iam.gserviceaccount.com \
  --add-volume name=state,type=cloud-storage,bucket=omcheck-reorder-state --add-volume-mount volume=state,mount-path=/state \
  --args="--firestore-collection,cart_suggestions"

gcloud scheduler jobs create http manwol-reorder-daily --location asia-northeast3 \
  --schedule "30 9 * * *" --time-zone Asia/Seoul \
  --uri "https://asia-northeast3-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/coffee-bean-setting/jobs/manwol-reorder:run" \
  --http-method POST --oauth-service-account-email manwol-reorder-batch@coffee-bean-setting.iam.gserviceaccount.com
```

`--as-of`를 생략하면 컨테이너가 "지금(KST)"을 쓴다. Scheduler 가 09:30 에 깨우면 그날 09:00 알림분이 due 가 된다.

## 서비스 계정 권한

SA `manwol-reorder-batch@coffee-bean-setting.iam.gserviceaccount.com` (2026-09-28 생성). 권한 부여는 프로젝트 소유자가 실행:

```bash
P=coffee-bean-setting; SA=manwol-reorder-batch@$P.iam.gserviceaccount.com
gcloud projects add-iam-policy-binding $P --member serviceAccount:$SA --role roles/datastore.user      # Firestore cart_suggestions 쓰기
gcloud projects add-iam-policy-binding $P --member serviceAccount:$SA --role roles/bigquery.jobUser    # BigQuery 쿼리 실행(이 프로젝트에서)
gcloud storage buckets add-iam-policy-binding gs://omcheck-reorder-state --member serviceAccount:$SA --role roles/storage.objectAdmin   # /state 볼륨
gcloud projects add-iam-policy-binding manwol-core --member serviceAccount:$SA --role roles/bigquery.dataViewer   # 주문 미러 읽기(다른 프로젝트)
gcloud projects add-iam-policy-binding $P --member serviceAccount:$SA --role roles/bigquery.dataViewer             # 앱 주문 표 omcheck_raw.app_orders 읽기
gcloud run jobs add-iam-policy-binding manwol-reorder --region asia-northeast3 --project $P --member serviceAccount:$SA --role roles/run.invoker   # Scheduler 가 Job 실행
```

## 매일 안에서 일어나는 일

1. BigQuery에서 2024-10-02 ~ 어제 주문을 다시 뽑는다 (증분 아님, 전체. 13.8만 행, 수 초). `--app-orders-table` 이면 앱 결제 주문(`omcheck_raw.app_orders`, 앱 쪽 `bq_daily_sync.sh` 가 04:00 적재)을 UNION — 앱에서 산 것도 마지막 구매로 잡혀 주기가 다시 계산된다. 대응 규칙은 `sql/extract_order_items_with_app_v1.sql` 머리말.
2. 정제 → 구매 이벤트 → 피처 (전부 순수 Python, 학습 없음).
3. 직전 실행의 제안 중 이번에 재구매가 관측된 것을 래칫 저장소에 기록.
4. XGBoost 예측 + 규칙 → `schedule.csv`.
5. `status != closed` 행만 저장소에 쓴다. 같은 `suggestion_key`는 덮어쓴다.

모델 재학습은 안 한다. 모델 파일은 이미지에 고정(2026-07-13까지 학습). 재학습은 우리가 새 이미지 태그로 전달.

## 장애 시

- Job 실패 = 그날 표 갱신 없음. 앱은 전날 `due` 행을 계속 보게 됨. 하루 이틀 밀려도 큰 문제 아님.
- 재실행은 그냥 다시 Run. 멱등이다(`/state` 기록도 `origin_event_id` 기준 멱등).
- Cloud Monitoring에서 Job 실패 알림 1개 걸어 우리에게 전달(`QUESTIONS.md` 8번).
- 알려진 중단 조건: 주문 행에 상품ID가 비어 있는 매장은 그 매장의 이전 이력이 격리되어 제안이 안 나옴(오류 아님, 로그에 `quarantine_rows`).
- 래칫 sqlite 가 손상되면(로그 `ratchet db corrupt, starting fresh`) 보정 없이 그날 실행하고 다음 날부터 다시 쌓는다. 첫 배포(2026-09-28 14:00) 실행분이 FUSE 랜덤 쓰기 오류로 손상됐고 v8 재빌드로 고쳤다.

## 배포 이력

- 2026-09-28 14:00 KST 첫 실행: Firestore `cart_suggestions` 5,399건(due 4,435 · holdout 274), 웹 주문만(`app_orders` 표 없음 경고). Job `manwol-reorder`, Scheduler `manwol-reorder-daily` 09:30.

## 로컬 검증 (우리 쪽)

동결된 2026-09-15 추출본으로 돌리면 기존 배치와 16,140행 전부 동일해야 한다.

```bash
.venv/bin/python -m delivery.run_daily --as-of 2026-09-16T00:00:00+09:00 \
  --work /tmp/run --state /tmp/state --export-csv data/order-items-full-v1-export.csv
```

도커로 같은 검증:

```bash
docker run --rm -v /tmp/state:/state -v $PWD/data:/data:ro manwol-reorder:v8 \
  --as-of 2026-09-16T00:00:00+09:00 --export-csv /data/order-items-full-v1-export.csv
```
