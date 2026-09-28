#!/usr/bin/env bash
# 예측 배치 배포 — coffee-bean-setting 프로젝트. 소유자 계정으로 저장소 루트에서 실행.
#   bash delivery/deploy.sh          # 전부
#   bash delivery/deploy.sh run      # 수동 1회 실행 + 확인만
# 전제: 이미지 빌드 완료(docker images | grep reorder), SA·버킷 생성 완료(2026-09-28).
set -euo pipefail
P=coffee-bean-setting
R=asia-northeast3
SA=manwol-reorder-batch@$P.iam.gserviceaccount.com
IMG=$R-docker.pkg.dev/$P/omcheck-jobs/reorder:v8
JOB=manwol-reorder

grant() {
  gcloud projects add-iam-policy-binding $P --member serviceAccount:$SA --role roles/datastore.user --condition=None --quiet >/dev/null
  gcloud projects add-iam-policy-binding $P --member serviceAccount:$SA --role roles/bigquery.jobUser --condition=None --quiet >/dev/null
  gcloud storage buckets add-iam-policy-binding gs://omcheck-reorder-state --member serviceAccount:$SA --role roles/storage.objectAdmin >/dev/null
  gcloud projects add-iam-policy-binding manwol-core --member serviceAccount:$SA --role roles/bigquery.dataViewer --condition=None --quiet >/dev/null
  gcloud projects add-iam-policy-binding $P --member serviceAccount:$SA --role roles/bigquery.dataViewer --condition=None --quiet >/dev/null   # omcheck_raw.app_orders
  echo "✅ IAM"
}

push() { docker push "$IMG"; echo "✅ image"; }

job() {
  gcloud run jobs describe $JOB --region $R --project $P >/dev/null 2>&1 && ACTION=update || ACTION=create
  gcloud run jobs $ACTION $JOB --project $P --region $R --image "$IMG" \
    --memory 2Gi --cpu 1 --task-timeout 20m --max-retries 1 --service-account "$SA" \
    --add-volume name=state,type=cloud-storage,bucket=omcheck-reorder-state \
    --add-volume-mount volume=state,mount-path=/state \
    --args="--firestore-collection,cart_suggestions,--app-orders-table"
  gcloud run jobs add-iam-policy-binding $JOB --region $R --project $P --member serviceAccount:$SA --role roles/run.invoker >/dev/null
  echo "✅ job"
}

schedule() {
  gcloud scheduler jobs describe $JOB-daily --location $R --project $P >/dev/null 2>&1 && ACTION=update || ACTION=create
  gcloud scheduler jobs $ACTION http $JOB-daily --project $P --location $R \
    --schedule "30 9 * * *" --time-zone Asia/Seoul --http-method POST \
    --uri "https://$R-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$P/jobs/$JOB:run" \
    --oauth-service-account-email "$SA"
  echo "✅ scheduler 09:30 KST"
}

run() {
  EXEC=$(gcloud run jobs execute $JOB --region $R --project $P --wait --format="value(metadata.name)")
  echo "---- 실행 $EXEC 로그 (요약·경고·오류만)"
  sleep 20   # 로그 반영 지연
  gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=$JOB AND labels.\"run.googleapis.com/execution_name\"=$EXEC" \
    --project $P --limit 500 --format="value(textPayload)" | grep -E '"as_of"|warning|Traceback|Error' | head -8
}

case "${1:-all}" in
  all) grant; push; job; schedule; run ;;
  grant|push|job|schedule|run) "$1" ;;
  *) echo "usage: $0 [all|grant|push|job|schedule|run]"; exit 1 ;;
esac
