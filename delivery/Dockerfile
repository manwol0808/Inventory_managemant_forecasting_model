# 저장소 루트에서 빌드:  docker build -f delivery/Dockerfile -t manwol-reorder:v8 .
# 실행 예:               docker run --rm -v state:/state manwol-reorder:v8 --as-of 2026-10-08T06:00:00+09:00 --firestore-collection cart_suggestions
FROM python:3.12-slim
WORKDIR /app
COPY delivery/requirements.txt delivery/requirements.txt
RUN pip install --no-cache-dir -r delivery/requirements.txt
# 배치 코드 + 고정 모델·설정. artifacts/·data/ 는 Git 제외라 로컬 체크아웃에 있어야 빌드된다.
COPY scripts scripts
COPY sql sql
COPY config config
COPY delivery delivery
COPY data/test-account-exclusions-full-v1.json data/test-account-exclusions-full-v1.json
COPY artifacts/router-champion-v1 artifacts/router-champion-v1
COPY artifacts/xgboost-short-v2 artifacts/xgboost-short-v2
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "delivery.run_daily", "--work", "/tmp/run", "--state", "/state"]
