# Inventory_managemant_forecasting_model

고객·상품별 재구매 시점과 수량을 예측해 자동 장바구니에 활용하는 프로젝트입니다. 현재 실험은 시점을 우선하고 수량 ±1·±2개 적중률과 불필요한 추천을 함께 평가합니다.

현재 챔피언은 고객×품목의 구매 주기 유형에 따라 XGBoost·TabPFN·기본 XGBoost를 고르는 [멀티 에이전트 라우터](docs/multi-agent-architecture.md)입니다. 장바구니는 예상 주의 전주 월요일에 담고, 이 품목의 첫 구매에는 담지 않습니다. [등록과 성능](docs/champion-benchmarks.md), 앱 연동은 남아 있습니다.

- [전체 문서와 대화 기록](docs/README.md)
- [다음 작업 시작점](docs/handoff.md)
- [데이터·아이템 추적·범주형 피처 설계](docs/data-contract.md)
- [평가·과적합 방지·XGBoost 학습 설계](docs/evaluation.md)
- [Grafana 패널과 수집 설계](docs/observability.md)

원본 고객 CSV와 학습 산출물은 Git에 포함하지 않습니다. 집계된 구조 점검 결과는 `docs/evidence/`에 있습니다.

로컬에 `datase.csv`를 준비한 뒤 저장소 루트에서 구조 점검을 재실행할 수 있습니다.

```bash
python3 scripts/audit_csv.py datase.csv --output docs/evidence/data-profile.json
```
