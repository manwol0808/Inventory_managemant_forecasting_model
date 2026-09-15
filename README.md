# Inventory_managemant_forecasting_model

고객·상품별 주문 수량을 XGBoost로 예측하고, 시기와 집단별 성능을 가드레일로 평가하는 프로젝트입니다. 학습·평가 진행은 Grafana로 추적할 계획입니다.

현재는 **설계 v0와 CSV 구조 점검까지 완료**했습니다. 모델 학습과 Grafana 설치는 아직 시작하지 않았습니다.

- [전체 문서와 대화 기록](docs/README.md)
- [내일 검토할 순서와 미결 질문](docs/handoff.md)
- [데이터·아이템 추적·범주형 피처 설계](docs/data-contract.md)
- [평가·과적합 방지·XGBoost 학습 설계](docs/evaluation.md)
- [Grafana 패널과 수집 설계](docs/observability.md)

원본 고객 CSV와 학습 산출물은 Git에 포함하지 않습니다. 집계된 구조 점검 결과는 `docs/evidence/`에 있습니다.

로컬에 `datase.csv`를 준비한 뒤 저장소 루트에서 구조 점검을 재실행할 수 있습니다.

```bash
python3 scripts/audit_csv.py datase.csv --output docs/evidence/data-profile.json
```
