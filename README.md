# Inventory_managemant_forecasting_model

고객·상품별 재구매 시점과 수량을 예측해 자동 장바구니에 활용하는 프로젝트입니다. 현재 실험은 시점을 우선하고 수량 ±1·±2개 적중률과 불필요한 추천을 함께 평가합니다.

사용자가 지정한 현재 [챔피언과 등록 방식](docs/champion-benchmarks.md), [전체 벤치마크 Grafana](http://127.0.0.1:3008/d/forecasting-benchmarks)에서 모델 비교를 확인할 수 있습니다. [규칙적인 이력만 학습한 후속 실험](docs/evidence/regular-history-v1.md)도 완료했습니다. 기존 최종 시험은 사용 완료했으며 새 결과는 과거 개발 비교입니다. 실제 앱 연결과 미래 로그 검증은 남아 있습니다. 현재 상태는 [작업 목록](docs/tasks.md)을 따릅니다.

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
