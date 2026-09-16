# 발주 예측 프로젝트

고객·상품별 재구매 시점을 우선 예측하고 수량 ±1·±2개 적중률과 불필요한 추천을 함께 평가한다.

## 현재 상태

최신: 사용자 선택에 따라 `xgboost-short-v2`를 [챔피언으로 등록하고 전체 벤치마크 화면](champion-benchmarks.md)을 만들었다. [규칙적인 품목 이력만 새 학습](evidence/regular-history-v1.md)한 비교도 완료했으며 같은 대상의 ±7일 적중률은 55.2% → 55.9%였다. 챔피언은 유지한다.

2026-09-16 기준. [주기·재구매 확률·TabPFN 비교](evidence/timing-model-comparison-v1.md)에 이어 [고객군별 모델·전체 이력·인공 시나리오](evidence/cadence-groups-v1.md)와 [반복 구매 이력·TabPFN 비교](evidence/repeat-history-v1.md)를 실행했다. 최신 비교는 동일한 90일 관찰 조건이며 이전 60일 평가 및 마감일까지 재구매한 사례만의 평가와 구분한다. [작업 상태](tasks.md)를 따른다. 원본 CSV는 보존했고 기존 최종 시험 구간은 사용 완료했다. 실제 앱 연결과 운영 승인은 미완료다.

현재 결과는 사후 정제 탐색 실험이다. 당시 상태의 가용성과 업무 통과 한도는 미확정이므로 운영 성능으로 해석하지 않는다.

## 읽는 순서

1. [결정 사항과 다음 질문](decisions.md): 확정·제안·미결 상태의 기준 문서.
2. [파이프라인과 아키텍처](architecture.md): 단계별 입력, 출력, 완료 조건.
3. [데이터와 피처 계약](data-contract.md): 행, 열, 정제, 피처의 정의.
4. [평가와 학습 설계](evaluation.md): baseline, Champion, 검증과 과적합 방지.
5. [연구 근거](research.md): 논문과 프로젝트 적용 범위.
6. [에이전트와 스킬 운영](agent-workflow.md): 실행자와 검토자의 책임.
7. [Grafana 추적 설계](observability.md): 학습·과적합·데이터·집단별 성능 패널.
8. [대화 기록](conversations/2026-09-15.md): 사용자와 응답의 원문 보존. 현재 설계와 다른 과거 제안도 남긴다.

## 지금 시작할 일

완료한 모델 비교를 바탕으로 일별 구매 확률·상품 최근 추세의 추가 가치를 검토한다. 앱 저장소·API 위치와 적용 기준이 정해지면 [미리보기 계약](app-integration-contract.md)을 연결하고 실제 미래 예측 로그로 검증한다.

이미 실행 가능한 첫 단계는 구조 점검이다. 저장소 루트에서 실행한다.

```bash
python3 scripts/audit_csv.py datase.csv --output docs/evidence/data-profile.json
```

[집계 결과](evidence/data-profile.json)는 개인정보 원문을 담지 않는 원본 구조 통계다. 재실행 시 이 결과 파일을 갱신하며, 원본 CSV는 변경하지 않는다. 해석은 [데이터 계약](data-contract.md)에 있다.

## 기록 관리

- 대화 원문은 날짜별 파일에 추가하고, 현재 판단은 주제별 문서 본문을 갱신한다.
- 변경된 결정의 이유와 상태는 `decisions.md`에 기록한다. 과거 발언을 현재의 확정사항으로 취급하지 않는다.
- 실험 결과는 데이터 해시·평가 버전·설정·예측값과 함께 저장한다. 수치만 복사해 보고하지 않는다.
- [검토 기록](evidence/review.md)은 실제로 수행한 점검과 남은 제약을 구분한다.
