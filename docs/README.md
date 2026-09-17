# 발주 예측 프로젝트 문서

카페 고객×품목의 다음 재구매일과 수량을 예측해, 예상 주의 전주 월요일 발주에 맞춰 장바구니를 준비한다. 모델은 45일 안에 다시 산 경우의 날짜 ±7일 적중으로 평가한다.

## 현재 상태 (2026-09-17)

- 챔피언은 사용자가 지정한 주기 유형 라우터 `multi-agent-router-v1`이다. 일정형은 XGBoost, 점점 짧아짐은 TabPFN, 나머지는 기본 모델 `xgboost-short-v2`가 예측하고, 이 품목의 첫 구매에는 담지 않는다. [등록과 성능](champion-benchmarks.md), [구조와 실험 기록](multi-agent-architecture.md).
- 모델·규칙 변경은 멈추고 앱 연동으로 넘어간다. 실제 앱 쓰기와 운영 승인은 없다. [다음 작업 시작점](handoff.md).
- 모든 수치는 여러 번 본 과거 개발 자료의 결과다. 새 검증은 2026-09-16 일정과 이후 실제 구매의 대조로 한다.

## 읽는 순서

1. [다음 작업 시작점](handoff.md): 현재 모델, 재개 방법, 남은 검수.
2. [결정 사항](decisions.md): 사용자 확정 방향과 미결 질문.
3. [멀티 에이전트 라우터](multi-agent-architecture.md): 트리, 시도한 대안과 수치, 챔피언 등록.
4. [챔피언 등록](champion-benchmarks.md): 등록 형식, 배치 일정 명령, 되돌리는 방법.
5. [앱 연결 계약](app-integration-contract.md): 전주 월요일 일정, 미리보기 입출력.
6. [데이터 계약](data-contract.md), [구매 상태 정책](purchase-policy-v1.md), [피처·시간 분할](feature-split-contract.md): 행·정답·피처의 정의.
7. [평가 설계](evaluation.md), [XGBoost 학습 계약](xgboost-contract.md), [기준 모델 계약](baseline-contract.md): 검증과 과적합 방지.
8. [Grafana 추적](observability.md), [연구 근거](research.md), [작업 목록](tasks.md).

## 실험 기록

진행 순서대로다. 각 문서에 설정·출처 해시와 한계가 있다.

- 정제: [구매 이벤트 검토](evidence/purchase-events-review.md), [전처리 v3](evidence/preprocessing-applied-v3.md), [전처리 v4](evidence/preprocessing-applied-v4.md), [상품 수명 검토](evidence/lifecycle-review.md), [설계 v0 점검](evidence/review.md)
- 초기 모델: [기준 모델](evidence/baseline-v1.md), [첫 XGBoost](evidence/xgboost-v1.md), [전체 이력 비교](evidence/history-comparison-v1.md)([실험 설계](full-history-experiment.md)), [챔피언 오차 진단](evidence/champion-error-diagnosis-v1.md)
- 주기·확률·TabPFN: [종합 결과](evidence/timing-model-comparison-v1.md)([주기 설계](timing-experiment-v2.md), [생존분석 설계](survival-experiment-v1.md)), [고객군](evidence/cadence-groups-v1.md), [반복 이력](evidence/repeat-history-v1.md), [규칙적 이력](evidence/regular-history-v1.md)
- 고객 유형·라우터: [유형·품목 속도](evidence/purchase-segments-v1.md)([설계](purchase-segments-v1.md)), [짧은 이력 비교](evidence/segment-history-comparison-v1.md), [±7일 기준 모델 선택](evidence/multi-agent-router-selection-v1.json), [라우터 전체 기록](multi-agent-architecture.md)

## 대화 기록

[2026-09-15](conversations/2026-09-15.md) · [2026-09-16](conversations/2026-09-16.md) · [2026-09-17](conversations/2026-09-17.md). 과거 발언은 현재 확정사항이 아니며, 현재 판단은 주제별 문서를 따른다.

## 기록 관리

- 대화 원문은 날짜별 파일에 추가하고, 현재 판단은 주제별 문서 본문을 갱신한다.
- 결정의 이유와 상태는 `decisions.md`에 기록한다.
- 실험 결과는 설정·출처 해시·예측값과 함께 `artifacts/`에 저장하고, 문서에는 요약과 한계를 남긴다. 원본 CSV·`data/`·`artifacts/`는 Git에 올리지 않는다.
