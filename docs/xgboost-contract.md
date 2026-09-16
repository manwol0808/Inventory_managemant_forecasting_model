# XGBoost 첫 탐색 실험

결과를 보기 전에 [xgboost-v1 설정](../config/xgboost-v1.json)을 고정한다. 현재 21개 피처로 수량과 다음 재구매 간격을 각각 학습한다. 상품 추세·스파이크 피처 추가, 하이퍼파라미터 탐색, 전체 이력 확장은 이번 실행에 포함하지 않는다.

학습 내부 `train_core` 4,654쌍과 `early_stop` 399쌍에서 각각 최적 반복 수를 정한다. 최대 1,000회, 50회 개선 없으면 중단한다. 이후 학습 전체 `train_pool` 6,579쌍에 선택된 반복 수만큼 새로 적합한다. 마지막으로 `validation` 1,196쌍을 한 번 평가한다. 검증 점수로 반복 수를 다시 선택하지 않으며 최종 시험 파일은 열지 않는다. [시간·상태 가용성 한계](feature-split-contract.md)는 그대로 적용한다.

train_pool에는 내부 경계인 06-23을 넘어 정답이 생겼지만 외부 학습 마감 07-13에는 관측된 1,526쌍이 추가로 포함된다. core/early_stop 합계 5,053과 전체 학습 6,579의 차이는 이 내부 경계 초과 사례다. 원본 분할의 포함 조건을 그대로 사용하며 과거 구간에 미래 정답을 소급해 넣지 않는다.

수량은 제곱오차 목적함수와 RMSE로 학습·조기 종료한다. 구매가 발생한 경우의 양수 개수를 예측하므로 연속 예측을 최소 1개로 제한한다. 이 연속값의 RMSE가 주지표이며 MAE·편향 등을 함께 보고한다. 앱 표시용 개수는 별도로 0.5 올림해 최소 1 정수로 만들고, 해당 정수 수량의 RMSE·MAE·정확 일치율도 따로 보고한다. 연속값과 정수값 중 좋은 쪽을 사후에 주지표로 고르지 않는다.

간격은 절대오차 목적함수와 MAE로 학습·조기 종료한다. 예측 후 0.5 올림하고 최소 1일로 제한한다. 예측일은 출발 구매일 + 정수 간격이며 기준 모델과 동일하게 날짜 MAE를 계산한다. 학습 곡선은 후처리 전 내부 학습/조기 종료 점수이고 최종 검증 점수와 다른 역할이다.

상품 코드는 학습 구간에서만 정한 문자열→정수 사전과 `feature_types=c`로 native categorical 처리한다. 숫자의 크기나 순서로 해석하지 않는다. 사전 밖 상품과 결측은 NaN 경로로 예측하고 신규 상품 집단의 점수·표본 수를 별도로 기록한다. 나머지 20개 피처는 numeric, 빈 간격은 NaN이다. 입력 컬럼·순서·사전은 모델과 함께 저장한다. core와 최종 refit은 각 학습 범위의 별도 사전을 사용한다. 고객 ID·주문 ID·정답·날짜 메타데이터는 예측 입력이 아니다.

각 모델은 JSON으로 저장하고 재로딩 예측이 동일한지 확인한다. early stopping 모델의 마지막 트리를 최적 트리로 오인하지 않도록 `best_iteration + 1`개만 추출해 저장하고 최종 재학습 모델의 실제 반복 수도 검사한다. [XGBoost 조기 종료](https://xgboost.readthedocs.io/en/stable/python/python_intro.html#early-stopping), [native categorical](https://xgboost.readthedocs.io/en/release_3.2.0/tutorials/categorical.html).

기준 모델과 원본 연결 키·정답이 같은지 전부 확인한 후 전체와 이력 1/2/3~5/6+ 집단을 비교한다. 주지표 개선만으로 운영에 채택하지 않는다. 업무 가드레일, 당시 상태 가용성, 관측 마감에 따른 짧은 간격 선택 편향은 아직 해결되지 않아 운영 승인은 판단 보류다.

```bash
python3 -m scripts.train_xgboost data/features-splits-v1 --baseline artifacts/baseline-v1 --output artifacts/xgboost-v1
```

기존 출력은 덮어쓰지 않는다. 기준 모델의 모니터링 슬롯을 유지하고 XGBoost 집계는 별도 `artifacts/monitoring/xgboost-latest.json`에 기록한다. `events.jsonl`, `learning-curves.json`, 모델·스키마·예측·보고서로 재현 근거를 남긴다.
