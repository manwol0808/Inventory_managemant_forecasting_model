# 연구 근거와 읽는 순서

확인일: 2026-09-15. 아래는 원 논문, 저자 공개본, 출판사/저자기관 페이지와 저자 교재다. 초록·관련 설명/발췌를 확인한 설계 참고 목록이며 전체 논문 재현 실험을 완료한 목록은 아니다. 검색 결과에 나타난 출판사 설명만 확인된 경우 이를 표시했다.

## 가장 먼저 읽을 자료

| ID | 자료 | 우리 설계와 연결되는 부분 | 적용 범위와 한계 |
|---|---|---|---|
| R01 | Mavrotas (2009), **Effective implementation of the ε-constraint method in Multi-Objective Mathematical Programming problems**. [저자기관](https://dspace.lib.ntua.gr/xmlui/handle/123456789/19533), [출판사](https://doi.org/10.1016/j.amc.2009.03.037) | 하나의 목적을 최적화하고 나머지를 제약으로 두는 방식 | 출판사 검색 발췌와 기관 초록 확인. 우리 main/sub 모델 선정에 연결하는 것은 설계상 적용이며 논문 알고리즘 자체를 구현한다는 뜻은 아님 |
| R02 | Hyndman & Koehler (2006), **Another look at measures of forecast accuracy**. [저자 공개 원고](https://robjhyndman.com/papers/mase.pdf), DOI: 10.1016/j.ijforecast.2006.03.001 | MAPE 등 비율 지표의 문제, MASE 및 기준 예측 대비 평가 | 공개 원고 날짜는 2005년, 논문 출판은 2006년. 원고의 지표 정의·0 분모 설명 확인. 우리의 집단별 baseline 오차비와 MASE는 서로 다른 계산 |
| R03 | Gneiting (2011), **Making and Evaluating Point Forecasts**. [저자 프리프린트](https://arxiv.org/abs/0912.0902) | 평균/중앙값/분위수 중 필요한 결과와 손실함수의 정합성 | 프리프린트 최초 제출 2009년. 초록의 목표 통계량·점수 정합성 확인. WAPE를 무조건 기본 KR로 고정하지 않는 이유 |
| R04 | Hyndman & Athanasopoulos, **Forecasting: Principles and Practice**, §5.10. [Time series cross-validation](https://otexts.com/fpp3/tscv.html) | 예측 시점 이전으로 학습하고 이후를 평가하는 rolling-origin 검증 | 논문이 아닌 교재. 현재 기간/H/라벨 지연에 맞춘 분할은 프로젝트에서 정의해야 함 |

## 모델링 단계에서 읽을 자료

첫 학습 모델은 사용자 선택에 따라 **XGBoost**다. 우선 Chen & Guestrin (2016)의 **XGBoost: A Scalable Tree Boosting System**을 읽는다. [원 논문](https://arxiv.org/abs/1603.02754), DOI: 10.1145/2939672.2939785. 제목·저자·초록을 확인했다. 트리 부스팅 시스템의 근거이며 시계열 피처·시간 검증을 자동으로 해결하는 알고리즘은 아니다. 구현 시 [공식 파라미터](https://xgboost.readthedocs.io/en/stable/parameter.html)와 [조기 종료](https://xgboost.readthedocs.io/en/stable/python/python_intro.html)를 확인한다.

범주형 처리의 구현 근거는 [공식 Categorical Data 문서](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html)다. dtype/enable_categorical, 저장 형식, 재인코딩, 신규 범주 한계를 확인했다. 현재 categorical API 동작을 2016년 원 논문만으로 설명하지 않는다.

| ID | 자료 | 우리 설계와 연결되는 부분 | 적용 범위와 한계 |
|---|---|---|---|
| R05 | Montero-Manso & Hyndman (2021), **Principles and algorithms for forecasting groups of time series: Locality and globality**. [원문 페이지](https://www.sciencedirect.com/science/article/pii/S0169207021000558), [프리프린트](https://arxiv.org/abs/2008.00444) | 여러 짧은 시계열을 함께 학습하는 전역 모델과 개별 모델의 관계 | 초록·서론·결론 관련 부분 확인. 전역 모델을 먼저 비교할 근거이며 이 CSV에서 반드시 더 좋다는 보장은 아님 |
| R06 | Makridakis et al. (2022), **M5 accuracy competition: Results, findings, and conclusions**. [출판사](https://doi.org/10.1016/j.ijforecast.2021.11.013), [공개 원고](https://statmodeling.stat.columbia.edu/wp-content/uploads/2021/10/M5_accuracy_competition.pdf) | 소매 다중 시계열, 트리 기반 전역 모델, 다양한 baseline 비교 | 출판사 검색 발췌 확인, 직접 본문 접속은 실패. LightGBM 등의 결과는 전역 트리 모델의 배경 근거로만 사용하며 XGBoost의 우수성을 직접 증명하지 않음 |
| R07 | Teunter, Syntetos & Babai (2011), **Intermittent demand: Linking forecasting to inventory obsolescence**. [저자기관 공개본](https://pure.rug.nl/ws/portalfiles/portal/145394864/Intermittent_demand_Linking_forecasting_to_inventory_obsolescence.pdf) | 수요 발생 확률과 양수 수량을 구분하는 TSB 접근 | 제목·초록 및 관련 본문 확인. 상품별 기간 수요 baseline 후보. 고객의 정확한 다음 주문 날짜 모델과 동일하지 않음 |
| R08 | Sagawa et al. (2019 프리프린트, 2020 개정), **Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization**. [원 논문](https://arxiv.org/abs/1911.08731) | 평균 성능이 좋아도 일부 집단이 실패하는 문제와 정규화 | 초록 확인. 최악 집단 성능을 보는 근거. 논문의 신경망 학습법과 우리의 동일 집단 baseline 가드레일은 다름 |

## 추가 지표 참고

R09. Hyndman (2025), [WAPE: Weighted Absolute Percentage Error](https://robjhyndman.com/hyndsight/wape.html). 저자 기술 글이며 동료 심사 논문으로 표시하지 않는다. WAPE의 0 분모, 중앙값 최적화, 비정상 시계열에서의 비교 문제, RMSSE 대안을 설명한다. [평가 문서](evaluation.md)의 지표 확정 논의에 사용했다.

## 지금 논의한 방식은 연구에 있는가?

구성 요소별로 관련 연구가 있다. **main + 제약은 R01**, **기준 예측과 비교하는 오차는 R02**, **집단별 실패를 보는 관점은 R08**, **전역/집단 모델의 선택은 R05**와 연결된다. 이를 묶은 현재 설계는 이 프로젝트의 의사결정 규칙이다. 그대로 동일한 방법을 검증한 단일 논문을 찾았다고 주장하지 않는다.

논문으로 정할 수 없는 것은 허용 날짜 오차, 수량 단위, 20%·10%p 같은 KR, 집단의 업무 중요도, 과소/과다 비용이다. 이 항목은 데이터와 운영 정보로 결정한다.

## 연구를 실험으로 연결하는 규칙

각 실험은 `가설 → 참고 자료 → 구현 차이 → 동일한 시간 검증 → 채택/기각 이유`로 기록한다. 연구를 읽었다는 이유로 모델을 추가하지 않는다. baseline을 이기지 못하면 실패 원인을 기록하고 해당 모델을 유지할 필요가 있는지 판단한다.
