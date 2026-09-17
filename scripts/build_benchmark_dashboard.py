"""Champion and every completed benchmark, separated by evaluation contract."""
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from scripts.build_features_and_splits import read_csv
from scripts.build_validation_dashboard import SOURCE, panel as time_panel
from scripts.predict_replenishment import base_champion_path
from scripts.run_baseline import digest, write_json


def csv_data(headers, rows):
    out=io.StringIO(newline='')
    writer=csv.writer(out)
    writer.writerow(headers)
    writer.writerows(rows)
    return out.getvalue()


def text_panel(pid,title,content,y,h=3):
    return {'id':pid,'type':'text','title':title,'gridPos':{'x':0,'y':y,'w':24,'h':h},
            'options':{'mode':'markdown','content':content}}


def bar_panel(pid,title,rows,key,y,x=0,w=12,h=10,unit='percentunit',description=''):
    # One row per candidate. Numeric data stays numeric; no external queries.
    data=csv_data(['모델',title],[[r['label'],r[key]] for r in rows])
    defaults={'unit':unit,'decimals':1 if unit=='percentunit' else 0 if unit=='suffix:건' else 4 if unit=='none' else 2,'min':0,
              'color':{'mode':'fixed','fixedColor':'blue'},
              'custom':{'axisPlacement':'auto','axisLabel':'','fillOpacity':90,'lineWidth':0,'gradientMode':'none'}}
    if unit=='percentunit':defaults['max']=1
    result={'id':pid,'type':'barchart','title':title,'description':description,
            'gridPos':{'x':x,'y':y,'w':w,'h':h},'datasource':SOURCE,
            'targets':[{'refId':'A','datasource':SOURCE,'scenarioId':'csv_content','csvContent':data}],
            'fieldConfig':{'defaults':defaults,'overrides':[]},
            'options':{'orientation':'horizontal','xField':'모델','showValue':'always','stacking':'none',
                       'groupWidth':.78,'barWidth':.9,'barRadius':.12,'xTickLabelRotation':0,
                       'legend':{'showLegend':False},'tooltip':{'mode':'single'}}}
    if key in {'qty_hit2','date_hit7','date_mae','brier','brier60'}:
        result['type']='bargauge'
        result['targets'][0]['csvContent']=csv_data([r['label'] for r in rows],[[r[key] for r in rows]])
        if unit!='percentunit':result['fieldConfig']['defaults']['max']=max(r[key] for r in rows)*1.1
        result['fieldConfig']['defaults']['color']['fixedColor']='green' if key=='qty_hit2' else 'orange' if key in {'date_mae','brier','brier60'} else 'blue'
        result['options']={'orientation':'horizontal','displayMode':'gradient','showUnfilled':True,
                           'reduceOptions':{'calcs':['lastNotNull'],'fields':'','values':False},
                           'namePlacement':'auto','valueMode':'color'}
    return result


def normalize(label,metrics):
    return {'label':label,'n':metrics['n'],'date_hit7':metrics['date_within_7_rate'],
            'qty_hit2':metrics['quantity_within_2_rate'],'date_mae':metrics['date_mae']}


def saved_predictions(root,filename='validation-predictions.csv'):
    root=Path(root)
    report=json.loads((root/'report.json').read_text())
    path=root/filename
    assert digest(path)==report['files'][filename]
    return read_csv(path)


def point_scores(label,rows):
    n=len(rows)
    de=[abs(int(r['predicted_gap_days'])-int(r['target_gap_days'])) for r in rows]
    qe=[abs(float(r.get('cart_quantity',r['predicted_quantity']))-int(r['target_quantity'])) for r in rows]
    return {'label':label,'n':n,'date_hit7':sum(e<=7 for e in de)/n,'qty_hit2':sum(e<=2 for e in qe)/n,'date_mae':sum(de)/n}


def same_truth(rows):
    fields=['customer_id','product_id','origin_on','target_on','target_quantity','target_gap_days']
    return hashlib.sha256(json.dumps(sorted(tuple(r[k] for k in fields) for r in rows)).encode()).hexdigest()


def build():
    registry=json.loads(Path('config/champion.json').read_text())
    champion=base_champion_path()
    reports={}
    names=['timing-experiment-v2','survival-experiment-v1','tabpfn-experiment-v1',
           'cadence-groups-v1','cadence-groups-full-v1','repeat-history-v1','tabpfn-repeat-v1']
    for name in names:
        reports[name]=json.loads((Path('artifacts')/name/'report.json').read_text())
    verification=json.loads(Path('artifacts/timing-verification-v1.json').read_text())
    for name in names[:3]:
        assert verification['reports'][name]['sha256']==digest(Path('artifacts')/name/'report.json')
    for verify_path,keys in [('artifacts/cadence-groups-v1/combined-verification.json',names[3:5]),
                              ('artifacts/repeat-history-v1/verification.json',names[5:])]:
        verified=json.loads(Path(verify_path).read_text())
        assert verified['passed']
        for name in keys:
            assert verified['runs'][name]['report_sha256']==digest(Path('artifacts')/name/'report.json')
    legacy=[]
    truth=None
    for name,label in [('baseline-v1','초기 단순 기준'),('baseline-short-v2','최근 이력 단순 기준'),
                       ('baseline-full-v1','전체 이력 단순 기준'),('xgboost-v1','초기 XGBoost'),
                       ('xgboost-short-v2','★ 챔피언 · 최근 이력 XGBoost'),
                       ('xgboost-full-v1','전체 이력 XGBoost'),('xgboost-full-recency180-v1','전체 + 최근 가중치 XGBoost')]:
        rows=saved_predictions(Path('artifacts')/name)
        signature=same_truth(rows)
        assert truth is None or signature==truth
        truth=signature
        legacy.append(point_scores(label,rows))
    testrows=saved_predictions('artifacts/final-test-v1','predictions.csv')
    baserows=saved_predictions('artifacts/final-test-v1','baseline-predictions.csv')
    assert same_truth(testrows)==same_truth(baserows)
    final=[point_scores('단순 기준 모델',baserows),point_scores('★ 챔피언 · 최근 이력 XGBoost',testrows)]
    manifest=json.loads(Path('data/features-splits-short-v2/report.json').read_text())
    trainpath=Path('data/features-splits-short-v2/train_pool.csv')
    assert digest(trainpath)==manifest['files'][trainpath.name]
    train=read_csv(trainpath)
    counts=Counter('1회' if int(r['episode_purchase_count'])==1 else '2회' if int(r['episode_purchase_count'])==2
                   else '3~5회' if int(r['episode_purchase_count'])<=5 else '6회 이상' for r in train)
    history=[{'label':g,'count':counts[g]} for g in ['1회','2회','3~5회','6회 이상']]
    cohorts={'legacy_validation':legacy,'legacy_test':final}
    timing=reports['timing-experiment-v2']
    sixty=[normalize(name,item['overall']) for name,item in timing['candidates'].items()]
    sixty += [normalize(name,item['overall']) for name,item in reports['tabpfn-experiment-v1']['candidates'].items()]
    assert {r['n'] for r in sixty}=={7882}
    cohorts['mature60']=sixty
    ninety=[]
    for run,prefix in [('cadence-groups-v1','270일'),('cadence-groups-full-v1','전체 이력')]:
        for candidate,label in [('global','전체 모델'),('group_specialists','고객군별 모델')]:
            ninety.append(normalize(prefix+' · '+label,reports[run]['real'][candidate]['overall']))
    repeat=reports['repeat-history-v1']
    for candidate,label in [('train_min2','2회 이상만 학습'),('train_min4','4회 이상만 학습'),('personal_gap_median','개인 품목 주기 중앙값')]:
        ninety.append(normalize(label,repeat['candidates'][candidate]['evaluate_min1']))
    assert {r['n'] for r in ninety}=={7551}
    cohorts['mature90_all']=ninety
    repeats={}
    for minimum in [2,4,6]:
        rs=[]
        for name,label in [('train_all','XGBoost · 전체 학습 사례'),('train_min2','XGBoost · 2회 이상 학습'),
                           ('train_min4','XGBoost · 4회 이상 학습'),('personal_gap_median','개인 품목 주기 중앙값')]:
            rs.append(normalize(label,repeat['candidates'][name][f'evaluate_min{minimum}']))
        for name,label in [('xgboost_matched_1000','XGBoost · 같은 1,000건'),('tabpfn_v2_1000','TabPFN v2 · 1,000건')]:
            rs.append(normalize(label,reports['tabpfn-repeat-v1']['candidates'][name][f'evaluate_min{minimum}']))
        assert len({r['n'] for r in rs})==1
        repeats[minimum]=rs
        cohorts[f'mature90_history{minimum}']=rs
    survival=[{'label':name,'brier':item['probability_metrics']['landmark_customer_macro_brier'],
               'brier60':item['probability_metrics']['brier_60d']} for name,item in reports['survival-experiment-v1']['candidates'].items()]
    panels=[]
    pid=0
    y=0
    def add_text(title,content,h=3):
        nonlocal pid,y
        pid+=1;panels.append(text_panel(pid,title,content,y,h));y+=h
    def bars(title,rows,h=10):
        nonlocal pid,y
        ordered=sorted(rows,key=lambda r:(-r['date_hit7'],r['label']))
        for i,(key,label,unit) in enumerate([('date_hit7','주기 ±7일 · 높을수록 좋음','percentunit'),
                                           ('qty_hit2','수량 ±2개 · 높을수록 좋음','percentunit'),
                                           ('date_mae','날짜 MAE · 낮을수록 좋음','suffix:일')]):
            pid+=1;panels.append(bar_panel(pid,label,ordered,key,y,i*8,8,h,unit,title))
        y+=h
    add_text('CHAMPION  ·  xgboost-short-v2',
        '**사용자가 선택한 현재 챔피언 · 2026-09-16 지정**  |  학습 2026-04-09 → 07-13  |  6,579개 학습 사례\n\n'
        '현재 비교 기준과 로컬 미리보기에 사용하는 모델입니다. 앱 자동 적용은 아직 연결하지 않았습니다. '
        '**서로 다른 평가 구간의 76%와 50%를 한 순위로 비교하지 않습니다.** 아래 구역 안에서 같은 표본의 후보끼리 비교하세요.',4)
    for i,(title,key,unit) in enumerate([('기존 검증 · 주기 ±7일','date_hit7','percentunit'),('기존 시험 · 주기 ±7일','date_hit7','percentunit'),
                                       ('기존 시험 · 수량 ±2개','qty_hit2','percentunit'),('기존 시험 · 날짜 MAE','date_mae','suffix:일')]):
        m=legacy[4] if i==0 else final[1]
        pid+=1
        p=bar_panel(pid,title,[m],key,y,i*6,6,4,unit)
        p['type']='stat'
        p['options']={'reduceOptions':{'calcs':['lastNotNull'],'fields':'','values':False},'textMode':'value','colorMode':'value','graphMode':'none','justifyMode':'center'}
        p['description']='기존 마감일까지 재구매가 확인된 검증 1,196건 / 사용 완료 시험 786건. 60·90일 관찰 점수와 직접 비교 불가.'
        panels.append(p)
    y+=4
    add_text('챔피언은 어떤 이력으로 학습했나?',
        '반복 주기가 여러 번 확인된 사례만으로 학습한 모델이 아닙니다. 예측 시점의 같은 품목 구매 이력 기준으로 '
        '**1회 3,150건 · 2회 1,451건 · 3~5회 1,485건 · 6회 이상 493건**입니다. '
        '모든 학습 사례에는 이후 실제 재구매 정답이 붙지만, 입력 시점에 관측된 구매 간격은 없거나 적을 수 있습니다. 사례 수는 고객 수가 아닙니다.',3)
    pid+=1;panels.append(bar_panel(pid,'학습 사례의 과거 구매 횟수',history,'count',y,0,24,7,'suffix:건'));y+=7
    add_text('01  기존 검증 · 같은 1,196건',
        '2026-07-14~08-14 구매 기준일 · 마감까지 재구매가 확인된 1,196 / 5,633건. '
        '긴 재구매가 빠질 수 있는 평가입니다. 초기 버전과 최근 버전의 정답 일치를 대조했습니다. ★는 현재 챔피언입니다.',3)
    bars('기존 검증 1,196건; 다른 구역과 순위 비교 불가',legacy,11)
    add_text('02  기존 최종 시험 · 같은 786건 · 사용 완료',
        '2026-08-15~09-15 구매 기준일 · 마감 전 재구매 786 / 4,956건. '
        '기준 모델 70.2%, 챔피언 76.3%는 이 구역의 ±7일 점수입니다. 새로운 독립 시험이 아닙니다.',3)
    bars('기존 최종 시험 786건',final,6)
    add_text('03  60일 관찰 · 전체 34구성 비교',
        '2026년 4~6월 구매 기준일 · 18,136건 중 60일 내 재구매 7,882건. XGBoost/규칙 32구성과 TabPFN/동일 표본 XGBoost 2구성을 모두 표시합니다. '
        '`w180/w90`=학습 기준일 범위, `abs/log`=학습 목표, `d`=깊이, `q`=수량 규칙. '
        '**이 구역의 수량은 별도 수량 AI가 아니라 직전 수량 또는 최근 3회 중앙값 규칙입니다.**',4)
    bars('60일 관찰, 같은 재구매 7,882건; 34구성',sixty,30)
    monthly=[]
    selected=timing['selected_development_candidate']
    for f in timing['config']['folds']:
        monthly.append({'origin_on':f['origin_start'],
                        'xgb':timing['candidates'][selected]['folds'][f['id']]['date_within_7_rate'],
                        'tab':reports['tabpfn-experiment-v1']['candidates']['tabpfn_v2_1000']['folds'][f['id']]['date_within_7_rate']})
    pid+=1;p=time_panel(pid,'60일 관찰 · 구매월별 ±7일 추이','평가 구간별 점수이며 학습 진행 곡선이 아닙니다.',
                       [('xgb','XGBoost 선택 후보'),('tab','TabPFN v2 · 1,000건')],monthly,y,'percentunit')
    p['gridPos']={'x':0,'y':y,'w':24,'h':9};p['fieldConfig']['defaults']['max']=1;p['fieldConfig']['defaults']['custom']['axisLabel']='적중률'
    panels.append(p);y+=9
    add_text('04  90일 관찰 · 전체 대상',
        '2026-04-01~05-31 및 06-01~06-16 구매 기준일 · 15,491건 중 90일 내 재구매 7,551건. '
        '같은 모델 설정에서 학습 범위·고객군 분리·반복 이력 제한을 비교했습니다. TabPFN 반복 이력 비교는 다음 구역에 있습니다.',3)
    bars('90일 관찰, 같은 재구매 7,551건',ninety,11)
    for minimum in [2,4,6]:
        n=repeats[minimum][0]['n']
        add_text(f'05-{minimum}  90일 관찰 · 같은 품목 {minimum}회 이상 이력',
            f'동일한 재구매 {n:,}건. 예측 시점의 과거 구매 횟수로 대상을 제한했습니다. '
            '2회·4회·6회 구역은 서로 겹치는 부분집합이며 구역 간 점수 상승을 모델 개선으로 해석하지 않습니다. '
            'TabPFN v2와 “같은 1,000건” XGBoost만 학습 표본도 같습니다. 다른 XGBoost는 더 많은 학습 사례를 사용합니다.',4)
        bars(f'90일 관찰 · 구매 이력 {minimum}회 이상 · {n:,}건',repeats[minimum],10)
    regular_path=Path('artifacts/regular-history-v1/report.json')
    if regular_path.exists():
        regular=json.loads(regular_path.read_text())
        checked=json.loads(Path('artifacts/regular-history-v1/verification.json').read_text())
        assert checked['passed'] and checked['report_sha256']==digest(regular_path)
        regular_rows=[normalize(label,regular['candidates'][name]['regularity']['regular'])
                      for name,label in [('train_all','90일 기준 · 전체 이력 학습'),('train_regular','90일 기준 · 규칙적 이력 학습')]]
        cohorts['mature90_regular']=regular_rows
        add_text('05-R  규칙적인 품목 이력 · 같은 2,732건',
            '과거 구매 4회 이상, 최근 최대 5개 간격의 표준편차/평균 ≤0.5인 사례만 고릅니다. '
            '14·21·28·14일은 포함, 2·60·2·60일은 분리합니다. 미래 정답을 보고 고르지 않습니다. '
            '같은 조건의 대상 3,117건 중 재구매 2,732건을 평가했고 384건 미구매, 1건 결과 불명입니다. '
            '**전체 이력 학습도 이 동일한 규칙적 대상에서 평가합니다.** 비교 기준은 90일 실험의 전체 학습 모델이며 기존 챔피언과 다릅니다.',4)
        bars('규칙적 이력 · 동일한 재구매 2,732건',regular_rows,7)
        content='구매 횟수는 예측 시점의 같은 고객×상품 이력입니다. 2·3회는 규칙성 판단을 유보합니다.\n\n'
        content+='|과거 구매 횟수|규칙적 재구매 평가 건|전체 학습 ±7일|규칙적 학습 ±7일|\n|---|---:|---:|---:|\n'
        for group in ['2','3','4','5','6+']:
            a=regular['candidates']['train_all']['regular_purchase_buckets'][group]
            b=regular['candidates']['train_regular']['regular_purchase_buckets'][group]
            content+=f"|{group}회|{a['n']}|"+(f"{a['date_within_7_rate']:.1%}|{b['date_within_7_rate']:.1%}|\n" if a['n'] else '판단 유보|판단 유보|\n')
        add_text('규칙적인 이력 · 2회 / 3회 / 4회 / 5회 / 6회 이상',content,9)
    add_text('06  재구매 확률 · 생존분석 14구성',
        '날짜 적중률과 다른 목표입니다. 아직 재구매하지 않은 시점의 향후 7일 확률을 평가합니다. '
        '주간 평가 기회 110,984건 · 구매 양성 7,711건. **Brier는 확률 오차이며 낮을수록 좋습니다.** 고객별 동일 가중과 60일 내 구매 확률 점수를 분리합니다.',3)
    for i,(key,title) in enumerate([('brier','주간 확률 · 고객 평균 Brier'),('brier60','60일 내 구매 확률 · Brier')]):
        pid+=1;panels.append(bar_panel(pid,title,sorted(survival,key=lambda r:r['brier']),key,y,i*12,12,17,'none'))
    y+=17
    synthetic=json.loads(Path('artifacts/cadence-groups-v1/synthetic-comparison.json').read_text())
    assert digest('artifacts/cadence-groups-v1/synthetic-comparison.csv')==synthetic['predictions_sha256']
    add_text('07  더미 시나리오 · 실제 고객 점수와 별도',
        '11개 시나리오 × 200건. 모델이 만든 예측값을 정답으로 사용하지 않은 인공 행동 점검입니다. '
        '학습 기간과 인공 입력 분포가 달라 실제 정확도나 전체 모델 순위의 근거가 아닙니다. '
        '아래는 재구매가 있는 10개 시나리오이며, 별도 미구매 200건에는 세 모델 모두 60일 이내 날짜를 출력했습니다.',4)
    labels={'weekly':'매주','fortnightly':'2주','monthly':'매달','every45days':'45일','every75days':'75일',
            'irregular':'불규칙','sudden_speedup':'갑작스런 주기 단축','quantity_spike':'수량 4배',
            'first_purchase':'첫 구매','weekly_shop_monthly_item':'주간 매장·월간 품목'}
    for i,(key,title) in enumerate([('date_within_7_rate','더미 · 주기 ±7일'),('quantity_within_2_rate','더미 · 수량 ±2개')]):
        pid+=1
        p=bar_panel(pid,title,[{'label':'placeholder','value':0}],'value',y,i*12,12,19)
        p['targets'][0]['csvContent']=csv_data(['모델','챔피언','270일 전체 모델','270일 고객군 모델'],
            [[label]+[synthetic['metrics'][m][s][key] for m in ['short_original','global_90d','customer_group_90d']] for s,label in labels.items()])
        p['fieldConfig']['defaults']['color']={'mode':'palette-classic'}
        p['options']['legend']={'showLegend':True,'displayMode':'list','placement':'bottom'}
        panels.append(p)
    y+=19
    add_text('평가 해석과 실행 기록',
        '저장된 실제 실험 결과를 표시하며 새로고침으로 재학습하지 않습니다. 모든 날짜·수량 점수는 각 관찰 기간에 재구매가 확인된 사례의 조건부 점수입니다. '
        '미구매 대상에도 날짜를 출력하는 문제와 과거 상태 가용성 미검증이 남아 있습니다. '
        '챔피언 지정은 사용자 선택이며 모든 조건에서 최고 성능이라는 뜻이 아닙니다. 고객별 원본 정보는 이 화면에 포함하지 않습니다.',4)
    dashboard={'uid':'forecasting-benchmarks','title':'자동 장바구니 · 챔피언 & 모델 벤치마크','schemaVersion':41,'version':1,
               'refresh':'','timezone':'Asia/Seoul','tags':['forecasting','champion','benchmarks'],
               'time':{'from':'2026-04-01T00:00:00+09:00','to':'2026-09-15T23:59:59+09:00'},
               'panels':panels,'description':'Cohort-separated saved benchmarks. Champion registry sha256='+digest('config/champion.json'),
               'links':[{'title':'챔피언 기존 시험','type':'link','url':'/d/forecasting-final-test','keepTime':False},
                        {'title':'학습 곡선','type':'link','url':'/d/forecasting-history-short','keepTime':False}]}
    write_json(Path('monitoring/dashboards/benchmarks.json'),dashboard)
    summary={'champion':registry,'training_history_counts':dict(counts),'training_rows':len(train),'cohorts':cohorts,
             'survival_candidates':survival,'synthetic_models':list(synthetic['metrics']),
             'report_hashes':{n:digest(Path('artifacts')/n/'report.json') for n in names},
             'dashboard_sha256':digest('monitoring/dashboards/benchmarks.json')}
    write_json(Path('docs/evidence/benchmark-dashboard-v1.json'),summary)
    print(json.dumps({'url':'http://127.0.0.1:3008/d/forecasting-benchmarks','panels':len(panels),'cohorts':{k:len(v) for k,v in cohorts.items()},'champion':registry['model_id']},ensure_ascii=False))


if __name__=='__main__':
    build()
