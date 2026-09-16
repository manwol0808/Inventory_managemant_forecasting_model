"""Train on past-regular customer-product histories, with fixed-cohort controls."""
import json
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean, median, pstdev

import xgboost as xgb

from scripts.build_features_and_splits import read_csv
from scripts.build_purchase_events import write_csv
from scripts.run_baseline import digest, write_json
from scripts.run_cadence_groups import score
from scripts.run_timing_experiment import training_rows, rounded
from scripts.train_xgboost import fit_schema, matrix


def gap_profile(gaps, cfg):
    recent=gaps[-cfg['recent_gap_count']:]
    cv=pstdev(recent)/mean(recent) if recent else None
    enough=len(recent)>=cfg['minimum_past_gaps']
    return {'recent_gap_count':len(recent),'gap_cv':cv,
            'regular':enough and cv<=cfg['maximum_gap_cv'],
            'status':'insufficient' if not enough else 'regular' if cv<=cfg['maximum_gap_cv'] else 'variable',
            'median_recent_gap':median(recent) if recent else None}


def profiles_at_origin(events, cfg):
    histories=defaultdict(list)
    result={}
    for e in sorted(events,key=lambda r:(r['ordered_on'],r['event_id'])):
        key=e['customer_id'],e['product_id']
        if e['event_state']=='held':
            histories[key]=[]
        elif e['event_state']=='completed':
            h=histories[key]
            h.append(date.fromisoformat(e['ordered_on']))
            gaps=[(b-a).days for a,b in zip(h,h[1:])]
            if any(g<=0 for g in gaps):raise ValueError('Invalid product history')
            result[e['event_id']]=gap_profile(gaps,cfg)
    return result


def run():
    cfgpath=Path('config/regular-history-v1.json')
    cfg=json.loads(cfgpath.read_text())
    protocol=json.loads(Path(cfg['source_protocol']).read_text())
    params=json.loads(Path(protocol['reference_protocol']).read_text())['params']
    root=Path('artifacts/regular-history-v1')
    if root.exists():raise FileExistsError(root)
    baseline_root=Path('artifacts/cadence-groups-full-v1')
    baseline_report=json.loads((baseline_root/'report.json').read_text())
    sourcepath=Path('data/features-splits-full-v1/all-origins-audit.csv')
    events_path=Path('data/purchase-events-full-v1/daily-events.csv')
    assert digest(sourcepath)==baseline_report['provenance']['full_features_sha256']
    assert digest(events_path)==baseline_report['provenance']['events_sha256']
    assert digest(baseline_root/'real-predictions.csv')==baseline_report['files']['real-predictions.csv']
    source=read_csv(sourcepath)
    by_id={r['origin_event_id']:r for r in source}
    profiles=profiles_at_origin(read_csv(events_path),cfg)
    base=[r for r in read_csv(baseline_root/'real-predictions.csv') if r['candidate']=='global']
    root.mkdir()
    write_json(root/'config-frozen.json',cfg)
    def record(row, method, gap, quantity):
        s=by_id[row['origin_event_id']]
        p=profiles[row['origin_event_id']]
        purchases=int(s['episode_purchase_count'])
        return {**row,'candidate':method,'predicted_gap_days':gap,'cart_quantity':quantity,
                'past_purchases':purchases,'purchase_bucket':str(purchases) if purchases<6 else '6+',
                'regularity':p['status'],'past_gap_cv':p['gap_cv'] if p['gap_cv'] is not None else ''}
    records=[record(r,'train_all',int(r['predicted_gap_days']),int(r['cart_quantity'])) for r in base]
    counts={}
    for fold in protocol['folds']:
        pool=training_rows(source,fold['train_end'],protocol['training_window_days'],protocol['horizon_days'])
        train=[r for r in pool if profiles[r['origin_event_id']]['regular']]
        valid_base=[r for r in base if r['fold']==fold['id']]
        valid=[by_id[r['origin_event_id']] for r in valid_base]
        assert train and max(r['target_on'] for r in train)<fold['origin_start']
        assert not {r['origin_event_id'] for r in train}&{r['origin_event_id'] for r in valid}
        work=root/fold['id']
        work.mkdir()
        schema=fit_schema(train)
        write_json(work/'schema.json',schema)
        write_json(work/'training-origin-ids.json',[r['origin_event_id'] for r in train])
        values={}
        for target in ['gap_days','quantity']:
            model=xgb.train({**params,'objective':protocol['specialist_objective'],'max_depth':protocol['specialist_depth']},
                            matrix(train,schema,'target_'+target),protocol['specialist_rounds'])
            model.save_model(work/(target+'.json'))
            saved=xgb.Booster({'nthread':4})
            saved.load_model(work/(target+'.json'))
            values[target]=[rounded(v) for v in saved.predict(matrix(valid,schema))]
        records += [record(r,'train_regular',values['gap_days'][i],values['quantity'][i]) for i,r in enumerate(valid_base)]
        counts[fold['id']]={'all_training_rows':len(pool),'regular_training_rows':len(train),
                            'customers':len({r['customer_id'] for r in train}),
                            'training_target_max':max(r['target_on'] for r in train)}
        print(json.dumps({'fold':fold['id'],**counts[fold['id']]}),flush=True)
    write_csv(root/'predictions.csv',list(records[0]),records)
    candidates={}
    for method in ['train_all','train_regular']:
        rs=[r for r in records if r['candidate']==method]
        candidates[method]={'all':score(rs),
          'regularity':{g:score([r for r in rs if r['regularity']==g]) for g in ['regular','variable','insufficient']},
          'purchase_buckets':{g:score([r for r in rs if r['purchase_bucket']==g]) for g in cfg['exact_purchase_buckets']},
          'regular_purchase_buckets':{g:score([r for r in rs if r['purchase_bucket']==g and r['regularity']=='regular']) for g in cfg['exact_purchase_buckets']},
          'folds_regular':{f['id']:score([r for r in rs if r['fold']==f['id'] and r['regularity']=='regular']) for f in protocol['folds']}}
    report={'version':cfg['version'],'config':cfg,'training_counts':counts,'candidates':candidates,
      'operational_approval':False,'independent_final_test':False,
      'limitations':['Regularity is past-only; future target gaps never determine inclusion',
                     'Date/quantity metrics conditional on observed repurchase within90days; coverage separately reported',
                     'Historical state availability unverified; reused development cohorts',
                     'Threshold0.5 and minimum3gaps fixed before this run; no post-score threshold tuning',
                     'Alternating purchase patterns are separated, not deleted or proven invalid',
                     'Scores on variable/insufficient histories are diagnostics outside specialist training scope'],
      'provenance':{'config_sha256':digest(cfgpath),'code_sha256':digest(__file__),
                    'protocol_sha256':digest(cfg['source_protocol']),'source_sha256':digest(sourcepath),
                    'events_sha256':digest(events_path),'baseline_report_sha256':digest(baseline_root/'report.json')},
      'files':{str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}}
    write_json(root/'report.json',report)
    print(json.dumps({m:v['regularity'] for m,v in candidates.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':run()
