"""Separate repeated-history eligibility from the effect of training exclusions."""
import json
from pathlib import Path

import xgboost as xgb

from scripts.build_features_and_splits import read_csv
from scripts.build_purchase_events import write_csv
from scripts.run_baseline import digest, write_json
from scripts.run_cadence_groups import score
from scripts.run_timing_experiment import training_rows, rounded
from scripts.train_xgboost import fit_schema, matrix


def run():
    cfgpath = Path('config/repeat-history-v1.json')
    cfg = json.loads(cfgpath.read_text())
    protocol = json.loads(Path(cfg['source_protocol']).read_text())
    params = json.loads(Path(protocol['reference_protocol']).read_text())['params']
    root = Path('artifacts/repeat-history-v1')
    if root.exists():
        raise FileExistsError(root)
    sourcepath = Path('data/features-splits-full-v1/all-origins-audit.csv')
    baseline_root = Path('artifacts/cadence-groups-full-v1')
    baseline_report = json.loads((baseline_root/'report.json').read_text())
    assert digest(sourcepath) == baseline_report['provenance']['full_features_sha256']
    assert digest(baseline_root/'real-predictions.csv') == baseline_report['files']['real-predictions.csv']
    source = read_csv(sourcepath)
    by_id = {r['origin_event_id']: r for r in source}
    base = [r for r in read_csv(baseline_root/'real-predictions.csv') if r['candidate'] == 'global']
    root.mkdir()
    write_json(root/'config-frozen.json', cfg)
    predictions = [{**r, 'candidate': 'train_all', 'past_product_purchases': int(by_id[r['origin_event_id']]['episode_purchase_count'])} for r in base]
    # Simple within-pair baseline uses observed past gaps, not future labels.
    for r in base:
        s = by_id[r['origin_event_id']]
        predictions.append({**r, 'candidate': 'personal_gap_median',
                            'past_product_purchases': int(s['episode_purchase_count']),
                            'predicted_gap_days': rounded(s['gap_median_days']) if int(s['gap_count']) else int(r['predicted_gap_days']),
                            'cart_quantity': rounded(s['qty_median_last3'])})
    counts = {}
    for fold in protocol['folds']:
        pool = training_rows(source, fold['train_end'], protocol['training_window_days'], cfg['fixed_horizon_days'])
        evaluation = [r for r in base if r['fold'] == fold['id']]
        valid = [by_id[r['origin_event_id']] for r in evaluation]
        counts[fold['id']] = {'train_all': len(pool)}
        for minimum in cfg['training_minimum_product_purchases']:
            name = f'train_min{minimum}'
            train = [r for r in pool if int(r['episode_purchase_count']) >= minimum]
            assert train and max(r['target_on'] for r in train) < fold['origin_start']
            assert not {r['origin_event_id'] for r in train} & {r['origin_event_id'] for r in valid}
            counts[fold['id']][name] = len(train)
            work = root/'models'/fold['id']/name
            work.mkdir(parents=True)
            schema = fit_schema(train)
            write_json(work/'schema.json', schema)
            values = {}
            for target in ['gap_days', 'quantity']:
                model = xgb.train({**params, 'objective': protocol['specialist_objective'], 'max_depth': protocol['specialist_depth']},
                                  matrix(train, schema, 'target_'+target), protocol['specialist_rounds'])
                model.save_model(work/(target+'.json'))
                saved = xgb.Booster({'nthread': 4})
                saved.load_model(work/(target+'.json'))
                values[target] = [rounded(v) for v in saved.predict(matrix(valid, schema))]
            for i, r in enumerate(evaluation):
                predictions.append({**r, 'candidate': name, 'past_product_purchases': int(valid[i]['episode_purchase_count']),
                                    'predicted_gap_days': values['gap_days'][i], 'cart_quantity': values['quantity'][i]})
        print(json.dumps({'fold': fold['id'], 'training': counts[fold['id']]}), flush=True)
    write_csv(root/'predictions.csv', list(predictions[0]), predictions)
    candidates = {}
    for name in ['train_all', 'train_min2', 'train_min4', 'personal_gap_median']:
        rs = [r for r in predictions if r['candidate'] == name]
        assert {r['origin_event_id'] for r in rs} == {r['origin_event_id'] for r in base}
        candidates[name] = {f'evaluate_min{minimum}': score([r for r in rs if r['past_product_purchases'] >= minimum])
                            for minimum in cfg['evaluation_minimum_product_purchases']}
    report = {'version': cfg['version'], 'config': cfg, 'training_counts': counts, 'candidates': candidates,
              'operational_approval': False, 'independent_final_test': False,
              'limitations': ['Retrospective development only; historical state availability not verified',
                              'Target metrics conditional on repurchase within90days; nonreturn counted separately',
                              'Higher eligibility score does not establish improvement from training exclusions',
                              'Training minimum applies to same customer-product purchases known at origin, never lifetime future count',
                              'Models trained on repeat histories are also diagnostically scored on cold histories; this does not authorize cold-history use'],
              'provenance': {'code_sha256': digest(__file__), 'config_sha256': digest(cfgpath),
                             'protocol_sha256': digest(cfg['source_protocol']), 'source_sha256': digest(sourcepath),
                             'baseline_report_sha256': digest(baseline_root/'report.json')},
              'files': {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file()}}
    write_json(root/'report.json', report)
    print(json.dumps({name: {cohort: {k: m.get(k) for k in ['origins', 'n', 'date_mae', 'date_within_7_rate', 'quantity_within_2_rate']} for cohort, m in scores.items()} for name, scores in candidates.items()}, indent=2))


if __name__ == '__main__':
    run()
