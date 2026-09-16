"""Local TabPFN v2 versus same-context XGBoost on repeated product histories."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

from scripts.build_features_and_splits import read_csv
from scripts.build_purchase_events import write_csv
from scripts.run_baseline import digest, write_json
from scripts.run_cadence_groups import score
from scripts.run_tabpfn_experiment import feature_array, sample_context
from scripts.run_timing_experiment import rounded, training_rows
from scripts.train_xgboost import fit_schema, matrix


def run():
    config_path = Path('config/tabpfn-repeat-v1.json')
    cfg = json.loads(config_path.read_text())
    protocol = json.loads(Path(cfg['source_protocol']).read_text())
    params = json.loads(Path(protocol['reference_protocol']).read_text())['params']
    root = Path('artifacts/tabpfn-repeat-v1')
    if root.exists():
        raise FileExistsError(root)
    weights = Path(cfg['weights']).resolve()
    previous = json.loads(Path('artifacts/tabpfn-experiment-v1/report.json').read_text())
    assert digest(weights) == previous['provenance']['weights_sha256']
    for key, value in {'HF_HUB_OFFLINE': '1', 'SKB_DATA_DIRECTORY': 'artifacts/cache/skrub',
                       'TABPFN_MODEL_CACHE_DIR': str(weights.parent), 'MPLCONFIGDIR': 'artifacts/cache/matplotlib',
                       'XDG_CACHE_HOME': 'artifacts/cache'}.items():
        os.environ[key] = value if key == 'HF_HUB_OFFLINE' else str(Path(value).resolve())
    source_path = Path('data/features-splits-full-v1/all-origins-audit.csv')
    baseline_root = Path('artifacts/cadence-groups-full-v1')
    baseline_report = json.loads((baseline_root/'report.json').read_text())
    assert digest(source_path) == baseline_report['provenance']['full_features_sha256']
    assert digest(baseline_root/'real-predictions.csv') == baseline_report['files']['real-predictions.csv']
    source = read_csv(source_path)
    by_id = {r['origin_event_id']: r for r in source}
    base = [r for r in read_csv(baseline_root/'real-predictions.csv') if r['candidate'] == 'global'
            and int(by_id[r['origin_event_id']]['episode_purchase_count']) >= cfg['minimum_product_purchases']]
    root.mkdir()
    write_json(root/'config-frozen.json', cfg)
    records, folds = [], {}
    for fold in protocol['folds']:
        pool = [r for r in training_rows(source, fold['train_end'], protocol['training_window_days'], protocol['horizon_days'])
                if int(r['episode_purchase_count']) >= cfg['minimum_product_purchases']]
        train = sample_context(pool, cfg['max_training_rows'])
        evaluation = [r for r in base if r['fold'] == fold['id']]
        valid = [by_id[r['origin_event_id']] for r in evaluation]
        assert max(r['target_on'] for r in train) < fold['origin_start']
        assert not {r['origin_event_id'] for r in train} & {r['origin_event_id'] for r in valid}
        schema = fit_schema(train)
        work = root/fold['id']
        work.mkdir()
        write_json(work/'schema.json', schema)
        write_json(work/'context-origin-ids.json', [r['origin_event_id'] for r in train])
        values = {'tabpfn_v2_1000': {}, 'xgboost_matched_1000': {}}
        for target in ['gap_days', 'quantity']:
            inputs = work/(target+'-input.npz')
            output = work/(target+'-tabpfn.npy')
            np.savez(inputs, x_train=feature_array(train, schema), x_valid=feature_array(valid, schema),
                     y_train=np.asarray([float(r['target_'+target]) for r in train]))
            subprocess.run([sys.executable, '-X', 'faulthandler', '-m', 'scripts.tabpfn_worker', '--input', str(inputs),
                            '--weights', str(weights), '--config', str(config_path), '--output', str(output)], check=True)
            values['tabpfn_v2_1000'][target] = [rounded(v) for v in np.load(output, allow_pickle=False)]
            booster = xgb.train({**params, 'objective': 'reg:absoluteerror', 'max_depth': 4}, matrix(train, schema, 'target_'+target), 200)
            booster.save_model(work/(target+'-xgboost.json'))
            saved = xgb.Booster({'nthread': 4})
            saved.load_model(work/(target+'-xgboost.json'))
            values['xgboost_matched_1000'][target] = [rounded(v) for v in saved.predict(matrix(valid, schema))]
        for method, predictions in values.items():
            for i, r in enumerate(evaluation):
                records.append({**r, 'candidate': method, 'past_product_purchases': int(valid[i]['episode_purchase_count']),
                                'predicted_gap_days': predictions['gap_days'][i], 'cart_quantity': predictions['quantity'][i]})
        folds[fold['id']] = {'eligible_training_rows': len(pool), 'context_rows': len(train), 'evaluation_rows': len(valid)}
        print(json.dumps({'completed_fold': fold['id'], **folds[fold['id']]}), flush=True)
    write_csv(root/'predictions.csv', list(records[0]), records)
    candidates = {method: {f'evaluate_min{minimum}': score([r for r in records if r['candidate'] == method and r['past_product_purchases'] >= minimum])
                           for minimum in [2, 4, 6]} for method in ['tabpfn_v2_1000', 'xgboost_matched_1000']}
    report = {'version': cfg['version'], 'config': cfg, 'folds': folds, 'candidates': candidates,
              'operational_approval': False, 'independent_final_test': False,
              'limitations': ['TabPFN v2, 1000-context examples, two estimators: bounded local benchmark, not maximum performance',
                              'Same customer-product at least two purchases before prediction; cohorts are reused development data',
                              'Date/quantity metrics conditional on repurchase within90days; no-repurchase separately reported',
                              'Historical state availability not verified'],
              'provenance': {'code_sha256': digest(__file__), 'worker_sha256': digest('scripts/tabpfn_worker.py'),
                             'config_sha256': digest(config_path), 'protocol_sha256': digest(cfg['source_protocol']),
                             'weights_sha256': digest(weights), 'source_sha256': digest(source_path),
                             'baseline_report_sha256': digest(baseline_root/'report.json')},
              'files': {str(p.relative_to(root)): digest(p) for p in root.rglob('*') if p.is_file()}}
    write_json(root/'report.json', report)
    print(json.dumps(candidates, indent=2))


if __name__ == '__main__':
    run()
