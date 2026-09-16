"""Reload saved predictions and independently aggregate fixed cadence experiments."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.build_features_and_splits import read_csv
from scripts.run_baseline import digest, write_json
from scripts.train_xgboost import matrix


def run():
    source = {r['origin_event_id']: r for r in read_csv('data/features-splits-full-v1/all-origins-audit.csv')}
    roots = [Path('artifacts/cadence-groups-v1'), Path('artifacts/cadence-groups-full-v1')]
    verified, frames = {}, {}
    for root in roots:
        report = json.loads((root/'report.json').read_text())
        cfg = report['config']
        for filename, sha in report['files'].items():
            assert digest(root/filename) == sha
        records = read_csv(root/'real-predictions.csv')
        assert len({(r['candidate'], r['origin_event_id']) for r in records}) == len(records)
        for r in records:
            s = source[r['origin_event_id']]
            if r['outcome'] == 'observed':
                assert int(r['target_gap_days']) == int(s['target_gap_days']) <= 90
                assert r['target_quantity'] == s['target_quantity']
                assert (pd.Timestamp(s['target_on'])-pd.Timestamp(s['origin_on'])).days == int(r['target_gap_days'])
            assert pd.Timestamp(r['origin_on'])+pd.Timedelta(days=90) <= pd.Timestamp('2026-09-14')
        for fold in cfg['folds']:
            for method in ['global', 'group_specialists']:
                batches = {}
                for r in records:
                    if r['fold'] != fold['id'] or r['candidate'] != method:
                        continue
                    group = r['cadence_group']
                    prefix = group if method == 'group_specialists' and group in report['training_counts'][fold['id']]['specialists_trained'] else 'global'
                    batches.setdefault(prefix, []).append(r)
                for prefix, rs in batches.items():
                    work = root/'models'/fold['id']
                    schema = json.loads((work/(prefix+'-schema.json')).read_text())
                    dm = matrix([source[r['origin_event_id']] for r in rs], schema)
                    for target, col in [('gap_days', 'predicted_gap_days'), ('quantity', 'cart_quantity')]:
                        model = xgb.Booster({'nthread': 4})
                        model.load_model(work/(prefix+'-'+target+'.json'))
                        predictions = np.floor(np.maximum(model.predict(dm).astype(float), 1)+.5).astype(int)
                        np.testing.assert_array_equal(predictions, [int(r[col]) for r in rs])
        df = pd.DataFrame(records)
        obs = df[df.outcome == 'observed'].copy()
        for col in ['predicted_gap_days', 'target_gap_days', 'cart_quantity', 'target_quantity']:
            obs[col] = obs[col].astype(int)
        obs['date_error'] = (obs.predicted_gap_days-obs.target_gap_days).abs()
        obs['quantity_error'] = (obs.cart_quantity-obs.target_quantity).abs()
        for method, group in obs.groupby('candidate'):
            saved = report['real'][method]['overall']
            checks = {'date_mae': group.date_error.mean(),
                      'customer_macro_date_mae': group.groupby('customer_id').date_error.mean().mean(),
                      'date_within_7_rate': group.date_error.le(7).mean(),
                      'date_within_14_rate': group.date_error.le(14).mean(),
                      'quantity_within_2_rate': group.quantity_error.le(2).mean(),
                      'joint_date_7_quantity_2_rate': (group.date_error.le(7)&group.quantity_error.le(2)).mean()}
            for key, value in checks.items():
                assert abs(saved[key]-value) < 1e-10, (key, value, saved[key])
        frames[root.name] = obs
        verified[root.name] = {'real_prediction_rows': len(records), 'unique_origins': df.origin_event_id.nunique(),
                               'observed_per_method': len(obs)//2, 'model_reload_exact': True,
                               'independent_pandas_metrics_match': True, 'report_sha256': digest(root/'report.json')}
    a, b = [frames[root.name] for root in roots]
    identity = ['candidate', 'origin_event_id', 'customer_id', 'target_gap_days', 'target_quantity', 'cadence_group']
    pd.testing.assert_frame_equal(a[identity].reset_index(drop=True), b[identity].reset_index(drop=True))
    diffs = {}
    by_model = {}
    for name, df in frames.items():
        for method, g in df.groupby('candidate'):
            by_model[name+'/'+method] = g.groupby('customer_id').date_error.mean()
    baseline = by_model['cadence-groups-v1/global']
    for name, series in by_model.items():
        d = (series-baseline).to_numpy()
        rng = np.random.default_rng(42)
        draws = [rng.choice(d, len(d), replace=True).mean() for _ in range(2000)]
        diffs[name] = {'customer_macro_mae_delta_vs_270day_global': float(d.mean()),
                       'descriptive_customer_bootstrap_95pct': np.quantile(draws, [.025, .975]).tolist()}
    output = {'passed': True, 'runs': verified, 'same_real_observed_cohort': True, 'paired_differences': diffs,
              'bootstrap_replicates': 2000, 'bootstrap_seed': 42, 'independent_final_test': False,
              'code_sha256': digest(__file__)}
    write_json(roots[0]/'combined-verification.json', output)
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    run()
