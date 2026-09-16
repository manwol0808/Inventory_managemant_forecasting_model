"""Verify repeated-history contexts, model outputs and conditional metrics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.build_features_and_splits import read_csv
from scripts.run_baseline import digest, write_json
from scripts.run_tabpfn_experiment import feature_array, sample_context
from scripts.run_timing_experiment import training_rows
from scripts.train_xgboost import matrix


def run():
    source = read_csv('data/features-splits-full-v1/all-origins-audit.csv')
    by_id = {r['origin_event_id']: r for r in source}
    protocol = json.loads(Path('config/cadence-groups-full-v1.json').read_text())
    roots = [Path('artifacts/repeat-history-v1'), Path('artifacts/tabpfn-repeat-v1')]
    results = {}
    for root in roots:
        report = json.loads((root/'report.json').read_text())
        for name, sha in report['files'].items():
            assert digest(root/name) == sha
        rows = read_csv(root/'predictions.csv')
        df = pd.DataFrame(rows)
        assert not df.duplicated(['candidate', 'origin_event_id']).any()
        for r in rows:
            s = by_id[r['origin_event_id']]
            assert int(r['past_product_purchases']) == int(s['episode_purchase_count'])
            if r['outcome'] == 'observed':
                assert r['target_gap_days'] == s['target_gap_days'] and r['target_quantity'] == s['target_quantity']
            assert pd.Timestamp(r['origin_on'])+pd.Timedelta(days=90) <= pd.Timestamp('2026-09-14')
        for fold in protocol['folds']:
            if root.name == 'tabpfn-repeat-v1':
                work = root/fold['id']
                schema = json.loads((work/'schema.json').read_text())
                ids = json.loads((work/'context-origin-ids.json').read_text())
                pool = [r for r in training_rows(source, fold['train_end'], 10000, 90) if int(r['episode_purchase_count']) >= 2]
                assert ids == [r['origin_event_id'] for r in sample_context(pool, 1000)]
                train = [by_id[i] for i in ids]
                assert max(r['target_on'] for r in train) < fold['origin_start']
                for method in ['tabpfn_v2_1000', 'xgboost_matched_1000']:
                    rs = [r for r in rows if r['fold'] == fold['id'] and r['candidate'] == method]
                    valid = [by_id[r['origin_event_id']] for r in rs]
                    assert not set(ids) & {r['origin_event_id'] for r in rs}
                    for target, col in [('gap_days', 'predicted_gap_days'), ('quantity', 'cart_quantity')]:
                        arrays = np.load(work/(target+'-input.npz'), allow_pickle=False)
                        np.testing.assert_array_equal(arrays['x_train'], feature_array(train, schema))
                        np.testing.assert_array_equal(arrays['x_valid'], feature_array(valid, schema))
                        np.testing.assert_array_equal(arrays['y_train'], [float(r['target_'+target]) for r in train])
                        if method == 'tabpfn_v2_1000':
                            raw = np.load(work/(target+'-tabpfn.npy'), allow_pickle=False)
                        else:
                            model = xgb.Booster({'nthread': 4})
                            model.load_model(work/(target+'-xgboost.json'))
                            raw = model.predict(matrix(valid, schema))
                        np.testing.assert_array_equal(np.floor(np.maximum(raw.astype(float), 1)+.5).astype(int), [int(r[col]) for r in rs])
            else:
                for minimum in [2, 4]:
                    rs = [r for r in rows if r['fold'] == fold['id'] and r['candidate'] == f'train_min{minimum}']
                    work = root/'models'/fold['id']/f'train_min{minimum}'
                    schema = json.loads((work/'schema.json').read_text())
                    dm = matrix([by_id[r['origin_event_id']] for r in rs], schema)
                    for target, col in [('gap_days', 'predicted_gap_days'), ('quantity', 'cart_quantity')]:
                        model = xgb.Booster({'nthread': 4})
                        model.load_model(work/(target+'.json'))
                        raw = model.predict(dm).astype(float)
                        np.testing.assert_array_equal(np.floor(np.maximum(raw, 1)+.5).astype(int), [int(r[col]) for r in rs])
        df['past_product_purchases'] = df.past_product_purchases.astype(int)
        for method, cohorts in report['candidates'].items():
            for cohort, saved in cohorts.items():
                minimum = int(cohort.split('min')[1])
                eligible = df[(df.candidate == method)&df.past_product_purchases.ge(minimum)]
                observed = eligible[eligible.outcome == 'observed'].copy()
                de = (observed.predicted_gap_days.astype(int)-observed.target_gap_days.astype(int)).abs()
                qe = (observed.cart_quantity.astype(int)-observed.target_quantity.astype(int)).abs()
                assert len(eligible) == saved['origins'] and len(observed) == saved['n']
                checks = {'date_mae': de.mean(), 'date_within_7_rate': de.le(7).mean(),
                          'date_within_14_rate': de.le(14).mean(), 'quantity_within_2_rate': qe.le(2).mean(),
                          'customer_macro_date_mae': de.groupby(observed.customer_id).mean().mean()}
                for key, value in checks.items():
                    assert abs(saved[key]-value) < 1e-10
        results[root.name] = {'prediction_rows': len(rows), 'identities_and_targets_match': True,
                             'xgboost_reload_exact': True, 'independent_metrics_match': True,
                             'report_sha256': digest(root/'report.json')}
    result = {'passed': True, 'runs': results, 'tabpfn_context_and_raw_output_match': True,
              'tabpfn_forward_pass_replayed': False, 'code_sha256': digest(__file__)}
    write_json(roots[0]/'verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
