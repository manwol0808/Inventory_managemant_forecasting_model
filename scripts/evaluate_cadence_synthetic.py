"""Compare frozen models on the same artificial histories; never fit or tune."""
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from scripts.build_features_and_splits import read_csv
from scripts.build_purchase_events import write_csv
from scripts.run_baseline import digest, write_json
from scripts.run_cadence_groups import score
from scripts.train_xgboost import matrix


def run():
    root = Path('artifacts/cadence-groups-v1')
    destination = root/'synthetic-comparison.json'
    if destination.exists():
        raise FileExistsError(destination)
    report = json.loads((root/'report.json').read_text())
    for name, sha in report['files'].items():
        assert digest(root/name) == sha
    features = {r['origin_event_id']: r for r in read_csv(root/'synthetic-features.csv')}
    original = read_csv(root/'synthetic-predictions.csv')
    records = [{**r, 'model': 'short_original'} for r in original]
    fold = report['config']['folds'][-1]['id']
    work = root/'models'/fold
    for method in ['global_90d', 'customer_group_90d']:
        batches = {}
        for r in original:
            group = r['cadence_group']
            prefix = group if method == 'customer_group_90d' and group in report['training_counts'][fold]['specialists_trained'] else 'global'
            batches.setdefault(prefix, []).append(r)
        for prefix, rows in batches.items():
            schema = json.loads((work/(prefix+'-schema.json')).read_text())
            dm = matrix([features[r['origin_event_id']] for r in rows], schema)
            predictions = {}
            for target in ['gap_days', 'quantity']:
                model = xgb.Booster({'nthread': 4})
                model.load_model(work/(prefix+'-'+target+'.json'))
                predictions[target] = np.floor(np.maximum(model.predict(dm).astype(float), 1)+.5).astype(int)
            records.extend({**r, 'model': method, 'predicted_gap_days': int(predictions['gap_days'][i]),
                            'cart_quantity': int(predictions['quantity'][i])} for i, r in enumerate(rows))
    write_csv(root/'synthetic-comparison.csv', list(records[0]), records)
    metrics = {}
    for method in ['short_original', 'global_90d', 'customer_group_90d']:
        metrics[method] = {}
        for scenario in report['config']['synthetic_scenarios']:
            rs = [r for r in records if r['model'] == method and r['scenario'] == scenario]
            m = score(rs, 60)
            m['predicted_gap_median'] = float(np.median([int(r['predicted_gap_days']) for r in rs]))
            if scenario == 'no_return_60d':
                m['no_repurchase_but_predicted_within_horizon_rate'] = sum(int(r['predicted_gap_days']) <= 60 for r in rs)/len(rs)
            metrics[method][scenario] = m
    result = {'metrics': metrics, 'models_fit_through': {'short_original': '2026-07-13', 'global_90d': '2026-05-31', 'customer_group_90d': '2026-05-31'},
              'synthetic_origin_on': '2026-08-15', 'rows_per_model': len(original),
              'same_synthetic_inputs': True, 'fit_or_tuning': False, 'operational_approval': False,
              'limitations': ['Synthetic rules do not establish real customer accuracy',
                              'Original short model and new 90-day models have different training populations and history contracts',
                              'Synthetic history begins Apr 9, unlike full-history models; diagnostic distribution shift',
                              'Compare global_90d versus customer_group_90d to isolate routing; all synthetic product aggregates are artificial'],
              'source_report_sha256': digest(root/'report.json'), 'code_sha256': digest(__file__),
              'predictions_sha256': digest(root/'synthetic-comparison.csv')}
    write_json(destination, result)
    print(json.dumps({model: {s: {k: m.get(k) for k in ['date_within_7_rate', 'quantity_within_2_rate', 'predicted_gap_median']} for s, m in scenarios.items()} for model, scenarios in metrics.items()}, indent=2))


if __name__ == '__main__':
    run()
