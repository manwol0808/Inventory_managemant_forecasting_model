"""Past-only customer cadence specialists; independent-rule synthetic stress test.

Synthetic examples do not train models or estimate deployment accuracy. Existing
short-history model is tested separately from the mature 90-day real comparison.
"""
import json
import argparse
import math
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from statistics import median, mean, pstdev

import numpy as np
import xgboost as xgb

from scripts.build_features_and_splits import FEATURE_COLUMNS, feature_rows, read_csv
from scripts.build_purchase_events import write_csv
from scripts.predict_replenishment import PreviewPredictor
from scripts.run_baseline import digest, write_json
from scripts.run_timing_experiment import training_rows, summarize, rounded, horizon_outcome
from scripts.train_xgboost import fit_schema, matrix, postprocess


def classify(gaps, cfg):
    if len(gaps) < cfg['minimum_past_gaps']:
        return 'insufficient_history'
    recent = gaps[-cfg['recent_gap_count']:]
    if pstdev(recent) / mean(recent) > cfg['irregular_cv_above']:
        return 'irregular'
    typical = median(recent)
    for upper in cfg['cadence_upper_bounds_days']:
        if typical <= upper:
            return f'within_{upper}d'
    return 'over_60d'


def groups_from_events(events, cfg):
    """One group per customer/day; individual product features remain separate.

    Multiple products purchased on one day count as one customer purchase day.
    A held item conservatively resets the customer's cadence history before any
    completed items of that day enter. Future days cannot change earlier groups.
    """
    histories = defaultdict(list)
    batches = defaultdict(list)
    for e in events:
        if e['event_state'] in {'completed', 'held'}:
            batches[(e['ordered_on'], e['customer_id'])].append(e)
    groups = {}
    for (day, customer), batch in sorted(batches.items()):
        if any(e['event_state'] == 'held' for e in batch):
            histories[customer] = []
        completed = [e for e in batch if e['event_state'] == 'completed']
        if completed:
            h = histories[customer]
            h.append(date.fromisoformat(day))
            gaps = [(b-a).days for a, b in zip(h, h[1:])]
            if any(g <= 0 for g in gaps):
                raise ValueError('Nonpositive historical gap')
            for e in completed:
                groups[e['event_id']] = classify(gaps, cfg)
    return groups


def score(records, horizon=90):
    result = summarize(records, horizon)
    observed = [r for r in records if r['outcome'] == 'observed']
    if observed:
        errors = sorted(abs(int(r['predicted_gap_days'])-int(r['target_gap_days'])) for r in observed)
        result['date_within_14_rate'] = sum(e <= 14 for e in errors)/len(errors)
        result['date_80pct_tolerance_days'] = errors[math.ceil(.8*len(errors))-1]
        result['joint_date_7_quantity_2_rate'] = sum(
            abs(int(r['predicted_gap_days'])-int(r['target_gap_days'])) <= 7
            and abs(int(r['cart_quantity'])-int(r['target_quantity'])) <= 2 for r in observed)/len(observed)
    return result


def synthetic_data(cfg, product, secondary_product='secondary_synthetic_product'):
    """Generate toy ledgers with targets sampled independently of every predictor.

    Most shops buy a single known catalog product; one scenario buys a second
    product weekly while its target item is monthly. Past observations are bounded
    to the original short model's history start. Last purchase is Aug 15. Future
    outcomes never enter the ledger passed to the production feature builder.
    """
    rng = np.random.default_rng(cfg['synthetic_seed'])
    anchor, start = date(2026, 8, 15), date(2026, 4, 9)
    events, labels, targets = [], [], {}
    fixed = {'weekly': 7, 'fortnightly': 14, 'monthly': 30, 'every45days': 45, 'every75days': 75,
             'weekly_shop_monthly_item': 30}
    for scenario in cfg['synthetic_scenarios']:
        for index in range(cfg['synthetic_customers_per_scenario']):
            customer = f'synthetic_{scenario}_{index}'
            base_qty = int(rng.integers(1, 9))
            base_gap = fixed.get(scenario, 14)
            def draw_gap():
                if scenario == 'irregular':
                    return int(rng.choice([4, 10, 25, 50, 75]))
                return max(1, base_gap+int(rng.integers(-2, 3)))
            days = [anchor]
            while scenario != 'first_purchase':
                previous = days[-1]-timedelta(days=draw_gap())
                if previous < start:
                    break
                days.append(previous)
            for j, day in enumerate(sorted(days)):
                event_id = f'{customer}_{j}'
                quantity = max(1, base_qty+int(rng.integers(-1, 2)))
                e = dict(event_id=event_id, customer_id=customer, product_id=product,
                         ordered_on=day.isoformat(), quantity=str(quantity), event_state='completed',
                         snapshot_observed_at='2026-09-16T00:00:00+09:00')
                events.append(e)
                labels.append(dict(origin_event_id=event_id, customer_id=customer, product_id=product,
                                   origin_on=day.isoformat(), label_state='right_censored', target_on='',
                                   target_quantity='', target_gap_days='', label_snapshot_observed_at=e['snapshot_observed_at']))
            next_gap = int(rng.integers(4, 8)) if scenario == 'sudden_speedup' else draw_gap()
            next_quantity = max(1, base_qty+int(rng.integers(-1, 2)))
            if scenario == 'quantity_spike':
                next_quantity = base_qty*4
            if scenario == 'first_purchase':
                next_gap = int(rng.choice([7, 14, 30, 45, 75]))
            targets[event_id] = dict(scenario=scenario, target_gap_days='' if scenario == 'no_return_60d' else next_gap,
                                     target_quantity='' if scenario == 'no_return_60d' else next_quantity,
                                     outcome='no_repurchase_within_horizon' if scenario == 'no_return_60d' else 'observed')
            if scenario == 'weekly_shop_monthly_item':
                day = anchor
                j = 0
                while day >= start:
                    eid = f'{customer}_secondary_{j}'
                    events.append(dict(event_id=eid, customer_id=customer, product_id=secondary_product,
                                       ordered_on=day.isoformat(), quantity='3', event_state='completed',
                                       snapshot_observed_at='2026-09-16T00:00:00+09:00'))
                    labels.append(dict(origin_event_id=eid, customer_id=customer, product_id=secondary_product,
                                       origin_on=day.isoformat(), label_state='right_censored', target_on='',
                                       target_quantity='', target_gap_days='', label_snapshot_observed_at='2026-09-16T00:00:00+09:00'))
                    day -= timedelta(days=7)
                    j += 1
    split_cfg = json.loads(Path('data/features-splits-short-v2/report.json').read_text())['split_config']
    features = [r for r in feature_rows(events, labels, split_cfg) if r['origin_event_id'] in targets]
    return features, targets, events


def run(cfg_path=Path('config/cadence-groups-v1.json'), output=Path('artifacts/cadence-groups-v1')):
    cfg_path, output = Path(cfg_path), Path(output)
    cfg = json.loads(cfg_path.read_text())
    reference = json.loads(Path(cfg['reference_protocol']).read_text())
    if output.exists():
        raise FileExistsError(output)
    source, event_dir = Path('data/features-splits-full-v1'), Path('data/purchase-events-full-v1')
    for root, names in [(source, ['all-origins-audit.csv']), (event_dir, ['daily-events.csv', 'next-purchase-labels.csv'])]:
        manifest = json.loads((root/'report.json').read_text())
        for name in names:
            assert digest(root/name) == manifest['files'][name]
    rows = read_csv(source/'all-origins-audit.csv')
    events_by_id = {r['event_id']: r for r in read_csv(event_dir/'daily-events.csv')}
    labels = {r['origin_event_id']: r for r in read_csv(event_dir/'next-purchase-labels.csv')}
    group_map = groups_from_events(list(events_by_id.values()), cfg)
    output.mkdir()
    write_json(output/'config-frozen.json', cfg)
    comparisons, counts = [], {}
    for fold in cfg['folds']:
        train = training_rows(rows, fold['train_end'], cfg['training_window_days'], cfg['horizon_days'])
        valid = [r for r in rows if fold['origin_start'] <= r['origin_on'] <= fold['origin_end']]
        assert not {r['origin_event_id'] for r in train} & {r['origin_event_id'] for r in valid}
        assert max(r['target_on'] for r in train) < fold['origin_start']
        outcomes = {r['origin_event_id']: horizon_outcome(r, cfg['horizon_days'], '2026-09-14', labels, events_by_id) for r in valid}
        assert 'not_mature' not in outcomes.values()
        work = output/'models'/fold['id']
        work.mkdir(parents=True)
        params = {**reference['params'], 'objective': cfg['specialist_objective'], 'max_depth': cfg['specialist_depth']}
        schema = fit_schema(train)
        write_json(work/'global-schema.json', schema)
        global_values, specialist_values = {}, {}
        for target in ['gap_days', 'quantity']:
            model = xgb.train(params, matrix(train, schema, 'target_'+target), cfg['specialist_rounds'])
            model.save_model(work/('global-'+target+'.json'))
            reloaded = xgb.Booster({'nthread': 4})
            reloaded.load_model(work/('global-'+target+'.json'))
            global_values[target] = [rounded(x) for x in reloaded.predict(matrix(valid, schema))]
            specialist_values[target] = global_values[target].copy()
        train_counts = Counter(group_map[r['origin_event_id']] for r in train)
        trained = []
        for group, n in sorted(train_counts.items()):
            if group == 'insufficient_history' or n < cfg['minimum_specialist_training_rows']:
                continue
            subset = [r for r in train if group_map[r['origin_event_id']] == group]
            indices = [i for i, r in enumerate(valid) if group_map[r['origin_event_id']] == group]
            if not indices:
                continue
            group_schema = fit_schema(subset)
            write_json(work/(group+'-schema.json'), group_schema)
            for target in ['gap_days', 'quantity']:
                model = xgb.train(params, matrix(subset, group_schema, 'target_'+target), cfg['specialist_rounds'])
                model.save_model(work/(group+'-'+target+'.json'))
                reloaded = xgb.Booster({'nthread': 4})
                reloaded.load_model(work/(group+'-'+target+'.json'))
                values = reloaded.predict(matrix([valid[i] for i in indices], group_schema))
                for i, value in zip(indices, values):
                    specialist_values[target][i] = rounded(value)
            trained.append(group)
        counts[fold['id']] = {'training': dict(train_counts), 'specialists_trained': trained,
                              'train_origin_min': min(r['origin_on'] for r in train),
                              'train_origin_max': max(r['origin_on'] for r in train),
                              'train_target_max': max(r['target_on'] for r in train)}
        for i, r in enumerate(valid):
            observed = outcomes[r['origin_event_id']] == 'observed'
            base = {k: r[k] for k in ['origin_event_id', 'customer_id', 'product_id', 'origin_on']}
            base.update(fold=fold['id'], outcome=outcomes[r['origin_event_id']],
                        target_gap_days=r['target_gap_days'] if observed else '',
                        target_quantity=r['target_quantity'] if observed else '')
            for method, values in [('global', global_values), ('group_specialists', specialist_values)]:
                comparisons.append({**base, 'candidate': method, 'cadence_group': group_map[r['origin_event_id']],
                                    'predicted_gap_days': values['gap_days'][i], 'cart_quantity': values['quantity'][i]})
        print(json.dumps({'fold': fold['id'], **counts[fold['id']]}), flush=True)
    write_csv(output/'real-predictions.csv', list(comparisons[0]), comparisons)
    real = {}
    for method in ['global', 'group_specialists']:
        selected = [r for r in comparisons if r['candidate'] == method]
        real[method] = {'overall': score(selected),
                        'folds': {f['id']: score([r for r in selected if r['fold'] == f['id']]) for f in cfg['folds']},
                        'groups': {g: score([r for r in selected if r['cadence_group'] == g]) for g in sorted(set(group_map.values()))}}
    # The user-requested saved short-history candidate, not the real-fold models.
    short = PreviewPredictor('artifacts/xgboost-short-v2')
    short_train = read_csv('data/features-splits-short-v2/train_pool.csv')
    manifest = json.loads(Path('data/features-splits-short-v2/report.json').read_text())
    assert digest('data/features-splits-short-v2/train_pool.csv') == manifest['files']['train_pool.csv']
    product, secondary_product = [p for p, n in Counter(r['product_id'] for r in short_train).most_common(2)]
    features, targets, events = synthetic_data(cfg, product, secondary_product)
    groups = groups_from_events(events, cfg)
    dm = matrix([{k: r[k] for k in FEATURE_COLUMNS} for r in features], short.schema)
    _, qty, gaps = postprocess(short.models['quantity'].predict(dm), short.models['gap'].predict(dm))
    synthetic = []
    for i, r in enumerate(features):
        t = targets[r['origin_event_id']]
        synthetic.append({**{k: r[k] for k in ['origin_event_id', 'customer_id', 'product_id', 'origin_on']},
                          **t, 'cadence_group': groups[r['origin_event_id']], 'predicted_gap_days': int(gaps[i]),
                          'cart_quantity': int(qty[i])})
    write_csv(output/'synthetic-features.csv', list(features[0]), features)
    write_csv(output/'synthetic-predictions.csv', list(synthetic[0]), synthetic)
    synthetic_scores = {s: score([r for r in synthetic if r['scenario'] == s], 60) for s in cfg['synthetic_scenarios']}
    for s, metrics in synthetic_scores.items():
        rs = [r for r in synthetic if r['scenario'] == s]
        metrics['predicted_gap_median'] = median(r['predicted_gap_days'] for r in rs)
        metrics['actual_gap_median'] = median(r['target_gap_days'] for r in rs) if s != 'no_return_60d' else None
        metrics['groups'] = dict(Counter(r['cadence_group'] for r in rs))
        if s == 'no_return_60d':
            metrics['no_repurchase_but_predicted_within_horizon_rate'] = sum(r['predicted_gap_days'] <= 60 for r in rs)/len(rs)
    result = {'version': cfg['version'], 'config': cfg, 'real': real, 'training_counts': counts,
              'synthetic': synthetic_scores, 'synthetic_model': short.model_version,
              'synthetic_known_product_id': product, 'synthetic_rows': len(synthetic),
              'operational_approval': False, 'independent_final_test': False,
              'limitations': ['Real metrics conditional on observed repurchase within 90 days; no-return reported separately',
                              'Synthetic rules are designer assumptions, not independent evidence of customer accuracy',
                              'Synthetic observed targets include 75-day gaps; synthetic no-return means no event within 60 days',
                              'Synthetic mostly single-product shops and product aggregates differ from real distribution',
                              'Latest-state historical availability unverified; reused development cohorts, no fresh test'],
              'provenance': {'config_sha256': digest(cfg_path), 'code_sha256': digest(__file__),
                             'reference_config_sha256': digest(cfg['reference_protocol']),
                             'full_features_sha256': digest(source/'all-origins-audit.csv'),
                             'events_sha256': digest(event_dir/'daily-events.csv'),
                             'labels_sha256': digest(event_dir/'next-purchase-labels.csv'),
                             'short_model_report_sha256': digest(short.root/'report.json'),
                             'short_training_sha256': digest('data/features-splits-short-v2/train_pool.csv')},
              'files': {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}}
    write_json(output/'report.json', result)
    print(json.dumps({'real': {k: v['overall'] for k, v in real.items()}, 'synthetic': synthetic_scores}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/cadence-groups-v1.json'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/cadence-groups-v1'))
    args = parser.parse_args()
    run(args.config, args.output)
