import json
import tempfile
import unittest
from pathlib import Path

from scripts.predict_replenishment import base_champion_path, champion_path
from scripts.run_baseline import digest


class ChampionRegistryTests(unittest.TestCase):
    def test_rejects_changed_selected_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'report.json').write_text('{}')
            (root/'gap-model.json').write_text('frozen-model')
            manifest={'version':'champion-v1','role':'champion','model_dir':str(root),
                      'report_sha256':digest(root/'report.json'),
                      'model_files':{'gap-model.json':digest(root/'gap-model.json')}}
            registry=root/'champion.json'
            registry.write_text(json.dumps(manifest))
            self.assertEqual(champion_path(registry),root)
            (root/'gap-model.json').write_text('different-model')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                champion_path(registry)

    def test_router_champion_verifies_bundle_and_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def entry(folder, version, **extra):
                folder.mkdir()
                (folder/'report.json').write_text('{}')
                (folder/'model.json').write_text(folder.name)
                return {'version':version,'role':'champion','model_dir':str(folder),'report_sha256':digest(folder/'report.json'),
                        'model_files':{'model.json':digest(folder/'model.json')},**extra}
            base=entry(root/'base','champion-v1')
            registry=root/'champion.json'
            registry.write_text(json.dumps(entry(root/'router','champion-v2',kind='router',base_model=base)))
            self.assertEqual(champion_path(registry),root/'router')
            self.assertEqual(base_champion_path(registry),root/'base')
            (root/'base'/'model.json').write_text('changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                champion_path(registry)

    def test_single_model_champion_is_its_own_base(self):
        registry=Path('config/champion.json')
        self.assertTrue(base_champion_path(registry).is_dir())


if __name__=='__main__':
    unittest.main()


class NoRouterTests(unittest.TestCase):
    def test_no_router_sends_every_group_to_the_base_model(self):
        from scripts.router_champion import REGISTRY, RouterChampion
        if not (REGISTRY.exists() and Path(json.loads(REGISTRY.read_text())["model_dir"]).exists()):
            self.skipTest("champion artifacts are not in this checkout")
        routes = RouterChampion(router=False).cfg["routes"]
        self.assertTrue(routes and set(routes.values()) == {"champion"})

    def test_ab_group_is_stable_and_holdout_is_about_five_percent(self):
        from scripts.router_champion import ab_group
        ids = [f"store-{i}" for i in range(20000)]
        groups = [ab_group(i) for i in ids]
        self.assertEqual(groups, [ab_group(i) for i in ids])
        share = groups.count("holdout") / len(groups)
        self.assertTrue(0.04 < share < 0.06, share)
        self.assertEqual(set(groups), {"holdout", "control", "treatment"})


class DailyExtractSqlTests(unittest.TestCase):
    def test_app_orders_table_is_substituted_and_both_windows_move(self):
        from delivery.run_daily import DEFAULT_APP_ORDERS_TABLE, extract_sql
        web = extract_sql("2026-10-07")
        self.assertNotIn("app_orders", web)
        self.assertIn("BETWEEN DATE '2024-10-02' AND DATE '2026-10-07'", web)
        both = extract_sql("2026-10-07", "proj.ds.app_orders")
        self.assertIn("`proj.ds.app_orders`", both)
        self.assertNotIn(f"`{DEFAULT_APP_ORDERS_TABLE}`", both)
        self.assertEqual(both.count("BETWEEN DATE '2024-10-02' AND DATE '2026-10-07'"), 2)
        self.assertIn("UNION ALL", both)
