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
