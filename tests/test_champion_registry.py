import json
import tempfile
import unittest
from pathlib import Path

from scripts.predict_replenishment import champion_path
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


if __name__=='__main__':
    unittest.main()
