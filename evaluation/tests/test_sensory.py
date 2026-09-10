from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation.sensory import enrich, main
from evaluation.harness import execute, Mode
from evaluation.media import prepare_media
from evaluation.tests.test_harness import FakeAdapters


class SensoryEvaluationTests(unittest.TestCase):
    def fixture(self):
        from PIL import Image
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'input.png'
            Image.new('RGB', (32, 24), 'white').save(path)
            with prepare_media(path) as frames:
                results = [execute(frames, mode, FakeAdapters(), run_id='s')[0] for mode in Mode]
        return dict(schema_version='evaluation-0.1', run_id='s', results=results,
                    comparisons=[{'sentinel': 'retain original comparisons'}])

    def test_real_harness_roundtrip_and_mode_separation(self):
        source = self.fixture()
        before = deepcopy(source)
        enriched = enrich(source)
        self.assertEqual(source, before)
        self.assertEqual(json.loads(json.dumps(enriched)), enriched)
        for old, new in zip(source['results'], enriched['results']):
            self.assertEqual({key: new[key] for key in old}, old)
            if old['mode'] == 'VLM_ONLY':
                self.assertNotIn('sensory_world_model', new)
            else:
                sensory = new['sensory_world_model']['evidence']['items']
                self.assertEqual({x['modality'] for x in sensory if x['source_type'] == 'sensory'},
                                 {'auditory', 'tactile', 'thermal', 'kinesthetic'})
        self.assertEqual(source['comparisons'], enriched['comparisons'])

    def test_repeat_is_deterministic(self):
        source = self.fixture()
        self.assertEqual(enrich(source), enrich(source))

    def test_rejected_frames_not_enriched(self):
        source = self.fixture()
        for item in source['results']:
            item['status'] = 'rejected_input'
        self.assertTrue(all('sensory_world_model' not in item for item in enrich(source)['results']))

    def test_invalid_and_double_enrichment_rejected(self):
        with self.assertRaises(ValueError):
            enrich({})
        with self.assertRaises(ValueError):
            enrich(enrich(self.fixture()))

    def test_cli_new_file_and_overwrite_protection(self):
        source = self.fixture()
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'raw.json'
            path.write_text(json.dumps(source), encoding='utf-8')
            output = Path(temporary) / 'new_sensory.json'
            with patch('sys.argv', ['sensory', str(path), '--output', str(output)]):
                main()
                first = output.read_bytes()
                with patch('sys.stderr'), self.assertRaises(SystemExit):
                    main()
                self.assertEqual(first, output.read_bytes())
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), source)
            with patch('sys.argv', ['sensory', str(path), '--output', str(Path(temporary) / 'results_case_a_combined.json')]):
                with patch('sys.stderr'), self.assertRaises(SystemExit):
                    main()
