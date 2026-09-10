from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fifth_layer.world_state import WorldState
from evaluation.adapters import RepositoryAdapters
from evaluation.audit import scan
from evaluation.claims import extract_claims, compare
from evaluation.harness import execute, Mode
from evaluation.media import prepare_media, file_hash


class FakeAdapters(RepositoryAdapters):
    """Real structured perception/fusion/reasoners; only model calls are mocked."""
    def __init__(self):
        super().__init__({})
        self.calls = []

    def detector(self, path):
        self.calls.append('detector')
        return WorldState(123., dict(image_width=32, image_height=24, detections=[
            dict(class_name='ball', confidence=.8, box_xyxy=[2, 2, 10, 10])], detection_count=1))

    def vlm(self, path):
        self.calls.append('vlm')
        return WorldState(456., dict(scene_description='A ball near a box.', model='test_model'))

    def fifth(self, state):
        self.calls.append('fifth')
        return super().fifth(state)


class HarnessTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.image = Path(self.temporary.name) / 'input.png'
        Image.new('RGB', (32, 24), 'white').save(self.image)

    def run_mode(self, mode, adapter=None):
        adapter = adapter or FakeAdapters()
        with prepare_media(self.image) as frames:
            return execute(frames, mode, adapter, run_id='test')[0], adapter

    def test_import_no_models_or_production_writes(self):
        before = file_hash('live_app.py')
        code = "import evaluation.runner,sys; assert 'fifth_layer.perception.smolvlm_scene' not in sys.modules; assert 'ultralytics' not in sys.modules"
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(file_hash('live_app.py'), before)

    def test_vlm_only_no_fifth_or_detector(self):
        result, adapter = self.run_mode(Mode.VLM_ONLY)
        self.assertEqual(adapter.calls, ['vlm'])
        self.assertIsNone(result['fifth_layer_output'])

    def test_fifth_only_no_vlm(self):
        result, adapter = self.run_mode(Mode.FIFTH_LAYER_ONLY)
        self.assertEqual(adapter.calls, ['detector', 'fifth'])
        self.assertIsNone(result['vlm_output'])
        self.assertNotIn('scene_description', result['fifth_layer_input']['data'])
        self.assertNotIn('semantic_evidence', result['fifth_layer_input']['data'])
        self.assertEqual(result['status'], 'ok')

    def test_combined_invokes_both_real_fusion(self):
        result, adapter = self.run_mode(Mode.VLM_PLUS_FIFTH_LAYER)
        self.assertEqual(adapter.calls, ['detector', 'vlm', 'fifth'])
        self.assertEqual(result['fifth_layer_input']['data']['scene_description'], 'A ball near a box.')
        self.assertIsNotNone(result['fusion_output'])
        self.assertIn('semantic_evidence', result['fifth_layer_input']['data'])

    def test_same_image_identity(self):
        with prepare_media(self.image) as frames:
            results = [execute(frames, mode, FakeAdapters(), run_id='same')[0] for mode in Mode]
        self.assertTrue(all(r['input'] == results[0]['input'] for r in results))
        self.assertEqual(results[0]['input']['source_sha256'], file_hash(self.image))
        self.assertTrue(compare(results)['comparable'])

    def test_temporary_snapshot_removed(self):
        with prepare_media(self.image) as frames:
            path = Path(frames[0]['evaluation_path'])
            self.assertTrue(path.exists())
        self.assertFalse(path.exists())
        self.assertTrue(self.image.exists())

    def test_same_video_selection_metadata(self):
        import cv2
        import numpy as np
        video = Path(self.temporary.name) / 'input.avi'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 10., (32, 24))
        self.assertTrue(writer.isOpened(), 'test requires MJPG video writer')
        try:
            for value in (0, 50, 100, 150):
                writer.write(np.full((24, 32, 3), value, dtype=np.uint8))
        finally:
            writer.release()
        with prepare_media(video, kind='video', start=1, stop=4, stride=2) as frames:
            self.assertEqual([f['frame_number'] for f in frames], [1, 3])
            self.assertEqual([f['timestamp'] for f in frames], [.1, .3])
            results = [execute(frames, mode, FakeAdapters(), run_id='video') for mode in Mode]
            self.assertEqual([r['input'] for r in results[0]], [r['input'] for r in results[1]])
            self.assertEqual([r['input'] for r in results[0]], [r['input'] for r in results[2]])

    def test_injected_vlm_field_rejected_before_reasoning(self):
        adapter = FakeAdapters()
        original = adapter.detector
        def dirty(path):
            state = original(path)
            state.data['scene_description'] = 'SECRET VLM FACT'
            return state
        with patch.object(adapter, 'detector', side_effect=dirty):
            result, _ = self.run_mode(Mode.FIFTH_LAYER_ONLY, adapter)
        self.assertEqual(result['status'], 'rejected_input')
        self.assertNotIn('fifth', adapter.calls)
        self.assertTrue(result['leakage_diagnostics']['findings'])

    def test_cached_vlm_state_detected(self):
        adapter = FakeAdapters()
        adapter.vlm_cache = {'scene_description': 'hidden fact'}
        result, _ = self.run_mode(Mode.FIFTH_LAYER_ONLY, adapter)
        self.assertEqual(result['status'], 'rejected_input')
        self.assertNotIn('fifth', adapter.calls)

    def test_serialized_snapshot_and_provenance_detected(self):
        for value in ({'snapshot': WorldState(data={'scene_description': 'x'})},
                      {'payload': json.dumps({'semantic_evidence': {'hidden': True}})},
                      {'provenance': {'source': 'SmolVLM'}}):
            self.assertTrue(scan(value))

    def test_unknown_detector_field_rejected(self):
        adapter = FakeAdapters()
        raw = adapter.detector('unused')
        raw.data['cached_future'] = 'secret'
        with patch.object(adapter, 'detector', return_value=raw):
            result, _ = self.run_mode(Mode.FIFTH_LAYER_ONLY, adapter)
        self.assertEqual(result['status'], 'rejected_input')

    def test_prepared_frame_tampering_rejected(self):
        with prepare_media(self.image) as frames:
            Path(frames[0]['evaluation_path']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                execute(frames, Mode.VLM_ONLY, FakeAdapters(), run_id='x')

    def test_json_raw_and_timing(self):
        result, _ = self.run_mode(Mode.VLM_PLUS_FIFTH_LAYER)
        copied = json.loads(json.dumps(result, allow_nan=False))
        self.assertEqual(copied['vlm_output']['timestamp'], 456.)
        self.assertEqual(copied['perception']['raw_detector']['timestamp'], 123.)
        self.assertEqual(copied['vlm_output']['data']['scene_description'], 'A ball near a box.')
        self.assertTrue(all(value >= 0 for value in result['timing'].values()))
        self.assertIn('fifth', result['provenance']['components'])

    def test_claim_normalization_exact_and_deterministic(self):
        raw = {'scene_description': ' A   BALL ', 'predicted': True}
        self.assertEqual(extract_claims(raw, 'test'), extract_claims(deepcopy(raw), 'test'))
        self.assertEqual({c['atom'] for c in extract_claims(raw, 'test')}, {'a ball', 'predicted=true'})

    def test_comparison_categories(self):
        def record(mode, atoms, field):
            return dict(input={'source_sha256': 'same'}, mode=mode, status='ok',
                        claims={field: [dict(atom=x) for x in atoms]})
        result = compare([record('VLM_ONLY', ['shared', 'vlm'], 'vlm'),
                          record('FIFTH_LAYER_ONLY', ['shared', 'inference'], 'fifth'),
                          record('VLM_PLUS_FIFTH_LAYER', ['new'], 'fusion')])
        self.assertEqual(result['shared'], ['shared'])
        self.assertEqual(result['vlm_only'], ['vlm'])
        self.assertEqual(result['fifth_layer_only'], ['inference'])
        self.assertEqual(result['combined_only'], ['new'])

    def test_different_inputs_rejected(self):
        a, _ = self.run_mode(Mode.VLM_ONLY)
        b = deepcopy(a)
        b['input']['frame_sha256'] = 'other'
        with self.assertRaisesRegex(ValueError, 'same source'):
            compare([a, b])

    def test_existing_state_does_not_cross_adapter_instances(self):
        a, b = FakeAdapters(), FakeAdapters()
        a.previous = [{'class_name': 'secret'}]
        result, _ = self.run_mode(Mode.FIFTH_LAYER_ONLY, b)
        self.assertNotIn('secret', json.dumps(result))

    def test_invalid_selection(self):
        for values in ({'start': -1}, {'stride': 0}, {'start': 3, 'stop': 2}):
            with self.assertRaises(ValueError), prepare_media(self.image, **values):
                pass

    def test_no_production_dependency(self):
        for path in [Path('live_app.py'), *Path('fifth_layer').rglob('*.py')]:
            self.assertNotIn('from evaluation', path.read_text(encoding='utf-8'))

    def test_module_global_cache_detected(self):
        import fifth_layer.reasoners.orchestrator as module
        with patch.object(module, 'test_cache', {'scene_description': 'secret'}, create=True):
            result, adapter = self.run_mode(Mode.FIFTH_LAYER_ONLY)
        self.assertEqual(result['status'], 'rejected_input')
        self.assertNotIn('fifth', adapter.calls)

    def test_late_cache_detected(self):
        adapter = FakeAdapters()
        original = adapter.fifth
        def dirty(state):
            output = original(state)
            adapter.vlm_cache = {'scene_description': 'late secret'}
            return output
        with patch.object(adapter, 'fifth', side_effect=dirty):
            result, _ = self.run_mode(Mode.FIFTH_LAYER_ONLY, adapter)
        self.assertEqual(result['status'], 'rejected_output')

    def test_runner_fresh_workers_and_mode_order(self):
        from evaluation.runner import run
        actual_run = subprocess.run
        code = ("import json,sys; from pathlib import Path; "
                "from evaluation.tests.test_harness import FakeAdapters; "
                "from evaluation.harness import execute; "
                "r=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')); "
                "result=execute(r['frames'],r['mode'],FakeAdapters(),run_id=r['run_id'],isolation='fresh_process'); "
                "Path(sys.argv[2]).write_text(json.dumps(result),encoding='utf-8')")
        def fixture_worker(command, **kwargs):
            self.assertEqual(command[1:3], ['-m', 'evaluation.worker'])
            self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
            return actual_run([sys.executable, '-c', code, *command[-2:]], **kwargs)
        with patch('evaluation.runner.subprocess.run', side_effect=fixture_worker):
            result = run(self.image, config={}, modes=list(reversed(list(Mode))))
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(len({r['provenance']['worker_pid'] for r in result['results']}), 3)
        self.assertTrue(result['production_files_unchanged'])
        self.assertTrue(result['comparisons'][0]['comparable'])

    def test_failed_worker_not_a_valid_comparison(self):
        from evaluation.runner import run
        failure = subprocess.CompletedProcess([], 1, stdout='', stderr='missing local model')
        with patch('evaluation.runner.subprocess.run', return_value=failure):
            result = run(self.image, config={})
        self.assertEqual(result['status'], 'incomplete_or_rejected')
        self.assertEqual(result['results'], [])
        self.assertEqual(result['comparisons'], [])
        self.assertTrue(all(w['stderr'] == 'missing local model' for w in result['workers']))

    def test_incomplete_modes_marked_not_comparable(self):
        result, _ = self.run_mode(Mode.VLM_ONLY)
        self.assertFalse(compare([result])['comparable'])


if __name__ == '__main__':
    unittest.main()
