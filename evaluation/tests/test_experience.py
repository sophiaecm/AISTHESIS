from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation.experience import evaluate_sequence, process_saved_result, main
from evaluation.sensory import _plain
from fifth_layer.world_model import SceneState, collect_evidence, MultiHypothesisGenerator


def frame(timestamp, direction='moving_left', observed=True):
    scene = SceneState(f's:{timestamp}', timestamp, image_width=300, image_height=400,
        observed_objects=[dict(track_id=7, box_xyxy=[0, 0, 10, 10])] if observed else [],
        motion_evidence=[dict(track_id=7, motion_state=direction)])
    evidence = collect_evidence(scene, include_sensory=True)
    hypotheses = MultiHypothesisGenerator().generate(scene, evidence)
    return dict(scene=scene, hypotheses=hypotheses, evidence=evidence)


def saved_result():
    records = []
    for sequence, timestamp in enumerate((10., 10.5, 11.)):
        f = frame(timestamp)
        records.append(dict(mode='FIFTH_LAYER_ONLY', status='ok',
            input=dict(sequence=sequence, timestamp=timestamp, source_sha256='fixture'),
            fifth_layer_input=dict(timestamp=timestamp, data=dict(image_width=300, image_height=400,
                accepted_detections=_plain(f['scene'].observed_objects),
                motion_evidence=_plain(f['scene'].motion_evidence))),
            sensory_world_model=_plain(dict(hypotheses=f['hypotheses'], evidence=f['evidence']))))
    return dict(schema_version='evaluation-0.1', run_id='run', results=records)


class ExperienceUtilityTests(unittest.TestCase):
    def test_target_frame_not_adjacent_frame(self):
        result = evaluate_sequence([frame(10.), frame(10.5, 'stationary'), frame(11.)])
        episodes = [e for e in result['episodes'] if e.prediction_timestamp == 10.]
        continuation = next(e for e in episodes if e.prediction_summary['hypothesis_type'] == 'continued_motion')
        self.assertEqual(continuation.observation_timestamp, 11.)
        self.assertEqual(continuation.evaluation_status, 'supported')

    def test_frame_permutation_stable(self):
        frames = [frame(10.), frame(11.), frame(12.)]
        self.assertEqual(evaluate_sequence(frames), evaluate_sequence(frames[::-1]))

    def test_last_predictions_recorded_as_unknown(self):
        result = evaluate_sequence([frame(10.)])
        self.assertTrue(result['episodes'])
        self.assertTrue(all(e.evaluation_status == 'unobservable' for e in result['episodes']))
        self.assertTrue(all(e.observation_timestamp is None for e in result['episodes']))

    def test_memory_bounded_separate_from_audit_report(self):
        result = evaluate_sequence([frame(10.), frame(11.)], capacity=1)
        self.assertGreater(len(result['episodes']), 1)
        self.assertEqual(result['memory_summary']['count'], 1)
        self.assertEqual(len(result['retained_episode_ids']), 1)

    def test_duplicates_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_sequence([frame(10.), frame(10.)])

    def test_unknown_timestamps_rejected(self):
        f = frame(10.)
        f['scene'] = replace(f['scene'], timestamp=None)
        with self.assertRaises(ValueError):
            evaluate_sequence([f])

    def test_selection_never_uses_outcome_quality(self):
        result = evaluate_sequence([frame(10.), frame(10.9, 'stationary'), frame(11.1)], time_tolerance_seconds=.2)
        e = next(e for e in result['episodes'] if e.prediction_timestamp == 10.
                 and e.prediction_summary['hypothesis_type'] == 'continued_motion')
        self.assertEqual(e.observation_timestamp, 10.9)
        self.assertEqual(e.evaluation_status, 'contradicted')

    def test_json_roundtrip_and_input_unchanged(self):
        source = saved_result()
        before = deepcopy(source)
        report = process_saved_result(source)
        self.assertEqual(source, before)
        self.assertEqual(json.loads(json.dumps(report)), report)
        self.assertEqual(report, process_saved_result(source))

    def test_no_stored_forecasts_are_not_regenerated(self):
        source = saved_result()
        for record in source['results']:
            del record['sensory_world_model']
        with patch.object(MultiHypothesisGenerator, 'generate', side_effect=AssertionError('inference invoked')):
            report = process_saved_result(source)
        self.assertEqual(report['sessions'], [])
        self.assertTrue(all(x['reason'] == 'no_stored_world_model_predictions' for x in report['diagnostics']))

    def test_postprocessing_does_not_call_inference_or_calibration(self):
        from fifth_layer.reasoners.auditory import AuditoryReasoner
        from fifth_layer.confidence_calibration import ConfidenceCalibrationMemory
        source = saved_result()
        with (patch.object(MultiHypothesisGenerator, 'generate', side_effect=AssertionError('generation')),
             patch.object(AuditoryReasoner, 'infer_expected_consequences', side_effect=AssertionError('sensory')),
             patch.object(ConfidenceCalibrationMemory, 'update', side_effect=AssertionError('calibration'))):
            report = process_saved_result(source)
        self.assertTrue(report['sessions'])

    def test_modes_and_sources_isolated(self):
        source = saved_result()
        other = deepcopy(source['results'])
        for record in other:
            record['mode'] = 'VLM_PLUS_FIFTH_LAYER'
        source['results'] += other
        self.assertEqual(len(process_saved_result(source)['sessions']), 2)

    def test_rejected_and_vlm_frames_not_outcomes(self):
        source = saved_result()
        source['results'][2]['status'] = 'rejected_input'
        result = process_saved_result(source)
        self.assertTrue(all(e['evaluation_status'] == 'unobservable'
                            for e in result['sessions'][0]['episodes']))
        source['results'][0]['mode'] = 'VLM_ONLY'
        self.assertEqual(len(process_saved_result(source)['diagnostics']), 2)

    def test_observation_timestamp_mismatch_rejected(self):
        source = saved_result()
        source['results'][0]['input']['timestamp'] = 100.
        with self.assertRaises(ValueError):
            process_saved_result(source)

    def test_cli_protects_input_and_existing_results(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / 'input_sensory.json'
            source.write_text(json.dumps(saved_result()), encoding='utf-8')
            before = source.read_bytes()
            output = Path(temporary) / 'new_experience.json'
            with patch('sys.argv', ['experience', str(source), '--output', str(output)]):
                main()
                written = output.read_bytes()
                with patch('sys.stderr'), self.assertRaises(SystemExit):
                    main()
                self.assertEqual(output.read_bytes(), written)
            self.assertEqual(source.read_bytes(), before)
            with patch('sys.argv', ['experience', str(source), '--output', str(Path(temporary) / 'results_case_a_combined.json')]):
                with patch('sys.stderr'), self.assertRaises(SystemExit):
                    main()


if __name__ == '__main__':
    unittest.main()
