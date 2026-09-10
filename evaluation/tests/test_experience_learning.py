from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation.experience_learning import ablate, main
from evaluation.experience import process_saved_result
from evaluation.sensory import _plain
from evaluation.tests.test_experience import saved_result
from fifth_layer.world_model import ExperienceMemory, MultiHypothesisGenerator
from fifth_layer.world_model.experience_learning import ExperienceQuery, ExperienceRetriever
from test_experience_learning import current, historical


def inputs():
    saved = saved_result()
    # Synthetic stored frames, produced by existing generator only in this fixture.
    for sequence, record in enumerate(saved['results']):
        time = (1., 2., 4.)[sequence]
        scene, evidence, hypotheses = current(time=time, name=f's:{time}')
        record['input']['timestamp'] = time
        record['fifth_layer_input'] = dict(timestamp=time, data=dict(image_width=100, image_height=100,
            accepted_detections=_plain(scene.observed_objects), motion_evidence=_plain(scene.motion_evidence)))
        record['sensory_world_model'] = _plain(dict(evidence=evidence, hypotheses=hypotheses))
    return saved, process_saved_result(saved)


class ExperienceAblationTests(unittest.TestCase):
    def test_same_current_inputs_and_base_scores(self):
        saved, history = inputs()
        report = ablate(saved, history)
        frames = report['sessions'][0]['frames']
        self.assertTrue(frames[-1]['affected_hypotheses'])
        for frame in frames:
            base, enabled = frame['NO_EXPERIENCE'], frame['EXPERIENCE_ENABLED']
            self.assertEqual(base['hypotheses'], enabled['hypotheses'])
            self.assertEqual([r['original_heuristic_score'] for r in base['ranking']],
                             [r['original_heuristic_score'] for r in enabled['ranking']])

    def test_future_and_equal_time_not_admitted(self):
        report = ablate(*inputs())
        frames = report['sessions'][0]['frames']
        self.assertEqual(frames[0]['leakage_audit']['admitted_before_query'], [])
        self.assertEqual(frames[1]['leakage_audit']['admitted_before_query'], [])
        self.assertTrue(frames[-1]['leakage_audit']['admitted_before_query'])
        for frame in frames:
            for item in frame['EXPERIENCE_ENABLED']['experience_evidence']['items']:
                self.assertTrue(all(t < frame['timestamp'] for t in item['provenance']['historical_timestamps'].values()))

    def test_future_history_insertion_does_not_evict_past(self):
        saved, history = inputs()
        baseline = ablate(saved, history, capacity=2)
        history['sessions'][0]['episodes'].extend(_plain(historical(start=100. + i, name=f'future-{i}'))
                                                   for i in range(5))
        self.assertEqual(baseline, ablate(saved, history, capacity=2))

    def test_input_and_episode_order_invariance(self):
        saved, history = inputs()
        before = ablate(saved, history)
        saved['results'].reverse()
        history['sessions'][0]['episodes'].reverse()
        self.assertEqual(before, ablate(saved, history))

    def test_inputs_not_mutated(self):
        saved, history = inputs()
        before = deepcopy((saved, history))
        ablate(saved, history)
        self.assertEqual(before, (saved, history))

    def test_byte_equivalent_json(self):
        args = inputs()
        self.assertEqual(json.dumps(ablate(*args), sort_keys=True), json.dumps(ablate(*args), sort_keys=True))

    def test_other_run_clock_domain_rejected(self):
        saved, history = inputs()
        history['source_run_id'] = 'other-run'
        with self.assertRaises(ValueError):
            ablate(saved, history)

    def test_other_mode_and_source_do_not_transfer(self):
        saved, history = inputs()
        for mode, source in (('VLM_PLUS_FIFTH_LAYER', 'fixture'), ('FIFTH_LAYER_ONLY', 'other-source')):
            changed = deepcopy(history)
            changed['sessions'][0]['mode'] = mode
            changed['sessions'][0]['source_sha256'] = source
            report = ablate(saved, changed)
            self.assertFalse(any(f['affected_hypotheses'] for f in report['sessions'][0]['frames']))

    def test_no_models_reasoners_evaluation_or_training(self):
        from fifth_layer.reasoners.auditory import AuditoryReasoner
        args = inputs()
        with (patch.object(MultiHypothesisGenerator, 'generate', side_effect=AssertionError('regeneration')),
              patch.object(AuditoryReasoner, 'infer_expected_consequences', side_effect=AssertionError('reasoning')),
              patch('evaluation.experience.evaluate_prediction', side_effect=AssertionError('reevaluation'))):
            self.assertTrue(ablate(*args)['sessions'])

    def test_snapshot_no_hidden_clock_and_no_ttl_mutation(self):
        m = ExperienceMemory(clock=lambda: 0.)
        e = historical()
        m.add(e)
        scene, evidence, hypotheses = current(time=60.)
        h = next(h for h in hypotheses.hypotheses if h.hypothesis_type == 'continued_motion')
        query = ExperienceQuery.from_current(scene, evidence, h)
        with patch.object(m, '_clock', side_effect=AssertionError('clock read')):
            result = ExperienceRetriever().retrieve(query, m, current_time=60.)
        self.assertEqual(result[0], ())
        self.assertEqual(len(m._episodes), 1)

    def test_cli_existing_and_baseline_paths_protected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            saved, history = inputs()
            source, episodes = root / 'sensory.json', root / 'experience.json'
            source.write_text(json.dumps(saved), encoding='utf-8')
            episodes.write_text(json.dumps(history), encoding='utf-8')
            before = source.read_bytes(), episodes.read_bytes()
            output = root / 'new_experience_learning.json'
            with patch('sys.argv', ['ablate', str(source), '--history', str(episodes), '--output', str(output)]):
                main()
                written = output.read_bytes()
                with patch('sys.stderr'), self.assertRaises(SystemExit):
                    main()
                self.assertEqual(output.read_bytes(), written)
            self.assertEqual((source.read_bytes(), episodes.read_bytes()), before)
            with patch('sys.argv', ['ablate', str(source), '--history', str(episodes), '--output', str(root / 'results_case_a_combined.json')]):
                with patch('sys.stderr'), self.assertRaises(SystemExit):
                    main()

    def test_duplicate_history_rejected(self):
        saved, history = inputs()
        history['sessions'][0]['episodes'].append(history['sessions'][0]['episodes'][0])
        with self.assertRaises(ValueError):
            ablate(saved, history)


if __name__ == '__main__':
    unittest.main()
