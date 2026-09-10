"""Isolated v0.2 evidence, deterministic candidate and trajectory contracts."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
import unittest
from unittest.mock import patch

from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.future_state import FutureState
from fifth_layer.latent_state import LatentState
from fifth_layer.reasoners.physics import PhysicsReasoner
from fifth_layer.reasoners.temporal_prediction import TemporalPredictionReasoner
from fifth_layer.reasoners.occlusion import OcclusionReasoner
from fifth_layer.reasoners.semantic_conflict import SemanticConflictReasoner
from fifth_layer.perception.semantic_evidence import extract_semantic_evidence
from fifth_layer.world_model import (
    SceneState, WorldStateAdapter, EvidenceItem, EvidenceBundle, EvidenceSource,
    PhysicsEvidenceProvider, TemporalEvidenceProvider, OcclusionEvidenceProvider,
    SemanticEvidenceProvider, collect_evidence, MultiHypothesisGenerator,
    adapt_temporal_trajectories,
)


def scene(**changes):
    values = dict(scene_id='s', timestamp=10., image_width=640, image_height=480,
                  motion_evidence=[{'track_id': 7, 'motion_state': 'moving_right'}])
    values.update(changes)
    return SceneState(**values)


def item(identity='e', **changes):
    values = dict(evidence_id=identity, scene_id='s', source_type='motion',
                  source_component='test.explicit_source', evidence_type='motion_state',
                  value={'motion_state': 'moving_right'}, timestamp=10.,
                  supports=('continued_motion',), track_id=7)
    values.update(changes)
    return EvidenceItem(**values)


def generate(current=None, bundle=None, **kwargs):
    current = current if current is not None else scene()
    return MultiHypothesisGenerator().generate(current,
        bundle if bundle is not None else collect_evidence(current), **kwargs)


def temporal_output():
    state = WorldState(10., dict(image_width=640, image_height=480,
        motion_evidence=[dict(track_id=7, class_name='person', motion_state='moving_right',
                             current_center=[100, 100], velocity_x=10., velocity_y=0.,
                             normalized_motion=.1)], detections=[]))
    return TemporalPredictionReasoner().infer_expected_consequences(state)


class EvidenceTests(unittest.TestCase):
    def test_scene_to_bundle(self):
        bundle = collect_evidence(scene())
        self.assertEqual(bundle.scene_id, 's')
        self.assertEqual(len(bundle.items), 1)
        self.assertEqual(bundle.items[0].source_type, EvidenceSource.MOTION)

    def test_actual_physics_output(self):
        state = WorldState(10., {'position': [0, 0], 'velocity': [1, 0],
                                'dt': 1., 'occlusion_zone': [0, 0, 2, 2]})
        output = PhysicsReasoner().infer_expected_consequences(state)
        result = PhysicsEvidenceProvider().provide(scene(), output)[0]
        self.assertEqual(result.value['expected_next_position'], (1., 0.))
        self.assertIn('object_becomes_occluded', result.supports)
        self.assertIsNone(result.confidence)
        self.assertNotIn('hidden_actor_possible', result.supports)

    def test_actual_temporal_output(self):
        result = TemporalEvidenceProvider().provide(scene(), temporal_output())[0]
        self.assertEqual(result.track_id, 7)
        self.assertIn('continued_motion', result.supports)
        self.assertEqual(result.value['trajectory'][0]['center'][1], 100)

    def test_actual_occlusion_output(self):
        output = OcclusionReasoner().infer_expected_consequences(WorldState(10., {
            'occlusion_evidence': [{'object_id': 2, 'has_overlap_evidence': True}]}))
        result = OcclusionEvidenceProvider().provide(scene(), output)[0]
        self.assertEqual(result.object_id, 2)
        self.assertIn('object_becomes_occluded', result.supports)
        self.assertIsNone(result.confidence)  # occlusion_probability is not confidence
        self.assertEqual(result.value['occlusion_probability'], .35)

    def test_actual_semantic_outputs(self):
        current = scene(semantic_evidence={'semantic_evidence': extract_semantic_evidence('A person walking behind a car')})
        result = SemanticEvidenceProvider().provide(current)[0]
        self.assertIn('person', result.value['mentioned_objects'])
        self.assertEqual(result.supports, ())  # unbound mentions cannot invent actors
        conflict = SemanticConflictReasoner().analyze(WorldState(data={
            'detections': [{'class_name': 'person', 'confidence': .9}], 'scene_description': 'A car'}))
        result = SemanticEvidenceProvider().provide(current, conflict)[0]
        self.assertEqual(result.evidence_type, 'semantic_conflicts')
        self.assertEqual(result.contradicts, ())

    def test_provenance_fields_and_timestamp(self):
        result = TemporalEvidenceProvider().provide(scene(), temporal_output(), timestamp=8.)[0]
        self.assertEqual(result.timestamp, 8.)
        self.assertIn('ExpectedConsequences.predictions.motion_state', result.provenance['source_fields'])
        self.assertEqual(result.provenance['source_timestamp'], 8.)
        self.assertIn('TemporalPredictionReasoner', result.source_component)

    def test_source_timestamp_over_scene(self):
        current = scene(motion_evidence=[{'timestamp': 9., 'track_id': 7, 'motion_state': 'stationary'}])
        self.assertEqual(collect_evidence(current).items[0].timestamp, 9.)

    def test_inherited_observation_timestamp_preserved(self):
        output = FutureState(data={'motion_state': 'moving_right', 'observation_timestamp': 7.})
        result = TemporalEvidenceProvider().provide(scene(), output, timestamp=9.)[0]
        self.assertEqual(result.timestamp, 7.)
        self.assertEqual(result.value['observation_timestamp'], 7.)

    def test_provider_does_not_mutate_source(self):
        source = temporal_output()
        before = deepcopy(source)
        result = TemporalEvidenceProvider().provide(scene(), source)[0]
        self.assertEqual(source, before)
        source.predictions['trajectory'][0]['center'][0] = 999
        self.assertNotEqual(result.value['trajectory'][0]['center'][0], 999)

    def test_missing_outputs(self):
        current = scene(motion_evidence=[])
        self.assertEqual(collect_evidence(current).items, ())
        for provider in (PhysicsEvidenceProvider(), TemporalEvidenceProvider(),
                         OcclusionEvidenceProvider(), SemanticEvidenceProvider()):
            self.assertEqual(provider.provide(current, {}), ())

    def test_future_and_latent_wrappers(self):
        for output in (FutureState(data={'motion_state': 'moving_right'}),
                       LatentState(features={'motion_state': 'moving_right'})):
            self.assertIn('continued_motion', TemporalEvidenceProvider().provide(scene(), output)[0].supports)

    def test_images_never_retained(self):
        import numpy as np
        output = {'motion_state': 'moving_right', 'frame': np.zeros((2, 2)),
                  'trajectory': [{'center': [1, 2], 'horizon_seconds': 1., 'extra': np.zeros((2, 2))}]}
        result = TemporalEvidenceProvider().provide(scene(), output)[0]
        self.assertNotIn('frame', result.value)
        self.assertNotIn('extra', result.value['trajectory'][0])

    def test_nested_immutability(self):
        original = {'nested': [{'tags': {'a'}}]}
        result = item(value=original)
        original['nested'][0]['tags'].add('b')
        self.assertEqual(result.value['nested'][0]['tags'], frozenset({'a'}))
        with self.assertRaises(TypeError):
            result.value['nested'][0]['new'] = 2
        with self.assertRaises(FrozenInstanceError):
            result.confidence = .8

    def test_bundle_validation_and_immutability(self):
        entries = [item()]
        bundle = EvidenceBundle('s', entries)
        entries.clear()
        self.assertEqual(len(bundle.items), 1)
        with self.assertRaisesRegex(ValueError, 'scene_id'):
            EvidenceBundle('other', [item()])
        with self.assertRaisesRegex(ValueError, 'duplicate evidence_id'):
            EvidenceBundle('s', [item(), item()])
        with self.assertRaises(FrozenInstanceError):
            bundle.items = ()

    def test_evidence_validation(self):
        for changes in ({'confidence': 1.1}, {'timestamp': -1}, {'source_type': 'auditory'},
                        {'evidence_id': ''}, {'value': []}, {'track_id': []}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                item(**changes)

    def test_no_hidden_reasoner_calls(self):
        with patch.object(TemporalPredictionReasoner, 'infer_expected_consequences', side_effect=AssertionError), \
             patch.object(PhysicsReasoner, 'infer_expected_consequences', side_effect=AssertionError):
            generate()

    def test_adapter_chain_provenance(self):
        current = WorldStateAdapter.from_world_state(WorldState(10., {'motion_evidence': [
            {'track_id': 7, 'motion_state': 'moving_right'}]}), scene_id='s')
        result = collect_evidence(current).items[0]
        self.assertEqual(result.provenance['scene_provenance']['source_field_mapping']['motion_evidence'],
                         'WorldState.data.motion_evidence')

    def test_nested_orchestrator_occlusion_output(self):
        current = scene(occlusion_evidence={'occlusion_reasoning': {'expected': {
            'occlusion_hypotheses': [{'object_id': 0, 'hypothesis': 'occlusion_possible'}]}}})
        result = OcclusionEvidenceProvider().provide(current)[0]
        self.assertIn('object_becomes_occluded', result.supports)
        self.assertIn('SceneState.occlusion_evidence.occlusion_reasoning.expected.occlusion_hypotheses[*].object_id',
                      result.provenance['source_fields'])

    def test_physics_input_confidence_preserved(self):
        current = scene(physics_evidence={'physics_confidence': .7})
        self.assertEqual(PhysicsEvidenceProvider().provide(current)[0].confidence, .7)

    def test_identical_duplicate_provider_records_merged(self):
        record = {'track_id': 7, 'motion_state': 'moving_right'}
        self.assertEqual(len(collect_evidence(scene(motion_evidence=[record, record])).items), 1)


class GenerationTests(unittest.TestCase):
    def test_multiple_active_same_scene(self):
        result = generate()
        self.assertEqual({h.hypothesis_type for h in result.hypotheses}, {'continued_motion', 'object_stops'})
        self.assertTrue(all(h.status == 'active' and h.scene_id == 's' for h in result.hypotheses))

    def test_evidence_links_and_assumptions(self):
        bundle = collect_evidence(scene())
        result = generate(bundle=bundle)
        for h in result.hypotheses:
            self.assertEqual(h.evidence_for, (bundle.items[0].evidence_id,))
            self.assertTrue(h.assumptions)
            self.assertTrue(h.provenance['generation_rule'])
        stop = next(h for h in result.hypotheses if h.hypothesis_type == 'object_stops')
        self.assertEqual(stop.evidence_against, (bundle.items[0].evidence_id,))

    def test_exact_scoring(self):
        result = generate()
        by_type = {h.hypothesis_type: h for h in result.hypotheses}
        self.assertEqual(by_type['continued_motion'].posterior_probability, 1.)
        self.assertEqual(by_type['object_stops'].posterior_probability, .125)
        self.assertTrue(all(h.prior_probability == .5 for h in result.hypotheses))
        self.assertTrue(all(h.provenance['calibrated'] is False for h in result.hypotheses))
        self.assertAlmostEqual(sum(result.normalized_posteriors.values()), 1.)

    def test_confidence_separate_from_scores(self):
        a = generate(bundle=EvidenceBundle('s', [item(confidence=.2)]))
        b = generate(bundle=EvidenceBundle('s', [item(confidence=.9)]))
        self.assertEqual([h.posterior_probability for h in a.hypotheses], [h.posterior_probability for h in b.hypotheses])
        self.assertTrue(all(h.confidence == .2 for h in a.hypotheses))
        self.assertTrue(all(h.confidence == .9 for h in b.hypotheses))

    def test_unknown_confidence_flag(self):
        for h in generate().hypotheses:
            self.assertEqual(h.confidence, 0.)
            self.assertFalse(h.provenance['confidence_assessed'])

    def test_missing_not_negative(self):
        result = generate(scene(motion_evidence=[]))
        self.assertEqual(len(result.hypotheses), 1)
        h = result.hypotheses[0]
        self.assertEqual(h.hypothesis_type, 'unknown_dynamic_event')
        self.assertEqual(h.evidence_against, ())
        self.assertEqual(h.posterior_probability, 0.)
        self.assertEqual(tuple(result.normalized_posteriors.values()), (1.,))
        self.assertEqual(h.status, 'active')

    def test_conflicts_retained(self):
        bundle = EvidenceBundle('s', [item('moving', contradicts=('object_stops',)),
            item('stationary', supports=('object_stops',), contradicts=('continued_motion',))])
        result = generate(bundle=bundle)
        self.assertEqual(len(result.hypotheses), 2)
        for h in result.hypotheses:
            self.assertEqual(len(h.evidence_for), 1)
            self.assertEqual(len(h.evidence_against), 1)
            self.assertEqual(h.posterior_probability, .5)

    def test_deterministic_order_ids_scores_and_links(self):
        items = [item('b'), item('a', confidence=.8)]
        a = generate(bundle=EvidenceBundle('s', items))
        b = generate(bundle=EvidenceBundle('s', list(reversed(items))))
        self.assertEqual(a, b)
        self.assertEqual(a.normalized_posteriors, b.normalized_posteriors)

    def test_duplicate_rule_candidates_merged(self):
        result = generate(bundle=EvidenceBundle('s', [item('a'), item('b')]))
        self.assertEqual(len(result.hypotheses), 2)
        self.assertEqual(len({h.hypothesis_id for h in result.hypotheses}), 2)
        self.assertTrue(all(len(h.evidence_for) == 2 for h in result.hypotheses))

    def test_weaker_candidate_retained(self):
        result = generate()
        winner = result.highest_probability_hypothesis
        self.assertEqual(winner.hypothesis_type, 'continued_motion')
        self.assertEqual(len(result.hypotheses), 2)

    def test_no_unsupported_actors_or_falls(self):
        current = scene(motion_evidence=[], physics_evidence={'physics_hidden_interaction_possible': True},
                        semantic_evidence={'semantic_evidence': extract_semantic_evidence('A person behind a car falling')})
        result = generate(current)
        self.assertTrue(all(h.hypothesis_type == 'unknown_dynamic_event' for h in result.hypotheses))
        self.assertFalse(any('child' in h.statement for h in result.hypotheses))

    def test_associated_occlusion_motion_continuity(self):
        current = scene(occlusion_evidence={'occlusion_evidence': [
            {'track_id': 7, 'has_overlap_evidence': True}]})
        kinds = {h.hypothesis_type for h in generate(current).hypotheses}
        self.assertIn('object_reappears', kinds)
        self.assertIn('object_becomes_occluded', kinds)

    def test_different_tracks_not_joined(self):
        current = scene(occlusion_evidence={'occlusion_evidence': [
            {'track_id': 8, 'has_overlap_evidence': True}]})
        self.assertNotIn('object_reappears', {h.hypothesis_type for h in generate(current).hypotheses})

    def test_no_class_name_association(self):
        current = scene(motion_evidence=[{'class_name': 'person', 'motion_state': 'moving_right'}],
                        occlusion_evidence={'occlusion_evidence': [{'class_name': 'person', 'has_overlap_evidence': True}]})
        self.assertNotIn('object_reappears', {h.hypothesis_type for h in generate(current).hypotheses})

    def test_object_id_association_without_track(self):
        current = scene(motion_evidence=[{'current_object_id': 2, 'motion_state': 'moving_right'}],
                        occlusion_evidence={'occlusion_evidence': [{'object_id': 2, 'has_overlap_evidence': True}]})
        reappears = next(h for h in generate(current).hypotheses if h.hypothesis_type == 'object_reappears')
        self.assertIsNone(reappears.track_id)
        self.assertEqual(reappears.provenance['association'], ('object', 'int', 2))

    def test_scene_mismatch_and_unknown_time(self):
        with self.assertRaisesRegex(ValueError, 'scene_id'):
            generate(bundle=EvidenceBundle('other'))
        with self.assertRaisesRegex(ValueError, 'created_timestamp'):
            generate(scene(timestamp=None))
        self.assertEqual(generate(scene(timestamp=None), created_timestamp=2.).created_timestamp, 2.)

    def test_predicted_track_not_observation(self):
        current = scene(motion_evidence=[], predicted_tracks=[dict(track_id=7,
            is_predicted=True, observation_state='predicted', possible_occlusion=True,
            position_uncertainty=25., position_uncertainty_units='pixels')])
        result = generate(current)
        self.assertEqual(result.hypotheses[0].hypothesis_type, 'object_reappears')
        self.assertEqual(current.observed_objects, ())
        self.assertEqual(result.hypotheses[0].provenance['score_inputs'][0]['value']['position_uncertainty'], 25.)

    def test_unrecognized_support_does_not_invent_category(self):
        result = generate(bundle=EvidenceBundle('s', [item(supports=('child_is_behind_car',))]))
        self.assertEqual(result.hypotheses[0].hypothesis_type, 'unknown_dynamic_event')

    def test_zero_score_multiple_unknowns_uniform(self):
        bundle = EvidenceBundle('s', [item('a', supports=(), track_id=1), item('b', supports=(), track_id=2)])
        result = generate(bundle=bundle)
        self.assertEqual(tuple(result.normalized_posteriors.values()), (.5, .5))

    def test_input_uncertainty_retained_not_scored(self):
        result = generate(scene(uncertainty=.9))
        self.assertTrue(all(h.provenance['scene_uncertainty'] == .9 for h in result.hypotheses))
        self.assertEqual([h.posterior_probability for h in result.hypotheses],
                         [h.posterior_probability for h in generate(scene(uncertainty=.1)).hypotheses])


class TrajectoryAndBoundaryTests(unittest.TestCase):
    def test_real_temporal_trajectory_linkage(self):
        current = scene()
        output = temporal_output()
        bundle = collect_evidence(current, temporal=output)
        hypotheses = generate(current, bundle, horizon_seconds=3.)
        trajectories = adapt_temporal_trajectories(current, bundle, hypotheses)
        self.assertEqual(len(trajectories), 1)
        trajectory = trajectories[0]
        owner = next(h for h in hypotheses.hypotheses if h.hypothesis_id == trajectory.hypothesis_id)
        self.assertEqual(owner.hypothesis_type, 'continued_motion')
        self.assertEqual(trajectory.track_id, 7)
        self.assertEqual(trajectory.start_timestamp, 10.)
        self.assertEqual(trajectory.predicted_centers, tuple(tuple(p['center']) for p in output.predictions['trajectory']))
        self.assertEqual(trajectory.predicted_bboxes, ())
        self.assertEqual(trajectory.uncertainty_by_step, ())

    def test_no_trajectory_fabrication(self):
        current = scene()
        bundle = collect_evidence(current)
        self.assertEqual(adapt_temporal_trajectories(current, bundle, generate(current, bundle)), ())

    def test_old_trajectory_not_retimed(self):
        current = scene()
        bundle = collect_evidence(current, temporal=temporal_output(), source_timestamps={'temporal': 8.})
        self.assertEqual(adapt_temporal_trajectories(current, bundle, generate(current, bundle)), ())

    def test_horizon_clipping_no_extrapolation(self):
        current = scene()
        bundle = collect_evidence(current, temporal=temporal_output())
        trajectories = adapt_temporal_trajectories(current, bundle, generate(current, bundle, horizon_seconds=.1))
        self.assertEqual(trajectories, ())

    def test_no_memory_calls(self):
        from fifth_layer.world_model import ExperienceMemory
        with patch.object(ExperienceMemory, 'recent', side_effect=AssertionError), \
             patch.object(ExperienceMemory, 'add', side_effect=AssertionError):
            generate()

    def test_production_dependency_direction(self):
        self.assertNotIn('world_model', Path('live_app.py').read_text(encoding='utf-8'))
        for path in Path('fifth_layer/reasoners').glob('*.py'):
            self.assertNotIn('world_model', path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
