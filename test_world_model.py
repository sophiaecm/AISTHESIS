"""Foundation contracts; deterministic tests without models, cameras or network."""
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError
from threading import Barrier
import unittest
from unittest.mock import patch

import numpy as np

from fifth_layer.world_state import WorldState
from fifth_layer.latent_state import LatentState
from fifth_layer.prediction_error import PredictionError
from fifth_layer.perception.analysis_snapshot import AnalysisSnapshot
from fifth_layer.perception.tracking import TrackMemory
from fifth_layer.perception.perception_fusion import PerceptionFusion
from fifth_layer.world_model import (
    SceneState, Hypothesis, HypothesisSet, HypothesisStatus, FutureTrajectory,
    validate_trajectories, ExperienceEpisode, ExperienceMemory, EvaluationStatus,
    WorldStateAdapter,
)


def hypothesis(identity='h1', **kwargs):
    values = dict(hypothesis_id=identity, scene_id='s1', hypothesis_type='manual',
                  statement='Object may remain hidden', prior_probability=.2,
                  posterior_probability=.3, confidence=.6, created_timestamp=10.,
                  horizon_seconds=1., target_timestamp=11.)
    values.update(kwargs)
    return Hypothesis(**values)


def episode(identity='e1', **kwargs):
    values = dict(episode_id=identity, source_scene_id='s1', hypothesis_id='h1',
                  created_timestamp=10.)
    values.update(kwargs)
    return ExperienceEpisode(**values)


def trajectory(**kwargs):
    values = dict(trajectory_id='t1', hypothesis_id='h1', track_id=7,
                  start_timestamp=10., horizon_seconds=1.,
                  predicted_centers=[[2., 3.]], predicted_bboxes=[[0, 0, 4, 6]],
                  predicted_states=[{'timestamp': 11., 'center': [2., 3.]}],
                  uncertainty_by_step=[.3], expected_event='remains_hidden')
    values.update(kwargs)
    return FutureTrajectory(**values)


class AdapterTests(unittest.TestCase):
    def source(self):
        return WorldState(10., dict(
            image_width=640, image_height=480,
            detections=[dict(track_id=1, class_name='person', confidence=.9, box=[2, 3, 4, 5])],
            predicted_tracks=[dict(track_id=2, predicted_center=[30, 40],
                                   predicted_bbox=[20, 30, 40, 50], is_predicted=True,
                                   observation_state='predicted', position_uncertainty=25.,
                                   position_uncertainty_units='pixels')],
            scene_relations=[{'relation': 'near'}], motion_evidence=[{'track_id': 1}],
            occlusion_evidence=[{'occlusion_score': .5}],
            occlusion_reasoning={'latent': {'possible_occluded_objects': []}},
            position=[2, 3], velocity=[1, 0], dt=1., occlusion_zone=[0, 0, 10, 10],
            physics_hidden_interaction_possible=True, physics_confidence=.7,
            latent_temporal_state='moving', semantic_evidence={'actor': True},
            scene_description='A person', risk_level='low', fused_uncertainty=.4,
            observation_timestamp=9., source_type='image', source_path='example.jpg',
            perception_sources={'detector': 'existing-detector'}))

    def test_legacy_conversion(self):
        scene = WorldStateAdapter.from_world_state(self.source(), scene_id='s1')
        self.assertEqual(scene.timestamp, 10.)
        self.assertEqual((scene.image_width, scene.image_height), (640, 480))
        self.assertEqual(scene.observed_objects[0]['box'], (2, 3, 4, 5))
        self.assertEqual(scene.risk, 'low')
        self.assertEqual(scene.uncertainty, .4)
        self.assertEqual(scene.spatial_relations[0]['relation'], 'near')
        self.assertEqual(scene.physics_evidence['velocity'], (1, 0))
        self.assertEqual(scene.latent_evidence['latent_temporal_state'], 'moving')
        self.assertTrue(scene.semantic_evidence['semantic_evidence']['actor'])

    def test_observed_predicted_separation(self):
        scene = WorldStateAdapter.from_world_state(self.source())
        self.assertEqual([d['track_id'] for d in scene.observed_objects], [1])
        self.assertEqual([d['track_id'] for d in scene.predicted_tracks], [2])
        self.assertEqual(scene.predicted_tracks[0]['position_uncertainty'], 25.)
        self.assertEqual(scene.predicted_tracks[0]['position_uncertainty_units'], 'pixels')

    def test_accepted_precedence_and_empty(self):
        state = self.source()
        state.data['accepted_detections'] = []
        self.assertEqual(WorldStateAdapter.from_world_state(state).observed_objects, ())
        state.data['accepted_detections'] = [{'track_id': 3}]
        self.assertEqual(WorldStateAdapter.from_world_state(state).observed_objects[0]['track_id'], 3)

    def test_explicit_nonobservations_excluded(self):
        state = self.source()
        state.data['detections'].extend([{'is_predicted': True}, {'observation_state': 'missing'}])
        self.assertEqual(len(WorldStateAdapter.from_world_state(state).observed_objects), 1)

    def test_input_unchanged_and_detached(self):
        state = self.source()
        before = deepcopy(state)
        scene = WorldStateAdapter.from_world_state(state)
        self.assertEqual(state, before)
        state.data['detections'][0]['box'][0] = 999
        state.data['perception_sources']['detector'] = 'changed'
        self.assertEqual(scene.observed_objects[0]['box'][0], 2)
        self.assertEqual(scene.provenance['source_metadata']['perception_sources']['detector'], 'existing-detector')

    def test_optional_fields_and_unknown_ids(self):
        scene = WorldStateAdapter.from_world_state(WorldState())
        self.assertTrue(scene.scene_id)
        self.assertIsNone(scene.timestamp)
        self.assertIsNone(scene.source_world_state_id)
        self.assertIsNone(scene.snapshot_sequence_id)
        self.assertIsNone(scene.risk)
        self.assertIsNone(scene.uncertainty)
        self.assertEqual(scene.observed_objects, ())
        self.assertEqual(scene.predicted_tracks, ())

    def test_mapping_and_timestamps(self):
        state = self.source()
        scene = WorldStateAdapter.from_world_state(state, source_world_state_id='external-1',
            snapshot_sequence_id=3, provenance={'source_component_version': 'known-version'})
        self.assertEqual(scene.source_world_state_id, 'external-1')
        self.assertEqual(scene.snapshot_sequence_id, 3)
        mapping = scene.provenance['source_field_mapping']
        self.assertEqual(mapping['observed_objects'], 'WorldState.data.detections')
        self.assertEqual(mapping['predicted_tracks'], 'WorldState.data.predicted_tracks')
        self.assertEqual(mapping['spatial_relations'], 'WorldState.data.scene_relations')
        self.assertEqual(mapping['physics_evidence.position'], 'WorldState.data.position')
        self.assertEqual(scene.provenance['source_timestamps']['WorldState.data.observation_timestamp'], 9.)
        self.assertEqual(scene.provenance['source_timestamps']['WorldState.timestamp'], 10.)
        self.assertEqual(scene.provenance['supplied']['source_component_version'], 'known-version')
        self.assertNotIn('source_component_version', scene.provenance)

    def test_no_visual_buffers_or_hidden_provenance_references(self):
        from PIL import Image
        state = self.source()
        array = np.zeros((2, 2, 3), dtype=np.uint8)
        buffers = dict(image=array, frame=array, raw_image=array, pixels=b'raw',
                       unknown_array=array, pil=Image.new('RGB', (2, 2)),
                       encoded_image='base64-data', unknown_bytes=b'raw')
        state.data.update(buffers)
        state.data['detections'][0]['extra'] = dict(buffers, safe='kept')
        scene = WorldStateAdapter.from_world_state(state,
            provenance={'nested': dict(buffers, safe='kept'), 'list': [array, {'frame': array}]})
        self.assertEqual(dict(scene.observed_objects[0]['extra']), {'safe': 'kept'})
        self.assertEqual(dict(scene.provenance['supplied']['nested']), {'safe': 'kept'})
        self.assertIs(state.data['frame'], array)
        def check(value):
            if isinstance(value, Mapping):
                for child in value.values():
                    check(child)
            elif isinstance(value, (tuple, frozenset)):
                for child in value:
                    check(child)
            else:
                self.assertIn(type(value), (str, int, float, bool, type(None)))
        for value in vars(scene).values():
            check(value)

    def test_actual_tracker_and_fusion_outputs(self):
        tracker = TrackMemory(observation_expiration=True, predict_missing_tracks=True)
        detections = [dict(class_name='person', confidence=.9, box_xyxy=[10, 10, 30, 50])]
        observed = tracker.update(detections, 1., 640, 480)
        state = PerceptionFusion().fuse(WorldState(1., dict(detections=observed,
            image_width=640, image_height=480)), WorldState(1., {'scene_description': 'person'}))
        scene = WorldStateAdapter.from_world_state(state)
        self.assertEqual(len(scene.observed_objects), 1)
        tracker.update([], 1.1, 640, 480)
        state.data.update(detections=[], predicted_tracks=tracker.predicted_tracks)
        later = WorldStateAdapter.from_world_state(state)
        self.assertEqual(later.observed_objects, ())
        self.assertEqual(len(later.predicted_tracks), 1)

    def test_snapshot_roundtrip(self):
        snapshot = AnalysisSnapshot.capture(np.zeros((2, 2, 3), dtype=np.uint8), self.source(), [])
        scene = WorldStateAdapter.from_world_state(snapshot.world_state())
        self.assertEqual(scene.timestamp, snapshot.timestamp)
        self.assertFalse(hasattr(scene, 'pixels'))

    def test_uncertainty_precedence(self):
        state = self.source()
        state.data['uncertainty'] = .2
        self.assertEqual(WorldStateAdapter.from_world_state(state).uncertainty, .2)


class HypothesisTests(unittest.TestCase):
    def test_three_normalized_and_raw_preserved(self):
        raw = [hypothesis(f'h{i}', posterior_probability=p) for i, p in enumerate((.2, .3, .1))]
        group = HypothesisSet('s1', raw, 10.)
        self.assertAlmostEqual(sum(group.normalized_posteriors.values()), 1.)
        self.assertEqual([h.prior_probability for h in group.hypotheses], [.2] * 3)
        self.assertEqual([h.posterior_probability for h in group.hypotheses], [.2, .3, .1])
        self.assertEqual(group.normalized_posteriors, group.normalized_posteriors)
        self.assertEqual(group.highest_probability_hypothesis.hypothesis_id, 'h1')
        self.assertEqual(len(group.hypotheses), 3)
        raw.clear()
        self.assertEqual(len(group.hypotheses), 3)

    def test_zero_sum_uniform(self):
        group = HypothesisSet('s1', [hypothesis(str(i), posterior_probability=0) for i in range(3)], 10.)
        self.assertEqual(list(group.normalized_posteriors.values()), [1/3] * 3)
        self.assertEqual(group.highest_probability_hypothesis.hypothesis_id, '0')

    def test_inactive_statuses_excluded_but_retained(self):
        inactive = [hypothesis(s.value, status=s, posterior_probability=1.)
                    for s in HypothesisStatus if s != HypothesisStatus.ACTIVE]
        group = HypothesisSet('s1', [hypothesis()] + inactive, 10.)
        self.assertEqual(dict(group.normalized_posteriors), {'h1': 1.})
        self.assertEqual(len(group.hypotheses), 5)
        self.assertEqual(group.highest_probability_hypothesis.hypothesis_id, 'h1')

    def test_empty_and_no_active(self):
        for entries in ([], [hypothesis(status='expired')]):
            group = HypothesisSet('s1', entries, 10.)
            self.assertEqual(dict(group.normalized_posteriors), {})
            self.assertIsNone(group.highest_probability_hypothesis)

    def test_invalid_probabilities(self):
        for name in ('prior_probability', 'posterior_probability'):
            for value in (-.1, 1.1, float('nan'), float('inf'), None, True):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    hypothesis(**{name: value})

    def test_invalid_confidence(self):
        with self.assertRaisesRegex(ValueError, 'confidence'):
            hypothesis(confidence=1.1)

    def test_boundary_probabilities(self):
        self.assertEqual(hypothesis(prior_probability=0., posterior_probability=1.).posterior_probability, 1.)

    def test_different_scene_rejected(self):
        with self.assertRaisesRegex(ValueError, 'scene_id'):
            HypothesisSet('other', [hypothesis()], 10.)

    def test_duplicate_hypotheses(self):
        with self.assertRaisesRegex(ValueError, 'duplicate hypothesis_id'):
            HypothesisSet('s1', [hypothesis(), hypothesis()], 10.)

    def test_invalid_status(self):
        with self.assertRaisesRegex(ValueError, 'status'):
            hypothesis(status='winner')

    def test_hypothesis_time_validation(self):
        for key in ('created_timestamp', 'horizon_seconds', 'target_timestamp'):
            for value in (-1, None, float('nan')):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, key):
                    hypothesis(**{key: value})

    def test_normalization_view_readonly(self):
        group = HypothesisSet('s1', [hypothesis()], 10.)
        with self.assertRaises(TypeError):
            group.normalized_posteriors['h1'] = .2

    def test_invalid_normalization_method(self):
        with self.assertRaisesRegex(ValueError, 'normalization_method'):
            HypothesisSet('s1', [], 10., 'softmax')


class TrajectoryTests(unittest.TestCase):
    def test_valid_construction(self):
        result = trajectory()
        self.assertEqual(result.predicted_centers, ((2., 3.),))
        self.assertEqual(result.predicted_bboxes, ((0, 0, 4, 6),))
        self.assertEqual(result.hypothesis_id, 'h1')

    def test_bbox_validation(self):
        for bbox in ([1, 2, 3], [3, 2, 1, 4], [0, 0, float('nan'), 1]):
            with self.subTest(bbox=bbox), self.assertRaisesRegex(ValueError, 'predicted_bboxes'):
                trajectory(predicted_bboxes=[bbox])

    def test_center_validation(self):
        for center in ([1], [1, 2, 3], [None, 1], [float('inf'), 0]):
            with self.subTest(center=center), self.assertRaisesRegex(ValueError, 'predicted_centers'):
                trajectory(predicted_centers=[center])

    def test_uncertainty_validation(self):
        for value in (-.1, 1.1, float('nan')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'uncertainty_by_step'):
                trajectory(uncertainty_by_step=[value])

    def test_negative_times_and_horizon(self):
        for name in ('start_timestamp', 'horizon_seconds'):
            for value in (-1, None, float('inf')):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                    trajectory(**{name: value})

    def test_state_timestamp_validation(self):
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            trajectory(predicted_states=[{'timestamp': -1}])

    def test_linkage_and_duplicates(self):
        group = HypothesisSet('s1', [hypothesis()], 10.)
        self.assertEqual(validate_trajectories([trajectory()], group), (trajectory(),))
        with self.assertRaisesRegex(ValueError, 'duplicate trajectory_id'):
            validate_trajectories([trajectory(), trajectory()], group)
        with self.assertRaisesRegex(ValueError, 'unknown hypothesis_id'):
            validate_trajectories([trajectory(hypothesis_id='other')], group)
        with self.assertRaisesRegex(ValueError, 'hypothesis_id'):
            trajectory(hypothesis_id='')

    def test_mismatched_steps(self):
        with self.assertRaisesRegex(ValueError, 'matching lengths'):
            trajectory(uncertainty_by_step=[.1, .2])

    def test_empty_trajectory_allowed(self):
        result = FutureTrajectory('t', 'h', None, 0, 0)
        self.assertEqual(result.predicted_states, ())

    def test_unstructured_states_rejected(self):
        with self.assertRaisesRegex(ValueError, 'predicted_states'):
            trajectory(predicted_states=['unstructured'])


class EpisodeTests(unittest.TestCase):
    def test_complete_chain_representation(self):
        result = episode(target_scene_id='s2', trajectory_id='t1', prediction_id='p1',
                         hypothesis_snapshot=hypothesis(), prediction_summary={'center': [2, 3]},
                         observation_summary={'observed_center': [2, 3]},
                         prediction_timestamp=10., observation_timestamp=11.,
                         evaluation_status='supported', evaluated_timestamp=11.)
        self.assertEqual((result.source_scene_id, result.target_scene_id, result.hypothesis_id,
                          result.trajectory_id, result.prediction_id), ('s1', 's2', 'h1', 't1', 'p1'))
        self.assertEqual(result.hypothesis_snapshot['posterior_probability'], .3)
        self.assertEqual(result.prediction_summary['center'], (2, 3))
        self.assertEqual(result.observation_summary['observed_center'], (2, 3))
        self.assertIsNone(result.prediction_error)

    def test_all_evaluation_statuses_and_optional_error(self):
        for status in EvaluationStatus:
            with self.subTest(status=status):
                self.assertIsNone(episode(evaluation_status=status.value).prediction_error)

    def test_absent_observation_not_contradicted(self):
        self.assertEqual(episode().evaluation_status, EvaluationStatus.PENDING)
        self.assertEqual(episode(evaluation_status='unevaluable').evaluation_status, EvaluationStatus.UNEVALUABLE)

    def test_invalid_evaluation_status(self):
        with self.assertRaisesRegex(ValueError, 'evaluation_status'):
            episode(evaluation_status='incorrect')

    def test_explicit_error_is_only_stored(self):
        error = PredictionError(details={'external_metric': .25})
        result = episode(prediction_error=error)
        error.details['external_metric'] = 99
        self.assertEqual(result.prediction_error['details']['external_metric'], .25)
        self.assertEqual(result.evaluation_status, EvaluationStatus.PENDING)

    def test_snapshot_linkage(self):
        for snapshot in (hypothesis('other'), hypothesis(scene_id='other')):
            with self.assertRaisesRegex(ValueError, 'linkage'):
                episode(hypothesis_snapshot=snapshot)

    def test_episode_timestamp_validation(self):
        for name in ('created_timestamp', 'prediction_timestamp', 'observation_timestamp', 'evaluated_timestamp'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                episode(**{name: -1})


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.memory = ExperienceMemory(clock=lambda: self.now)

    def test_capacity_512_and_fifo_eviction(self):
        for index in range(513):
            self.memory.add(episode(str(index), prediction_id=f'p{index}'))
        self.assertEqual(self.memory.summary()['count'], 512)
        self.assertIsNone(self.memory.get('0'))
        self.assertIsNotNone(self.memory.get('1'))
        self.assertEqual(self.memory.recent(1)[0].episode_id, '512')

    def test_invalid_capacity(self):
        for value in (0, 513, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'capacity'):
                ExperienceMemory(capacity=value)

    def test_ttl_boundary_and_id_reuse(self):
        self.memory.add(episode(prediction_id='p'))
        self.now += 59.999
        self.assertIsNotNone(self.memory.get('e1'))
        self.now = 160.
        self.assertEqual(self.memory.cleanup(), 1)
        self.assertIsNone(self.memory.get('e1'))
        self.memory.add(episode(prediction_id='p'))
        self.assertEqual(self.memory.summary()['count'], 1)

    def test_reads_and_add_cleanup_without_refresh(self):
        for operation in ('get', 'recent', 'summary', 'add'):
            with self.subTest(operation=operation):
                self.now = 100.
                memory = ExperienceMemory(clock=lambda: self.now)
                memory.add(episode())
                self.now = 159.
                memory.get('e1')
                self.now = 160.
                if operation == 'get':
                    self.assertIsNone(memory.get('e1'))
                elif operation == 'recent':
                    self.assertEqual(memory.recent(), ())
                elif operation == 'summary':
                    self.assertEqual(memory.summary()['count'], 0)
                else:
                    memory.add(episode())
                self.assertEqual(memory.summary()['count'], 1 if operation == 'add' else 0)

    def test_duplicate_episode(self):
        self.memory.add(episode())
        with self.assertRaisesRegex(ValueError, 'duplicate episode_id'):
            self.memory.add(episode())

    def test_duplicate_prediction(self):
        self.memory.add(episode(prediction_id='p'))
        with self.assertRaisesRegex(ValueError, 'duplicate prediction_id'):
            self.memory.add(episode('e2', prediction_id='p'))

    def test_null_predictions_not_duplicates(self):
        self.memory.add(episode('e1'))
        self.memory.add(episode('e2'))
        self.assertEqual(self.memory.summary()['count'], 2)

    def test_duplicate_checked_before_eviction(self):
        memory = ExperienceMemory(capacity=1)
        memory.add(episode(prediction_id='p'))
        with self.assertRaisesRegex(ValueError, 'duplicate prediction_id'):
            memory.add(episode('other', prediction_id='p'))
        self.assertIsNotNone(memory.get('e1'))

    def test_eviction_releases_prediction_id(self):
        memory = ExperienceMemory(capacity=1)
        memory.add(episode(prediction_id='p'))
        memory.add(episode('other'))
        memory.add(episode(prediction_id='p'))
        self.assertIsNotNone(memory.get('e1'))

    def test_thread_safe_duplicate_admission(self):
        barrier = Barrier(16)
        def worker(index):
            barrier.wait()
            try:
                self.memory.add(episode(str(index), prediction_id='one'))
                return True
            except ValueError:
                return False
        with ThreadPoolExecutor(max_workers=16) as executor:
            results = list(executor.map(worker, range(16)))
        self.assertEqual(sum(results), 1)
        self.assertEqual(self.memory.summary()['count'], 1)

    def test_concurrent_reads_writes_cleanup_bounded(self):
        def worker(index):
            self.memory.add(episode(str(index), prediction_id=str(index)))
            self.memory.get(str(index))
            self.memory.recent(3)
            self.memory.cleanup()
            return self.memory.summary()['count']
        with ThreadPoolExecutor(max_workers=8) as executor:
            counts = list(executor.map(worker, range(800)))
        self.assertLessEqual(max(counts), 512)
        self.assertEqual(self.memory.summary()['count'], 512)
        self.assertEqual(len({e.prediction_id for e in self.memory.recent()}), 512)

    def test_no_persistence_or_probability_changes(self):
        source = hypothesis()
        result = episode(hypothesis_snapshot=source)
        with patch('builtins.open', side_effect=AssertionError('Unexpected I/O')), \
             patch('sqlite3.connect', side_effect=AssertionError('Unexpected database')):
            self.memory.add(result)
            self.memory.get('e1')
            self.memory.recent()
            self.memory.summary()
            self.memory.cleanup()
            self.assertEqual(ExperienceMemory().recent(), ())
        self.assertEqual(source.posterior_probability, .3)
        self.assertEqual(self.memory.get('e1').hypothesis_snapshot['posterior_probability'], .3)
        self.assertIs(self.memory.get('e1'), result)

    def test_default_clock_is_monotonic(self):
        with patch('fifth_layer.world_model.experience_memory.time.monotonic', return_value=1.) as clock:
            memory = ExperienceMemory()
            memory.add(episode())
            clock.return_value = 61.
            self.assertEqual(memory.cleanup(), 1)

    def test_recent_and_summary_are_detached(self):
        self.memory.add(episode('a'))
        self.memory.add(episode('b', evaluation_status='unevaluable'))
        self.assertEqual([e.episode_id for e in self.memory.recent()], ['b', 'a'])
        self.assertEqual(self.memory.recent(0), ())
        result = self.memory.summary()
        result['by_status'].clear()
        self.assertEqual(self.memory.summary()['by_status'], {'pending': 1, 'unevaluable': 1})
        with self.assertRaisesRegex(ValueError, 'limit'):
            self.memory.recent(-1)


class ImmutabilityValidationTests(unittest.TestCase):
    def test_all_contracts_frozen(self):
        instances = [SceneState('s'), hypothesis(), HypothesisSet('s1', [hypothesis()], 10.),
                     trajectory(), episode()]
        for value in instances:
            with self.subTest(type=type(value)), self.assertRaises(FrozenInstanceError):
                value.provenance = {}

    def test_nested_structures_detached_and_readonly(self):
        data = {'nested': [{'labels': {'one', 'two'}}]}
        objects = [SceneState('s', provenance=data), hypothesis(provenance=data),
                   trajectory(provenance=data), episode(provenance=data)]
        data['nested'][0]['labels'].add('three')
        data['nested'].append({})
        for item in objects:
            with self.subTest(type=type(item)):
                self.assertEqual(item.provenance['nested'][0]['labels'], frozenset({'one', 'two'}))
                self.assertEqual(len(item.provenance['nested']), 1)
                with self.assertRaises(TypeError):
                    item.provenance['nested'][0]['labels'] = ()

    def test_mutable_dataclass_detached(self):
        latent = LatentState({'key': [1, 2]})
        scene = SceneState('s', latent_evidence=latent)
        latent.features['key'].append(3)
        self.assertEqual(scene.latent_evidence['features']['key'], (1, 2))

    def test_direct_visual_payloads_rejected(self):
        for payload in ({'frame': [[1, 2]]}, {'x': np.zeros((2, 2))}, {'x': b'image'},
                        {'x': memoryview(b'image')}, {'x': object()}, {'x': 'a' * 16385}):
            for constructor in (lambda: SceneState('s', provenance=payload),
                                lambda: episode(observation_summary=payload),
                                lambda: hypothesis(evidence_for=[payload]),
                                lambda: trajectory(predicted_states=[payload])):
                with self.subTest(payload=type(payload)), self.assertRaises(ValueError):
                    constructor()

    def test_scene_uncertainty_invalid(self):
        for value in (-.1, 1.1, float('nan')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'uncertainty'):
                SceneState('s', uncertainty=value)

    def test_scene_geometry_and_nested_probabilities(self):
        for record in ({'box': [0, 0, -1, 2]}, {'box_xyxy': [3, 0, 2, 1]},
                       {'center': [1]}, {'confidence': 1.1}, {'timestamp': -1}):
            with self.subTest(record=record), self.assertRaises(ValueError):
                SceneState('s', observed_objects=[record])

    def test_scene_metadata_invalid(self):
        for field, value in (('timestamp', -1), ('image_width', -1), ('snapshot_sequence_id', 1.5)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                SceneState('s', **{field: value})

    def test_direct_scene_rejects_prediction_as_observation(self):
        with self.assertRaisesRegex(ValueError, 'observed_objects'):
            SceneState('s', observed_objects=[{'is_predicted': True}])

    def test_cycles_descriptive_rejection(self):
        cyclic = {}
        cyclic['self'] = cyclic
        with self.assertRaisesRegex(ValueError, 'cyclic'):
            SceneState('s', provenance=cyclic)

    def test_provenance_requires_mapping(self):
        with self.assertRaisesRegex(ValueError, 'provenance'):
            SceneState('s', provenance=['unstructured'])

    def test_raw_confidence_validation_preserves_pixel_uncertainty(self):
        with self.assertRaisesRegex(ValueError, 'raw_confidence'):
            SceneState('s', predicted_tracks=[{'raw_confidence': 2.}])
        scene = SceneState('s', predicted_tracks=[{'raw_confidence': .5,
                           'position_uncertainty': 25., 'position_uncertainty_units': 'pixels'}])
        self.assertEqual(scene.predicted_tracks[0]['position_uncertainty'], 25.)


if __name__ == '__main__':
    unittest.main()
