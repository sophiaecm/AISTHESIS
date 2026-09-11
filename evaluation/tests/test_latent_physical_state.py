"""Non-neural latent summaries: temporal, epistemic and offline boundaries."""
from dataclasses import replace, FrozenInstanceError
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from fifth_layer.world_model.physical_state import PhysicalAttribute as A, PhysicalObjectState as O, PhysicalWorldState as W, PhysicalRelation
from fifth_layer.world_model.physics_constraints import PhysicsConstraintResult as R, PhysicsConstraintBundle as B, PhysicsTransitionAssessment
from fifth_layer.world_model.latent_physical_state_builder import LatentPhysicalStateBuilder
from evaluation.latent_physical_state import process_saved_results


def obj(identity='ball', **attributes):
    values = {'class_name': A(identity, 'observed', ('detector.label',)),
              'track_id': A(identity, 'observed', ('tracker.id',)),
              'visibility_state': A('observed', 'observed', ('detector',))}
    values.update(attributes)
    return O(identity, None, values)


def state(t=1, objects=None, **kwargs):
    return W('scene'+str(t), t, tuple(replace(o, timestamp=t) for o in ((obj(),) if objects is None else objects)), **kwargs)


def build(s=None, constraints=None, history=()):
    return LatentPhysicalStateBuilder().build(s or state(), constraints, history=history,
                                              session_id='test', coordinate_frame_id='pixels')


def constraint(s, family='collision_possibility', finding='collision_possible', status='satisfied', ids=('ball', 'car'), **kwargs):
    return R('constraint-'+family, family, s.scene_id, s.timestamp, ids, status, finding, 'source explanation',
             derived_from=('physical.measurement',), provenance={'session_id': 'test', 'coordinate_frame_id': 'pixels'}, **kwargs)


class LatentPhysicalStateTests(unittest.TestCase):
    def test_deterministic_construction(self):
        self.assertEqual(build().to_json(), build().to_json())

    def test_future_history_rejected(self):
        with self.assertRaises(ValueError):
            build(state(1), history=(build(state(2)),))

    def test_duplicate_time_rejected(self):
        with self.assertRaises(ValueError):
            build(replace(state(1), scene_id='other'), history=(build(state(1)),))

    def test_reordered_history_rejected(self):
        with self.assertRaises(ValueError):
            build(state(3), history=(build(state(2)), build(state(1))))

    def test_undated_history_rejected(self):
        with self.assertRaises(ValueError):
            build(state(2), history=(build(state(None)),))

    def test_undated_single_snapshot(self):
        self.assertIsNone(build(state(None)).timestamp)

    def test_missing_geometry_unknown(self):
        a = build().objects[0]['attributes']['center']
        self.assertIsNone(a['value'])
        self.assertEqual(a['status'], 'unknown')

    def test_missing_value_not_zero(self):
        self.assertIsNone(build().objects[0]['attributes']['velocity']['value'])

    def test_displacement_pixels(self):
        s = state(objects=(obj(displacement=A((3, 0), 'estimated', ('motion',), 'delta', 'pixels')),))
        self.assertEqual(build(s).objects[0]['attributes']['displacement']['units'], 'pixels')

    def test_velocity_pixels_per_second(self):
        s = state(objects=(obj(velocity=A((3, 0), 'estimated', ('motion',), 'delta/dt', 'pixels/second')),))
        a = build(s).objects[0]['attributes']['velocity']
        self.assertEqual(a['units'], 'pixels/second')
        self.assertEqual(a['status'], 'estimated')

    def test_metric_units_not_promoted(self):
        s = state(objects=(obj(velocity=A((3, 0), 'observed', ('sensor',), units='m/s')),))
        self.assertIsNone(build(s).objects[0]['attributes']['velocity']['value'])
        self.assertNotIn('m/s', build(s).to_json())

    def test_no_mass(self):
        self.assertNotIn('mass', build(state(objects=(obj(mass=A(2, 'observed', ('external',))),))).objects[0]['attributes'])

    def test_no_force(self):
        self.assertNotIn('force', build().objects[0]['attributes'])

    def test_no_friction(self):
        self.assertNotIn('friction', build().objects[0]['attributes'])

    def test_no_exact_depth(self):
        self.assertNotIn('depth', build().objects[0]['attributes'])
        self.assertIn('depth_unknown', build().uncertainty)

    def test_no_hidden_actor(self):
        self.assertEqual(len(build().objects), 1)
        self.assertEqual(build().objects[0]['physical_object_id'], 'ball')

    def test_overlap_not_collision(self):
        s = state(objects=(obj(), obj('car')), relations=(PhysicalRelation('ball', 'overlaps', 'car', A(True, 'estimated', ('bbox',), 'intersection')),))
        self.assertEqual(build(s).relations[0]['signal'], 'overlaps')
        self.assertNotIn('collision_risk_geometry', [r['signal'] for r in build(s).relations])

    def test_collision_remains_possible(self):
        s = state(objects=(obj(), obj('car')))
        latent = build(s, B(s.scene_id, s.timestamp, (constraint(s),)))
        self.assertEqual(latent.relations[0]['status'], 'possible')
        self.assertEqual(latent.relations[0]['signal'], 'collision_risk_geometry')
        self.assertEqual(latent.active_constraints[0]['status'], 'satisfied')

    def test_support_remains_possible(self):
        s = state(objects=(obj(), obj('car')))
        r = constraint(s, 'support_stability_possibility', 'support_geometry_possible',
                       measured_values={'upper_object_id': 'ball', 'lower_object_id': 'car'})
        latent = build(s, B(s.scene_id, s.timestamp, (r,)))
        self.assertEqual(latent.relations[0]['signal'], 'support_candidate')
        self.assertEqual(latent.relations[0]['subject_id'], 'ball')
        self.assertEqual(latent.relations[0]['status'], 'possible')

    def test_permanence_no_unseen_object(self):
        s = state(2, objects=())
        r = constraint(s, 'object_permanence', 'unexplained_track_loss', 'indeterminate', ('ball',))
        latent = build(s, B(s.scene_id, s.timestamp, (r,)), (build(),))
        self.assertEqual(latent.objects, ())
        self.assertEqual(latent.dynamics[0]['status'], 'possible')

    def test_frame_exit_possible(self):
        s = state(2, objects=())
        r = constraint(s, 'object_permanence', 'possible_frame_exit', 'indeterminate', ('ball',))
        latent = build(s, B(s.scene_id, s.timestamp, (r,)), (build(),))
        self.assertEqual(latent.dynamics[0]['signal'], 'boundary_exit')
        self.assertEqual(latent.dynamics[0]['status'], 'possible')

    def test_occlusion_possible(self):
        s = state(objects=(obj(occlusion_state=A('possible', 'possible', ('bbox',), 'overlap')),))
        self.assertEqual(build(s).objects[0]['attributes']['occlusion_state']['status'], 'possible')

    def test_semantic_shortcut_blocked(self):
        s = state(provenance={'semantic': 'a person will collide'})
        self.assertEqual(build(s).relations, ())
        self.assertEqual(len(build(s).objects), 1)

    def test_experience_shortcut_blocked(self):
        self.assertEqual(build(state(provenance={'experience': {'collision': True}})).relations, ())

    def test_expected_sensory_shortcut_blocked(self):
        self.assertEqual(build(state(provenance={'expected_sensory': {'impact': True}})).relations, ())

    def test_provenance_preserved(self):
        refs = build().objects[0]['attributes']['class_name']['derived_from']
        self.assertIn('detector.label', refs)
        self.assertTrue(any('attributes.class_name' in r for r in refs))

    def test_uncertainty_preserved(self):
        s = state(uncertainty={'calibration': 'unavailable'})
        self.assertEqual(build(s).provenance['source_uncertainty']['calibration'], 'unavailable')
        self.assertIn('actual_contact_unknown', build(s).uncertainty)

    def test_raw_constraints_not_copied(self):
        s = state(objects=(obj(), obj('car')))
        latent = build(s, B(s.scene_id, s.timestamp, (constraint(s, measured_values={'large_raw_record': [1, 2, 3]}),)))
        self.assertNotIn('large_raw_record', latent.to_json())
        self.assertNotIn('explanation', latent.active_constraints[0])
        self.assertIn('constraint_id', latent.active_constraints[0])

    def test_stable_serialization(self):
        latent = build()
        self.assertEqual(json.loads(latent.to_json()), latent.to_dict())

    def test_feature_mask_missing(self):
        feature = build().to_feature_dict()['objects'][0]['numeric']['velocity']
        self.assertEqual(feature['validity_mask'], [False, False])
        self.assertIsNone(feature['value'])

    def test_feature_mask_valid_zero(self):
        s = state(objects=(obj(velocity=A((0, 0), 'estimated', ('motion',), 'delta/dt', 'pixels/second')),))
        feature = build(s).to_feature_dict()['objects'][0]['numeric']['velocity']
        self.assertEqual(feature['validity_mask'], [True, True])
        self.assertEqual(feature['value'], [0, 0])

    def test_possible_geometry_mask_invalid(self):
        s = state(objects=(obj(center=A((1, 2), 'possible', ('source',), 'possibility', 'pixels')),))
        feature = build(s).to_feature_dict()['objects'][0]['numeric']['center']
        self.assertEqual(feature['validity_mask'], [False, False])
        self.assertEqual(feature['status'], 'possible')

    def test_wrong_scene_constraints(self):
        with self.assertRaises(ValueError):
            build(state(), B('other', 1))

    def test_future_constraint_timestamp(self):
        with self.assertRaises(ValueError):
            build(state(), B('scene1', 2))

    def test_future_source_timestamp(self):
        with self.assertRaises(ValueError):
            build(state(provenance={'source_timestamp': 2}))

    def test_cross_coordinate_history(self):
        with self.assertRaises(ValueError):
            build(state(2), history=(replace(build(), coordinate_frame_id='other'),))

    def test_unobserved_constraint_actor_rejected(self):
        s = state()
        with self.assertRaises(ValueError):
            build(s, B(s.scene_id, s.timestamp, (constraint(s),)))

    def test_all_constraint_statuses_retained(self):
        s = state()
        for status in ('satisfied', 'violated', 'indeterminate', 'unsupported', 'not_applicable'):
            r = constraint(s, 'kinematic_continuity', 'unknown', status, ('ball',))
            self.assertEqual(build(s, B(s.scene_id, s.timestamp, (r,))).active_constraints[0]['status'], status)

    def test_outlier_possible_discontinuity(self):
        s = state()
        r = constraint(s, 'implausible_displacement', 'kinematic_outlier', 'violated', ('ball',))
        d = build(s, B(s.scene_id, s.timestamp, (r,))).dynamics[0]
        self.assertEqual(d['signal'], 'tracking_or_motion_discontinuity')
        self.assertEqual(d['status'], 'possible')

    def test_empty_scene(self):
        self.assertEqual(build(state(objects=())).to_feature_dict()['counts']['object_count'], 0)

    def test_ordering(self):
        self.assertEqual(build(state(objects=(obj(), obj('car')))).to_json(),
                         build(state(objects=(obj('car'), obj()))).to_json())

    def test_immutable(self):
        latent = build()
        with self.assertRaises(FrozenInstanceError):
            latent.timestamp = 2
        with self.assertRaises(TypeError):
            latent.objects[0]['attributes']['class_name']['value'] = 'person'

    def test_estimated_class_not_observed(self):
        s = state(objects=(obj(class_name=A('person', 'estimated', ('text',), 'guess')),))
        self.assertIsNone(build(s).objects[0]['attributes']['class_name']['value'])

    def test_no_models_in_offline_transform(self):
        p, c, ph, ch = self.fixture()
        with (patch('fifth_layer.world_model.physics_constraint_engine.PhysicsConstraintEngine.assess', side_effect=AssertionError('rerun')),
              patch('fifth_layer.world_model.physical_state_builder.PhysicalStateBuilder.build', side_effect=AssertionError('rerun'))):
            report = process_saved_results(p, c, physical_source_sha256=ph, constraint_source_sha256=ch)
        self.assertEqual(report['sessions'][0]['summary']['snapshot_count'], 1)

    def test_source_hash_mismatch(self):
        p, c, ph, ch = self.fixture()
        with self.assertRaises(ValueError):
            process_saved_results(p, c, physical_source_sha256='0'*64, constraint_source_sha256=ch)

    def test_assessment_alignment_rejected(self):
        s = state()
        a = PhysicsTransitionAssessment('a', s.scene_id, s.timestamp, None, None, B(s.scene_id, s.timestamp),
            {'session_id': 'test', 'coordinate_frame_id': 'pixels', 'history_scene_ids': ()})
        with self.assertRaises(ValueError):
            build(state(2), a, (build(),))

    def test_constraint_cross_session_rejected(self):
        s = state(objects=(obj(), obj('car')))
        r = replace(constraint(s), provenance={'session_id': 'other', 'coordinate_frame_id': 'pixels'})
        with self.assertRaises(ValueError):
            build(s, B(s.scene_id, s.timestamp, (r,)))

    def test_semantic_family_cannot_map_collision(self):
        s = state(objects=(obj(), obj('car')))
        r = constraint(s, 'semantic', 'collision_possible')
        self.assertEqual(build(s, B(s.scene_id, s.timestamp, (r,))).relations, ())

    def test_constraint_uncertainty_on_object(self):
        s = state(objects=(obj(), obj('car')))
        r = constraint(s, uncertainty=('camera_motion_unresolved',))
        latent = build(s, B(s.scene_id, s.timestamp, (r,)))
        self.assertIn('camera_motion_unresolved', latent.objects[0]['uncertainty'])

    def test_offline_determinism(self):
        p, c, ph, ch = self.fixture()
        a = process_saved_results(p, c, physical_source_sha256=ph, constraint_source_sha256=ch)
        self.assertEqual(a, process_saved_results(p, c, physical_source_sha256=ph, constraint_source_sha256=ch))

    def test_offline_session_mismatch(self):
        p, c, ph, ch = self.fixture()
        c['sessions'][0]['source_sha256'] = 'other'
        with self.assertRaises(ValueError):
            process_saved_results(p, c, physical_source_sha256=ph, constraint_source_sha256=ch)

    def test_cli_and_overwrite(self):
        p, c, ph, ch = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            physical, constraints, target = root/'physical.json', root/'constraints.json', root/'test_latent_physical_state.json'
            physical.write_bytes(json.dumps(p).encode())
            constraints.write_bytes(json.dumps(c).encode())
            cmd = [sys.executable, '-m', 'evaluation.latent_physical_state', str(physical), str(constraints), '--output', str(target)]
            subprocess.run(cmd, check=True, capture_output=True)
            before = target.read_bytes()
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(before, target.read_bytes())

    @staticmethod
    def fixture():
        s = state()
        p = {'schema_version': 'physical-world-evaluation-0.1', 'sessions': [
            {'mode': 'TEST', 'source_sha256': 'video', 'snapshots': [s.to_dict()]}]}
        ph = sha256(json.dumps(p).encode()).hexdigest()
        assessment = PhysicsTransitionAssessment('a', s.scene_id, s.timestamp, None, None, B(s.scene_id, s.timestamp),
            {'session_id': 'TEST:video', 'coordinate_frame_id': 'video:source-pixels', 'history_scene_ids': ()})
        c = {'schema_version': 'physics-constraints-evaluation-0.2', 'source_file_sha256': ph,
             'sessions': [{'mode': 'TEST', 'source_sha256': 'video', 'assessments': [assessment.to_dict()]}]}
        return p, c, ph, sha256(json.dumps(c).encode()).hexdigest()


if __name__ == '__main__':
    unittest.main()
