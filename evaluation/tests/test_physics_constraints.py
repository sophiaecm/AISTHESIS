"""Offline constraint contracts, epistemic boundaries and transition regression."""
from dataclasses import replace, FrozenInstanceError
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from fifth_layer.world_model.physical_state import PhysicalAttribute as A, PhysicalObjectState as O, PhysicalWorldState as W
from fifth_layer.world_model.physics_constraint_engine import PhysicsConstraintEngine, PhysicsConstraintPolicy
from fifth_layer.world_model.physics_constraints import PhysicsConstraintResult
from evaluation.physics_constraints import physical_state_from_dict, process_saved_result


def obj(track=1, x=10, y=10, *, geometry=True, **attrs):
    source = (f'observation.track[{track!r}]',)
    values = {'track_id': A(track, 'observed', source), 'visibility_state': A('observed', 'observed', source),
              'class_name': A('ball', 'observed', source)}
    if geometry:
        values.update(center=A((x+5, y+5), 'observed', source, units='pixels'),
                      bbox=A((x, y, x+10, y+10), 'observed', source, units='pixels'))
    values.update(attrs)
    return O('object-'+str(track), None, values)


def state(t=1, *objects, **kwargs):
    return W('scene-'+str(t), t, tuple(replace(o, timestamp=t) for o in objects), **kwargs)


def assess(current, *history, policy=None):
    return PhysicsConstraintEngine(policy).assess(current, history=history, session_id='test-session', coordinate_frame_id='pixels-test')


def result(assessment, family, finding=None):
    return next(r for r in assessment.constraints.results if r.constraint_type == family and (finding is None or r.finding == finding))


class PhysicsConstraintTests(unittest.TestCase):
    def test_same_track_displacement(self):
        r = result(assess(state(2, obj(x=20)), state(1, obj())), 'kinematic_continuity')
        self.assertEqual(r.measured_values['displacement_pixels'], (10, 0))
        self.assertEqual(r.status, 'satisfied')

    def test_velocity_valid_dt(self):
        r = result(assess(state(3, obj(x=20)), state(1, obj())), 'kinematic_continuity')
        self.assertEqual(r.measured_values['velocity_pixels_per_second'], (5, 0))

    def test_missing_dt_no_velocity(self):
        previous = replace(state(None, obj()), scene_id='old')
        r = result(assess(state(None, obj(x=20)), previous), 'kinematic_continuity')
        self.assertIsNone(r.measured_values['velocity_pixels_per_second'])
        self.assertEqual(r.status, 'indeterminate')

    def test_missing_geometry(self):
        r = result(assess(state(2, obj(geometry=False)), state(1, obj())), 'kinematic_continuity')
        self.assertIsNone(r.measured_values['displacement_pixels'])

    def test_missing_not_zero(self):
        r = result(assess(state(1, obj())), 'inertia_consistency')
        self.assertIsNone(r.measured_values['velocity_change_pixels_per_second'])

    def test_track_change_no_fake_motion(self):
        r = result(assess(state(2, obj(2, x=20)), state(1, obj())), 'kinematic_continuity')
        self.assertIsNone(r.measured_values['displacement_pixels'])

    def test_typed_track_identity(self):
        r = result(assess(state(2, obj('1', x=20)), state(1, obj(1))), 'kinematic_continuity')
        self.assertIsNone(r.measured_values['displacement_pixels'])

    def test_abrupt_change_is_heuristic_only(self):
        r = result(assess(state(3, obj(x=0)), state(1, obj(x=0)), state(2, obj(x=10)),
                          policy=PhysicsConstraintPolicy(max_velocity_change_pixels_per_second=5)), 'inertia_consistency')
        self.assertEqual(r.finding, 'abrupt_change_detected')
        self.assertEqual(r.status, 'violated')
        self.assertEqual(r.measured_values['velocity_change_pixels_per_second'], 20)

    def test_inertia_constant_velocity(self):
        r = result(assess(state(3, obj(x=20)), state(1, obj(x=0)), state(2, obj(x=10)),
                          policy=PhysicsConstraintPolicy(max_velocity_change_pixels_per_second=0)), 'inertia_consistency')
        self.assertEqual(r.finding, 'motion_continuity_consistent')

    def test_no_silent_outlier_threshold(self):
        r = result(assess(state(2, obj(x=10000)), state(1, obj())), 'implausible_displacement')
        self.assertEqual(r.status, 'unsupported')

    def test_explicit_jump_outlier(self):
        r = result(assess(state(2, obj(x=10000)), state(1, obj()), policy=PhysicsConstraintPolicy(max_displacement_pixels=100)), 'implausible_displacement')
        self.assertEqual(r.finding, 'kinematic_outlier')
        self.assertEqual(r.expected_range['threshold_kind'], 'heuristic')

    def test_overlap_is_not_confirmed_collision(self):
        r = result(assess(state(1, obj(), obj(2, x=15))), 'collision_possibility')
        self.assertEqual(r.finding, 'collision_indeterminate')

    def test_approach_overlap_possible_only(self):
        r = result(assess(state(2, obj(), obj(2, x=15)), state(1, obj(), obj(2, x=40))), 'collision_possibility')
        self.assertEqual(r.finding, 'collision_possible')
        self.assertNotIn('collision_observed', json.dumps(r.measured_values.copy()))

    def test_separate_geometry_not_no_collision_fact(self):
        r = result(assess(state(1, obj(), obj(2, x=40))), 'collision_possibility')
        self.assertEqual(r.finding, 'collision_geometry_absent')
        self.assertEqual(r.status, 'not_applicable')

    def test_boundary_exit_possible(self):
        r = result(assess(state(2), state(1, obj(frame_truncation=A(True, 'estimated', ('bbox',), 'boundary')))), 'object_permanence')
        self.assertEqual(r.finding, 'possible_frame_exit')
        self.assertEqual(r.status, 'indeterminate')

    def test_occlusion_remains_possible(self):
        r = result(assess(state(2), state(1, obj(occlusion_state=A('possible', 'possible', ('overlap',), 'geometry')))), 'object_permanence')
        self.assertEqual(r.finding, 'possible_occlusion')

    def test_disappearance_does_not_create_object(self):
        current = state(2)
        a = assess(current, state(1, obj()))
        self.assertEqual(current.objects, ())
        self.assertFalse(a.provenance['new_physical_objects'])
        self.assertEqual(result(a, 'object_permanence').finding, 'unexplained_track_loss')

    def test_reappearance_not_retroactive_occlusion(self):
        r = result(assess(state(3, obj()), state(1, obj()), state(2)), 'object_permanence')
        self.assertEqual(r.finding, 'reobserved_track')
        self.assertIsNone(result(assess(state(3, obj()), state(1, obj()), state(2)), 'kinematic_continuity').measured_values['displacement_pixels'])

    def test_support_geometry_only(self):
        a = assess(state(1, obj(y=0), obj(2, y=10)))
        r = result(a, 'support_stability_possibility', 'support_geometry_possible')
        self.assertEqual(r.status, 'satisfied')
        self.assertIn('depth_contact_and_mechanical_stability_unknown', r.uncertainty)

    def test_no_support_geometry(self):
        a = assess(state(1, obj(y=0), obj(2, x=100, y=10)))
        self.assertEqual(result(a, 'support_stability_possibility').finding, 'support_geometry_not_present')

    def test_semantic_text_ignored(self):
        base = state(1, obj())
        self.assertEqual(assess(base).to_json(), assess(replace(base, provenance={'semantic': 'ball hit a hidden person'})).to_json())

    def test_experience_ignored(self):
        base = state(1, obj())
        self.assertEqual(assess(base).to_json(), assess(replace(base, provenance={'experience': {'collision': True}})).to_json())

    def test_expected_sensory_ignored(self):
        base = state(1, obj())
        self.assertEqual(assess(base).to_json(), assess(replace(base, provenance={'sensory': {'expected_impact': True}})).to_json())

    def test_no_mass(self):
        self.assertNotIn('mass', assess(state(1, obj())).to_json())

    def test_no_force(self):
        self.assertFalse(any('force' in r.measured_values for r in assess(state(1, obj())).constraints.results))

    def test_no_friction(self):
        self.assertNotIn('friction', assess(state(1, obj())).to_json())

    def test_no_depth_value(self):
        self.assertFalse(any('depth' in r.measured_values for r in assess(state(1, obj())).constraints.results))

    def test_no_real_world_speed(self):
        a = assess(state(2, obj(x=20)), state(1, obj()))
        self.assertNotIn('meters', a.to_json())
        self.assertNotIn('m/s', a.to_json())

    def test_future_rejected(self):
        with self.assertRaises(ValueError):
            assess(state(1, obj()), state(2, obj()))

    def test_future_source_provenance_rejected(self):
        with self.assertRaises(ValueError):
            assess(state(1, obj(), provenance={'source_timestamp': 2}))

    def test_duplicate_time_rejected(self):
        with self.assertRaises(ValueError):
            assess(state(1, obj()), replace(state(1, obj()), scene_id='other'))

    def test_deterministic(self):
        a = assess(state(2, obj()), state(1, obj()))
        b = assess(state(2, obj()), state(1, obj()))
        self.assertEqual(a.to_json(), b.to_json())
        self.assertEqual(a.assessment_id, b.assessment_id)

    def test_stable_ordering(self):
        a = assess(state(1, obj(), obj(2, x=30)))
        b = assess(state(1, obj(2, x=30), obj()))
        self.assertEqual(a.to_json(), b.to_json())

    def test_empty(self):
        self.assertEqual(assess(state()).constraints.results, ())

    def test_multiple_tracks(self):
        a = assess(state(2, obj(x=20), obj(2, x=50)), state(1, obj(), obj(2, x=30)))
        self.assertEqual(sum(r.constraint_type == 'kinematic_continuity' for r in a.constraints.results), 2)

    def test_missing_provenance_is_indeterminate(self):
        a = assess(state(1, O('unattributed', None, {})))
        self.assertEqual(result(a, 'kinematic_continuity').status, 'indeterminate')

    def test_all_results_have_provenance(self):
        a = assess(state(2, obj(), obj(2)), state(1, obj(), obj(2)))
        self.assertTrue(all(r.provenance and r.derived_from for r in a.constraints.results))

    def test_result_rejects_missing_provenance(self):
        with self.assertRaises(ValueError):
            PhysicsConstraintResult('x', 'x', 's', 1, (), 'indeterminate', 'unknown', 'missing')

    def test_models_immutable(self):
        a = assess(state(1, obj()))
        with self.assertRaises(FrozenInstanceError):
            a.timestamp = 3
        with self.assertRaises(TypeError):
            a.constraints.results[0].provenance['policy']['support_gap_pixels'] = 2

    def test_possible_geometry_not_observed_geometry(self):
        p = obj(center=A((1, 1), 'possible', ('semantic',), 'possible center', 'pixels'))
        self.assertIsNone(result(assess(state(2, p), state(1, obj())), 'kinematic_continuity').measured_values['displacement_pixels'])

    def test_wrong_units_not_used(self):
        p = obj(center=A((1, 1), 'observed', ('sensor',), units='meters'))
        self.assertIsNone(result(assess(state(2, p), state(1, obj())), 'kinematic_continuity').measured_values['displacement_pixels'])

    def test_round_trip_v01(self):
        s = state(1, obj())
        self.assertEqual(physical_state_from_dict(s.to_dict()).to_json(), s.to_json())

    def test_saved_report_offline(self):
        data = self.saved()
        a = process_saved_result(data)
        self.assertEqual(a, process_saved_result(data))
        self.assertEqual(a['sessions'][0]['summary']['snapshot_count'], 2)

    def test_saved_future_order_rejected(self):
        data = self.saved()
        data['sessions'][0]['snapshots'].reverse()
        with self.assertRaises(ValueError):
            process_saved_result(data)

    def test_cli_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.json'
            target = Path(directory)/'test_physics_constraints.json'
            source.write_text(json.dumps(self.saved()), encoding='utf-8')
            command = [sys.executable, '-m', 'evaluation.physics_constraints', str(source), '--output', str(target)]
            subprocess.run(command, check=True, capture_output=True)
            old = target.read_bytes()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(old, target.read_bytes())

    @staticmethod
    def saved():
        return {'schema_version': 'physical-world-evaluation-0.1', 'sessions': [
            {'mode': 'FIFTH_LAYER_ONLY', 'source_sha256': 'test',
             'snapshots': [state(1, obj()).to_dict(), state(2, obj(x=20)).to_dict()]}]}


if __name__ == '__main__':
    unittest.main()
