"""Read-only compression and temporal alignment of frozen physical contracts."""
from collections.abc import Mapping
import json

from ._structured import geometry, identifier, number
from .evidence import stable_id
from .physical_state import PhysicalWorldState
from .physics_constraints import PhysicsConstraintBundle, PhysicsTransitionAssessment
from .latent_physical_state import LatentPhysicalState


FIELDS = ('track_id', 'class_name', 'center', 'bbox', 'size', 'frame_truncation',
          'displacement', 'velocity', 'motion_state', 'motion_direction', 'visibility_state', 'occlusion_state')
UNITS = {'center': 'pixels', 'bbox': 'pixels', 'size': 'pixels', 'displacement': 'pixels', 'velocity': 'pixels/second'}
UNCERTAINTY = ('depth_unknown', 'metric_scale_unknown', 'actual_contact_unknown', 'events_between_snapshots_unknown',
               'tracking_identity_uncertain', 'occlusion_unresolved', 'support_stability_unknown', 'collision_not_confirmed',
               'hidden_state_unknown')
# Exact family/finding/status matching prevents arbitrary semantic labels becoming signals.
DYNAMICS = {
    ('implausible_displacement', 'kinematic_outlier', 'violated'): ('tracking_or_motion_discontinuity', 'possible'),
    ('inertia_consistency', 'abrupt_change_detected', 'violated'): ('tracking_or_motion_discontinuity', 'possible'),
    ('inertia_consistency', 'motion_continuity_consistent', 'satisfied'): ('sampled_motion_continuity', 'estimated'),
    ('object_permanence', 'possible_frame_exit', 'indeterminate'): ('boundary_exit', 'possible'),
    ('object_permanence', 'possible_occlusion', 'indeterminate'): ('occlusion', 'possible'),
    ('object_permanence', 'unexplained_track_loss', 'indeterminate'): ('disappearance', 'possible'),
    ('object_permanence', 'reobserved_track', 'satisfied'): ('reobserved_object', 'estimated'),
    ('object_permanence', 'object_permanence_consistent', 'satisfied'): ('permanence_compatibility', 'estimated'),
}


def _times(value, cutoff):
    if isinstance(value, Mapping):
        for key, child in value.items():
            if (key == 'timestamp' or key.endswith('_timestamp')) and child is not None:
                number(child, key, nonnegative=True)
                if cutoff is None or child > cutoff:
                    raise ValueError('future or unorderable source timestamp')
            elif isinstance(child, (Mapping, tuple, list)):
                _times(child, cutoff)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _times(child, cutoff)


class LatentPhysicalStateBuilder:
    def build(self, physical, constraints=None, *, history=(), session_id, coordinate_frame_id):
        identifier(session_id, 'session_id')
        identifier(coordinate_frame_id, 'coordinate_frame_id')
        if not isinstance(physical, PhysicalWorldState) or physical.schema_version != 'physical-world-0.1':
            raise ValueError('expected PhysicalWorldState v0.1')
        history = tuple(history)
        if any(not isinstance(h, LatentPhysicalState) or h.schema_version != 'latent-physical-state-0.1' for h in history):
            raise ValueError('history must contain latent v0.1 states')
        if any(h.session_id != session_id or h.coordinate_frame_id != coordinate_frame_id for h in history):
            raise ValueError('cross-session or coordinate-frame history')
        if len({h.scene_id for h in history} | {physical.scene_id}) != len(history) + 1:
            raise ValueError('duplicate history scene')
        if history:
            stamps = [h.timestamp for h in history] + [physical.timestamp]
            if any(t is None for t in stamps) or any(a >= b for a, b in zip(stamps, stamps[1:])):
                raise ValueError('history requires strictly increasing known timestamps')
        _times(physical.provenance, physical.timestamp)
        for obj in physical.objects:
            _times(obj.provenance, physical.timestamp)
        assessment_id = None
        if isinstance(constraints, PhysicsTransitionAssessment):
            if constraints.schema_version != 'physics-transition-0.2':
                raise ValueError('unsupported assessment schema')
            if constraints.scene_id != physical.scene_id or constraints.timestamp != physical.timestamp:
                raise ValueError('assessment must align to current physical scene/time')
            expected_previous = (history[-1].scene_id, history[-1].timestamp) if history else (None, None)
            if (constraints.previous_scene_id, constraints.previous_timestamp) != expected_previous:
                raise ValueError('assessment previous scene/time must match supplied history')
            if tuple(constraints.provenance.get('history_scene_ids', ())) != tuple(h.scene_id for h in history):
                raise ValueError('assessment history does not match supplied history')
            if constraints.provenance.get('session_id') != session_id or constraints.provenance.get('coordinate_frame_id') != coordinate_frame_id:
                raise ValueError('assessment session or coordinate frame mismatch')
            _times(constraints.provenance, physical.timestamp)
            assessment_id = constraints.assessment_id
            constraints = constraints.constraints
        if constraints is not None and (not isinstance(constraints, PhysicsConstraintBundle)
                or constraints.schema_version != 'physics-constraints-0.2'
                or constraints.scene_id != physical.scene_id or constraints.timestamp != physical.timestamp):
            raise ValueError('constraint bundle must match current physical scene/time')
        current_ids = {o.object_id for o in physical.objects}
        historical_ids = {o['physical_object_id'] for h in history for o in h.objects}
        results = constraints.results if constraints is not None else ()
        for r in results:
            if r.rule_version != 'physics-constraints-0.2':
                raise ValueError('unsupported constraint rule version')
            if not set(r.object_ids) <= current_ids | historical_ids:
                raise ValueError('constraint references unobserved object IDs')
            if r.provenance.get('session_id') != session_id or r.provenance.get('coordinate_frame_id') != coordinate_frame_id:
                raise ValueError('constraint session or coordinate frame mismatch')
            _times(r.provenance, physical.timestamp)
            _times(r.measured_values, physical.timestamp)

        objects = []
        uncertainties = set(UNCERTAINTY)
        for obj in physical.objects:
            attrs = {}
            unknown = set(UNCERTAINTY)
            for name in FIELDS:
                attr = obj.attributes.get(name)
                path = f'{physical.scene_id}.objects[{obj.object_id!r}].attributes.{name}'
                value, status, units = (attr.value, attr.status, attr.units) if attr else (None, 'unknown', UNITS.get(name))
                refs = tuple(sorted(set((path,) + (attr.derived_from if attr else ()))))
                if name == 'class_name' and status != 'observed':
                    value, status = None, 'unknown'
                if name in UNITS and value is not None:
                    if units != UNITS[name]:
                        value, status, units = None, 'unknown', UNITS[name]
                        unknown.add(name + '_unsupported_units')
                    else:
                        geometry(value, name, 4 if name == 'bbox' else 2)
                if value is None:
                    unknown.add(name + '_' + status)
                attrs[name] = {'value': value, 'status': status, 'units': units, 'derived_from': refs}
            objects.append({'physical_object_id': obj.object_id, 'attributes': attrs, 'uncertainty': tuple(sorted(unknown))})
            uncertainties.update(unknown)

        relations = []
        for r in physical.relations:
            # Preserve image-plane predicates; contact/support remain possibilities.
            signal = {'contact_possible': 'contact_candidate', 'support_possible': 'support_candidate'}.get(r.relation, r.relation)
            status = 'possible' if r.relation in ('contact_possible', 'support_possible') else r.evidence.status
            relations.append({'subject_id': r.subject_id, 'object_id': r.object_id, 'signal': signal,
                'status': status, 'value': r.evidence.value,
                'derived_from': tuple(sorted(set(r.evidence.derived_from +
                    (f'{physical.scene_id}.relations[{r.subject_id!r},{r.relation!r},{r.object_id!r}]',))))})
        active = []
        dynamics = []
        for r in results:
            active.append({'constraint_id': r.constraint_id, 'constraint_type': r.constraint_type,
                'status': r.status, 'finding': r.finding, 'object_ids': r.object_ids})
            uncertainties.update(r.uncertainty)
            for obj in objects:
                if obj['physical_object_id'] in r.object_ids:
                    obj['uncertainty'] = tuple(sorted(set(obj['uncertainty']) | set(r.uncertainty)))
            key = (r.constraint_type, r.finding, r.status)
            if key in DYNAMICS:
                signal, status = DYNAMICS[key]
                dynamics.append({'signal': signal, 'status': status, 'object_ids': r.object_ids, 'derived_from': (r.constraint_id,)})
            if key == ('collision_possibility', 'collision_possible', 'satisfied'):
                if len(r.object_ids) != 2 or not set(r.object_ids) <= current_ids:
                    raise ValueError('collision possibility requires two current endpoints')
                relations.append({'subject_id': r.object_ids[0], 'object_id': r.object_ids[1],
                    'signal': 'collision_risk_geometry', 'status': 'possible', 'value': 'possible', 'derived_from': (r.constraint_id,)})
            if key == ('support_stability_possibility', 'support_geometry_possible', 'satisfied'):
                upper, lower = r.measured_values.get('upper_object_id'), r.measured_values.get('lower_object_id')
                if upper == lower or {upper, lower} != set(r.object_ids) or not {upper, lower} <= current_ids:
                    raise ValueError('support possibility requires explicit current directional endpoints')
                relations.append({'subject_id': upper, 'object_id': lower, 'signal': 'support_candidate',
                    'status': 'possible', 'value': 'possible', 'derived_from': (r.constraint_id,)})
        if history:
            for oid in sorted(current_ids - historical_ids):
                dynamics.append({'signal': 'newly_observed_object', 'status': 'observed', 'object_ids': (oid,),
                    'derived_from': (physical.scene_id + '.objects', history[-1].latent_state_id + '.objects')})
        else:
            uncertainties.add('prior_observation_history_unavailable')
        if constraints is None:
            uncertainties.add('constraint_evidence_unavailable')

        def ordered(records):
            # Stable deduplication of identical summaries, never probability pooling.
            return tuple(v for _, v in sorted({json.dumps(v, sort_keys=True, separators=(',', ':')): v for v in records}.items()))

        provenance = {'physical_scene_ref': physical.scene_id, 'physics_assessment_ref': assessment_id,
            'previous_latent_state_ref': history[-1].latent_state_id if history else None,
            'source_uncertainty_ref': physical.scene_id + '.uncertainty',
            'source_uncertainty': physical.uncertainty,
            'context_evidence_refs': tuple(sorted(set(physical.evidence_references))),
            'role': 'structured_non_neural_summary_not_new_physical_measurement'}
        values = dict(scene_id=physical.scene_id, timestamp=physical.timestamp, session_id=session_id,
            coordinate_frame_id=coordinate_frame_id, objects=tuple(sorted(objects, key=lambda o: o['physical_object_id'])),
            relations=ordered(relations), dynamics=ordered(dynamics), active_constraints=ordered(active),
            uncertainty=tuple(sorted(uncertainties)), provenance=provenance)
        return LatentPhysicalState(stable_id('latent-physical', values), **values)
