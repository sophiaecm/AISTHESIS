"""Read-only normalization into the existing EvidenceItem/EvidenceBundle API."""
from .evidence import EvidenceItem, EvidenceBundle, stable_id
from .latent_physical_state import LatentPhysicalState
from .physics_constraints import PhysicsConstraintBundle, PhysicsTransitionAssessment
from .hybrid_world_state import HybridWorldState, LearnedRepresentationSignal
from .common_evidence_state import CommonEvidenceState, FRAME_BOUND, bounded_plain, check_context


class CommonEvidenceStateBuilder:
    def build(self, *, scene_id, session_id, timestamp, coordinate_frame_id=None,
              evidence=(), hybrid_state=None, physical_state=None,
              physics_constraints=None, learned_signal=None, present_families=()):
        context = dict(scene_id=scene_id, session_id=session_id, timestamp=timestamp,
                       coordinate_frame_id=coordinate_frame_id)
        items, provenance = [], {}
        present = set(present_families)
        if isinstance(evidence, (EvidenceItem, EvidenceBundle)):
            evidence = (evidence,)
        if not isinstance(evidence, (tuple, list)):
            raise ValueError('evidence must be an item, bundle, or ordered sequence')
        for source in evidence:
            if isinstance(source, EvidenceItem):
                items.append(source)
            elif isinstance(source, EvidenceBundle):
                if source.scene_id != scene_id:
                    raise ValueError('bundle scene mismatch')
                data = bounded_plain(source)
                check_context(data['provenance'], **context, source_time=timestamp,
                              frame_bound=any(i.source_type.value in FRAME_BOUND for i in source.items))
                provenance[stable_id('bundle', data)] = data['provenance']
                items.extend(source.items)
            else:
                raise ValueError('expected EvidenceItem or EvidenceBundle')
        if hybrid_state is not None:
            if not isinstance(hybrid_state, HybridWorldState):
                raise ValueError('expected HybridWorldState')
            if any(s is not None for s in (physical_state, physics_constraints, learned_signal)):
                raise ValueError('supply hybrid or standalone components, not both')
            physical_state = hybrid_state.physical_state
            physics_constraints = hybrid_state.physics_constraints
            learned_signal = hybrid_state.learned_signal
            provenance['hybrid'] = bounded_plain({
                'source_references': hybrid_state.source_references,
                'provenance': hybrid_state.provenance, 'uncertainty': hybrid_state.uncertainty,
                'availability': hybrid_state.availability})
        if physical_state is not None:
            if not isinstance(physical_state, LatentPhysicalState) or physical_state.schema_version != 'latent-physical-state-0.1':
                raise ValueError('expected LatentPhysicalState v0.1')
            data = bounded_plain(physical_state)
            check_context(data, **context, source_time=physical_state.timestamp, frame_bound=True)
            items.append(EvidenceItem(physical_state.latent_state_id, physical_state.scene_id,
                'physical', 'LatentPhysicalState', 'structured_physical_summary',
                data, physical_state.timestamp, epistemic_status='mixed',
                provenance={'session_id': physical_state.session_id,
                    'coordinate_frame_id': physical_state.coordinate_frame_id,
                    'source_provenance': data['provenance']}))
        if physics_constraints is not None:
            assessment = physics_constraints if isinstance(physics_constraints, PhysicsTransitionAssessment) else None
            bundle = assessment.constraints if assessment is not None else physics_constraints
            if not isinstance(bundle, PhysicsConstraintBundle) or bundle.schema_version != 'physics-constraints-0.2':
                raise ValueError('expected physics bundle/assessment v0.2')
            if assessment is not None and assessment.schema_version != 'physics-transition-0.2':
                raise ValueError('unsupported assessment version')
            check_context(bounded_plain(physics_constraints), **context, source_time=bundle.timestamp, frame_bound=True)
            present.add('physics_constraint')
            provenance['physics_constraints'] = bounded_plain({
                'scene_id': bundle.scene_id, 'timestamp': bundle.timestamp,
                'assessment_id': None if assessment is None else assessment.assessment_id,
                'previous_scene_id': None if assessment is None else assessment.previous_scene_id,
                'previous_timestamp': None if assessment is None else assessment.previous_timestamp,
                'assessment_kind': None if assessment is None else assessment.assessment_kind,
                'provenance': None if assessment is None else assessment.provenance})
            for result in bundle.results:
                if result.rule_version != 'physics-constraints-0.2':
                    raise ValueError('unsupported constraint rule version')
                items.append(EvidenceItem(result.constraint_id, result.scene_id, 'physics_constraint',
                    'PhysicsConstraintEngine', result.constraint_type, bounded_plain(result),
                    result.timestamp, epistemic_status='assessment', provenance=result.provenance))
        if learned_signal is not None:
            if not isinstance(learned_signal, LearnedRepresentationSignal):
                raise ValueError('expected LearnedRepresentationSignal')
            if timestamp is None or learned_signal.timestamp != timestamp:
                raise ValueError('learned timestamp must match known integration timestamp')
            data = bounded_plain(learned_signal)
            check_context(data, **context, source_time=learned_signal.timestamp, frame_bound=True)
            items.append(EvidenceItem(learned_signal.representation_id, learned_signal.scene_id,
                'learned_representation', learned_signal.source_model, learned_signal.source_type,
                data, learned_signal.timestamp, epistemic_status='learned_signal',
                provenance={'session_id': learned_signal.session_id,
                    'coordinate_frame_id': learned_signal.coordinate_frame_id,
                    'source_provenance': data['provenance']}))
        return CommonEvidenceState(**context, evidence=EvidenceBundle(scene_id, tuple(items)),
                                   present_families=tuple(sorted(present)), provenance=provenance)
