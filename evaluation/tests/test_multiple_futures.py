"""Synthetic Step 18 contract/epistemic tests, not predictive validation."""
from dataclasses import FrozenInstanceError, replace
import json
import os
import subprocess
import sys

import pytest

from fifth_layer.world_model.multiple_futures import (
    FutureBranchStatus, FutureHorizonKind,
    FutureRelation, MultipleFuturesBuilder, VERSION)
from fifth_layer.world_model.cross_modal_consequences import CrossModalConsequenceBuilder
from fifth_layer.world_model.hybrid_world_state import HybridWorldState, LearnedRepresentationSignal
from fifth_layer.world_model.physics_constraints import PhysicsConstraintBundle as B
from fifth_layer.world_model.physics_constraint_engine import PhysicsConstraintEngine
from fifth_layer.world_model.latent_physical_state_builder import LatentPhysicalStateBuilder
from fifth_layer.world_model.physical_state import PhysicalAttribute as A
from evaluation.tests.test_cross_modal_consequences import collision, contact, latent, snapshot, constraint


def moving():
    world = snapshot(positions=(0,))
    obj = world.objects[0]
    obj = replace(obj, attributes={**obj.attributes,
        'motion_state': A('moving', 'estimated', ('motion.source',), 'sampled motion')})
    return latent(replace(world, objects=(obj,)))


def build(state=None, constraints=None, consequences=None, **kwargs):
    return MultipleFuturesBuilder().build(moving() if state is None else state, constraints, consequences, **kwargs)


def branch(kind='motion_persists', bundle=None):
    return next(b for b in (bundle or build()).branches if b.assumptions[0].assumption_type == kind)


def change(b, name):
    return next(c for c in b.predicted_changes if c['field'] == name)


def discontinuity():
    r = constraint('inertia_consistency', 'abrupt_change_detected', 'violated', ('o0',))
    b = B(r.scene_id, r.timestamp, (r,))
    return latent(constraints=b), b


@pytest.mark.parametrize('field,value', [
    ('future_id', ''), ('scene_id', ''), ('session_id', ' '), ('source_physical_state_id', ''),
    ('coordinate_frame_id', ''), ('rule_id', ''), ('branch_family', 'hidden_actor_arrives'),
    ('status', 'observed'), ('status', 'true'), ('status', 'expected'),
    ('horizon_kind', 'long_horizon'), ('horizon_kind', None),
    ('horizon_value', 0), ('horizon_value', 1), ('horizon_value', -1), ('horizon_value', float('nan')),
    ('confidence', 0), ('confidence', .5), ('confidence', 1), ('confidence', float('nan')),
    ('timestamp', -1), ('timestamp', True), ('timestamp', float('inf')),
    ('object_ids', ()), ('source_field_references', ()), ('source_constraint_ids', 'constraint'),
    ('assumptions', ()), ('assumptions', ('motion persists',)), ('predicted_changes', ()),
    ('provenance', {}), ('provenance', {'tensor': [1, 2]}),
])
def test_candidate_validation(field, value):
    with pytest.raises(ValueError):
        replace(branch(), **{field: value})


@pytest.mark.parametrize('status', list(FutureBranchStatus))
def test_valid_statuses_are_not_observations(status):
    b = replace(branch(), status=status)
    assert b.confidence is None and b.status != 'observed'


@pytest.mark.parametrize('kind', list(FutureHorizonKind))
def test_qualitative_horizons(kind):
    bundle = build(horizon_kind=kind)
    assert all(b.horizon_kind == kind and b.horizon_value is None for b in bundle.branches)


def test_numeric_horizon_rejected_at_builder():
    with pytest.raises(ValueError):
        build(horizon_value=1.)


def test_horizon_changes_identity():
    a, b = build(), build(horizon_kind='short_horizon')
    assert {b.future_id for b in a.branches}.isdisjoint(b.future_id for b in b.branches)


@pytest.mark.parametrize('field,value', [('assumption_id', ''), ('assumption_type', 'fact'),
    ('statement', ''), ('source_refs', ())])
def test_assumption_validation(field, value):
    with pytest.raises(ValueError):
        replace(branch().assumptions[0], **{field: value})


def test_assumption_sources_must_exist_on_branch():
    b = branch()
    with pytest.raises(ValueError):
        replace(b, assumptions=(replace(b.assumptions[0], source_refs=('unrelated',)),))


def test_duplicate_assumptions_rejected():
    b = branch()
    with pytest.raises(ValueError):
        replace(b, assumptions=b.assumptions * 2)


@pytest.mark.parametrize('changes', [
    ({'field': 'temperature', 'value': 300, 'epistemic_status': 'branch_assumption'},),
    ({'field': 'motion_state', 'value': 'moving', 'epistemic_status': 'observed'},),
    ({'field': 'motion_state', 'value': 0, 'epistemic_status': 'branch_assumption'},),
    ({'field': 'motion_state', 'value': 'moving', 'epistemic_status': 'branch_assumption', 'probability': .5},),
    ({'field': 'consequence_reference', 'value': 'missing', 'epistemic_status': 'possible'},),
])
def test_predicted_change_contract(changes):
    with pytest.raises(ValueError):
        replace(branch(), predicted_changes=changes)


def test_duplicate_changes_rejected():
    b = branch()
    with pytest.raises(ValueError):
        replace(b, predicted_changes=b.predicted_changes * 2)


def test_deep_immutability_and_detachment():
    metadata = {'source': {'references': ['a']}}
    b = replace(branch(), provenance=metadata)
    metadata['source']['references'].append('b')
    assert b.provenance['source']['references'] == ('a',)
    with pytest.raises(FrozenInstanceError):
        b.confidence = .5
    with pytest.raises(TypeError):
        b.predicted_changes[0]['value'] = 'stationary'
    with pytest.raises(TypeError):
        b.provenance['source']['references'] = ()
    with pytest.raises(FrozenInstanceError):
        b.assumptions[0].statement = 'fact'
    bundle = build()
    with pytest.raises(FrozenInstanceError):
        bundle.branches = ()
    with pytest.raises(FrozenInstanceError):
        bundle.branch_relations[0].basis = 'different'


@pytest.mark.parametrize('value', [{'x': object()}, {'x': float('nan')}, {'x': 'a'*16385}])
def test_bounded_metadata(value):
    with pytest.raises(ValueError):
        replace(branch(), provenance=value)


def test_metadata_depth_bound():
    value = {'x': None}
    for _ in range(30):
        value = {'x': value}
    with pytest.raises(ValueError):
        replace(branch(), provenance=value)


def test_bundle_rejects_duplicate_identity_and_semantics():
    bundle = build()
    b = bundle.branches[0]
    for duplicate in (b, replace(b, future_id='different')):
        with pytest.raises(ValueError):
            replace(bundle, branches=(b, duplicate), branch_relations=())


@pytest.mark.parametrize('field,value', [('scene_id', 'other'), ('timestamp', 3.), ('timestamp', None),
    ('session_id', 'other'), ('coordinate_frame_id', 'other'), ('source_physical_state_id', 'other')])
def test_bundle_branch_context_mismatch(field, value):
    bundle = build()
    with pytest.raises(ValueError):
        replace(bundle, branches=(replace(bundle.branches[0], **{field: value}),), branch_relations=())


@pytest.mark.parametrize('field,value', [('branches', ('wrong',)), ('branch_relations', ('wrong',)),
                                        ('source_physical_state_id', ''), ('provenance', {})])
def test_bundle_contract(field, value):
    with pytest.raises(ValueError):
        replace(build(), **{field: value})


@pytest.mark.parametrize('changes', [{'relation_id': ''}, {'branch_ids': ('a', 'a')},
    {'branch_ids': ('a',)}, {'relation': 'best'}, {'basis': ''}])
def test_relation_validation(changes):
    with pytest.raises(ValueError):
        replace(build().branch_relations[0], **changes)


def test_relation_dangling_reference_rejected():
    bundle = build()
    relation = replace(bundle.branch_relations[0], branch_ids=(bundle.branches[0].future_id, 'missing'))
    with pytest.raises(ValueError):
        replace(bundle, branch_relations=(relation,))


def test_duplicate_relation_pair_rejected():
    bundle = build()
    r = bundle.branch_relations[0]
    with pytest.raises(ValueError):
        replace(bundle, branch_relations=(r, replace(r, relation_id='other')))


def test_false_exclusivity_rejected():
    bundle = build()
    r = replace(bundle.branch_relations[0], relation='mutually_exclusive')
    with pytest.raises(ValueError):
        replace(bundle, branch_relations=(r,))


def test_contact_exclusivity_is_structural():
    bundle = build(contact())
    contact_branch = branch('contact_within_horizon', bundle)
    no_contact = branch('no_contact_within_horizon', bundle)
    r = next(r for r in bundle.branch_relations if set(r.branch_ids) == {contact_branch.future_id, no_contact.future_id})
    assert r.relation == 'mutually_exclusive'
    assert all(r.relation == 'unresolved_relation' for r in bundle.branch_relations if 'outcome_unresolved' in {
        b.assumptions[0].assumption_type for b in bundle.branches if b.future_id in r.branch_ids})
    with pytest.raises(ValueError):
        replace(bundle, branch_relations=(replace(r, relation='co_possible'),))


def test_different_horizons_not_exclusive():
    bundle = build(contact())
    b = branch('no_contact_within_horizon', bundle)
    with pytest.raises(ValueError):
        replace(bundle, branches=tuple(replace(x, horizon_kind='short_horizon') if x == b else x for x in bundle.branches))


def test_co_possible_contract_without_generated_compatibility_claim():
    bundle = build()
    r = replace(bundle.branch_relations[0], relation=FutureRelation.CO_POSSIBLE)
    assert replace(bundle, branch_relations=(r,)).branch_relations[0].relation == 'co_possible'
    assert all(r.relation == 'unresolved_relation' for r in build().branch_relations)


def test_motion_continuation_and_unresolved_alternative():
    bundle = build()
    assert len(bundle.branches) == 2
    assert branch(bundle=bundle).status == 'possible'
    assert change(branch(bundle=bundle), 'motion_state')['value'] == 'moving'
    assert branch('outcome_unresolved', bundle).status == 'indeterminate'
    assert all(b.object_ids == ('o0',) for b in bundle.branches)
    assert all(b.source_constraint_ids == b.source_consequence_ids == () for b in bundle.branches)


@pytest.mark.parametrize('motion', [
    {'value': None, 'status': 'unknown', 'derived_from': ('motion',)},
    {'value': 'moving', 'status': 'possible', 'derived_from': ('motion',)},
    {'value': 'moving', 'status': 'estimated', 'derived_from': ()},
    {'value': 'stationary', 'status': 'estimated', 'derived_from': ('motion',)},
])
def test_motion_requires_actual_attributed_moving_signal(motion):
    state = moving()
    obj = dict(state.objects[0])
    obj['attributes'] = dict(obj['attributes'], motion_state=motion)
    assert build(replace(state, objects=(obj,))).branches == ()


@pytest.mark.parametrize('source', [contact, lambda: collision()[0]])
def test_contact_sources_make_three_distinct_hypotheses(source):
    bundle = build(source())
    assert len(bundle.branches) == 3
    assert {a.assumption_type for b in bundle.branches for a in b.assumptions} == {
        'contact_within_horizon', 'no_contact_within_horizon', 'outcome_unresolved'}
    assert branch('contact_within_horizon', bundle).status == 'possible'
    no_contact = branch('no_contact_within_horizon', bundle)
    assert change(no_contact, 'contact_status')['epistemic_status'] == 'branch_assumption'
    assert branch('outcome_unresolved', bundle).status == 'indeterminate'


def test_discontinuity_has_no_cause_inference():
    state, constraints = discontinuity()
    bundle = build(state, constraints)
    assert len(bundle.branches) == 2
    assert {b.assumptions[0].assumption_type for b in bundle.branches} == {'continuity_resumes', 'outcome_unresolved'}
    assert all(b.status == 'indeterminate' for b in bundle.branches)
    assert all('tracking_or_camera_change_may_explain_signal' in b.uncertainty for b in bundle.branches)
    assert all(b.object_ids == ('o0',) for b in bundle.branches)


@pytest.mark.parametrize('status', ['unsupported', 'not_applicable'])
def test_constraints_do_not_turn_unsupported_sources_into_events(status):
    state, constraints = collision()
    constraints = replace(constraints, results=(replace(constraints.results[0], status=status),))
    assert build(state, constraints).branches == ()


def test_conflicting_constraint_keeps_all_alternatives_indeterminate():
    state, constraints = collision()
    constraints = replace(constraints, results=(replace(constraints.results[0], status='indeterminate'),))
    bundle = build(state, constraints)
    assert len(bundle.branches) == 3
    assert all(b.status == 'indeterminate' for b in bundle.branches)


def test_bare_constraint_does_not_create_physical_trigger():
    _, constraints = collision()
    assert build(latent(), constraints).branches == ()


@pytest.mark.parametrize('signal', ['overlaps', 'support_candidate', 'occlusion', 'shattering'])
def test_no_unimplemented_semantic_or_support_rules(signal):
    state = contact()
    state = replace(state, relations=(dict(state.relations[0], signal=signal),))
    assert build(state).branches == ()


def test_empty_and_missing_state_is_not_impossibility():
    for state in (latent(), latent(snapshot(positions=()))):
        b = build(state)
        assert b.branches == ()
        assert 'no_eligible_explicit_branch_source' in b.uncertainty
        assert 'no_branch_is_not_impossibility' in b.uncertainty
        assert b.provenance['missing_prerequisite_policy'] == 'skip_not_impossible'


def test_no_numeric_values_are_fabricated():
    state = moving()
    assert state.objects[0]['attributes']['velocity']['value'] is None
    before = state.to_json()
    for b in build(state).branches:
        assert b.horizon_value is None
        assert all(c['field'] == 'motion_state' for c in b.predicted_changes)
        assert 'metric_future_state_unknown' in b.uncertainty
    assert state.to_json() == before


def test_consequence_candidates_remain_references():
    state, constraints = collision()
    consequences = CrossModalConsequenceBuilder().build(state, constraints)
    before = consequences.to_json()
    bundle = build(state, constraints, consequences)
    assert len(bundle.branches) == 3
    for kind in ('contact_within_horizon', 'outcome_unresolved'):
        b = branch(kind, bundle)
        assert set(b.source_consequence_ids) == {c.consequence_id for c in consequences.candidates}
        refs = [c for c in b.predicted_changes if c['field'] == 'consequence_reference']
        assert len(refs) == 3 and all(c['epistemic_status'] == 'possible' for c in refs)
        for c in consequences.candidates:
            assert b.to_dict()['provenance']['consequence_sources'][c.consequence_id] == c.to_dict()
    assert branch('no_contact_within_horizon', bundle).source_consequence_ids == ()
    assert consequences.to_json() == before


@pytest.mark.parametrize('status', ['expected', 'possible', 'indeterminate', 'unsupported', 'unavailable'])
def test_consequence_status_is_preserved_or_not_associated(status):
    state = contact()
    source = CrossModalConsequenceBuilder().build(state)
    c = replace(source.candidates[0], status=status)
    bundle = build(state, consequences=replace(source, candidates=(c,)))
    b = branch('contact_within_horizon', bundle)
    if status in ('unsupported', 'unavailable'):
        assert b.source_consequence_ids == ()
        assert 'unassociated_consequence_candidates' in bundle.uncertainty
    else:
        assert change(b, 'consequence_reference')['epistemic_status'] == status
    assert b.status == 'possible' and b.confidence is None


def test_consequence_only_never_generates_new_semantic_branch():
    original = contact()
    source = CrossModalConsequenceBuilder().build(original)
    state = replace(original, relations=())
    bundle = build(state, consequences=source)
    assert bundle.branches == ()
    assert bundle.provenance['unassociated_consequence_ids'] == (source.candidates[0].consequence_id,)


def test_wrong_consequence_binding_rejected():
    state = contact()
    source = CrossModalConsequenceBuilder().build(state)
    c = replace(source.candidates[0], event_family='shattering')
    with pytest.raises(ValueError):
        build(state, consequences=replace(source, candidates=(c,)))


@pytest.mark.parametrize('changes', [{'scene_id': 'other'}, {'timestamp': 3.}, {'timestamp': None},
    {'session_id': 'other'}, {'coordinate_frame_id': 'other'}])
def test_consequence_bundle_context_rejected(changes):
    state = contact()
    source = CrossModalConsequenceBuilder().build(state)
    other = replace(source, candidates=(), **changes)
    with pytest.raises(ValueError):
        build(state, consequences=other)


@pytest.mark.parametrize('changes', [{'source_physical_state_id': 'other'}, {'object_ids': ('hidden',)},
    {'provenance': {'source_timestamp': 3.}}, {'provenance': {'session_id': 'other'}},
    {'provenance': {'coordinate_frame_id': 'other'}}])
def test_consequence_source_context_rejected(changes):
    state = contact()
    source = CrossModalConsequenceBuilder().build(state)
    c = replace(source.candidates[0], **changes)
    with pytest.raises(ValueError):
        build(state, consequences=replace(source, candidates=(c,)))


def test_empty_consequence_bundle_source_mismatch_rejected():
    state = contact()
    source = CrossModalConsequenceBuilder().build(state)
    source = replace(source, candidates=(), provenance={'source_physical_state_id': 'other'})
    with pytest.raises(ValueError):
        build(state, consequences=source)


def test_provenance_assumptions_and_unknowns_preserved():
    state, constraints = collision()
    source = CrossModalConsequenceBuilder().build(state, constraints)
    b = branch('contact_within_horizon', build(state, constraints, source))
    assert b.source_physical_state_id == state.latent_state_id
    assert b.object_ids == ('o0', 'o1')
    assert b.source_constraint_ids == ('c1',)
    assert 'bbox.source' in b.source_field_references
    assert b.assumptions[0].source_refs == b.source_field_references
    assert b.provenance['assumption_ids'] == (b.assumptions[0].assumption_id,)
    assert b.session_id == 'session' and b.coordinate_frame_id == 'pixels'
    assert b.rule_version == b.schema_version == VERSION
    assert b.provenance['rule_id'] == b.rule_id == 'branch_collision_risk_geometry'
    assert 'upstream_uncertainty' in b.uncertainty
    assert set(state.uncertainty) <= set(b.uncertainty)


def test_learned_signal_is_ignored_for_generation():
    state = moving()
    learned = LearnedRepresentationSignal('synthetic', 2., 'session', 'scene2.0', 'rep', 16,
        {'semantic_claim': 'hidden person behind occluder'}, temporal_change_score=100.)
    assert build(HybridWorldState(state, learned_signal=learned)).to_json() == build(state).to_json()
    assert build(HybridWorldState(latent(), learned_signal=learned)).branches == ()


@pytest.mark.parametrize('changes', [{'timestamp': 3.}, {'scene_id': 'other'}, {'session_id': 'other'},
                                    {'coordinate_frame_id': 'other'}])
def test_hybrid_mismatch_protection(changes):
    signal = LearnedRepresentationSignal('synthetic', 2., 'session', 'scene2.0', 'rep', 16, {'source': 'test'})
    with pytest.raises(ValueError):
        build(HybridWorldState(moving(), learned_signal=replace(signal, **changes)))


@pytest.mark.parametrize('changes', [{'scene_id': 'other'}, {'timestamp': 3.}, {'timestamp': 1.}])
def test_constraint_context_rejected(changes):
    with pytest.raises(ValueError):
        build(moving(), B(**({'scene_id': 'scene2.0', 'timestamp': 2.} | changes)))


@pytest.mark.parametrize('metadata', [{'source_timestamp': 3.}, {'session_id': 'other'},
                                     {'coordinate_frame_id': 'other'}])
def test_physical_source_context_rejected(metadata):
    with pytest.raises(ValueError):
        build(replace(moving(), provenance=metadata))


def test_unknown_timestamp_stays_unknown():
    b = build(replace(moving(), timestamp=None))
    assert b.timestamp is None
    assert all(x.timestamp is None and x.horizon_value is None for x in b.branches)
    assert 'temporal_alignment_unknown' in b.uncertainty


def test_retrospective_assessment_remains_source_not_future():
    old, current = snapshot(1., (0, 30)), snapshot()
    assessment = PhysicsConstraintEngine().assess(current, history=(old,),
        session_id='session', coordinate_frame_id='pixels')
    state = LatentPhysicalStateBuilder().build(current, assessment, history=(latent(old),),
        session_id='session', coordinate_frame_id='pixels')
    source = CrossModalConsequenceBuilder().build(state, assessment)
    before = assessment.to_json()
    b = build(state, assessment, source)
    assert len(b.branches) == 3
    assert b.provenance['constraint_source']['assessment_kind'] == assessment.assessment_kind
    assert b.provenance['constraint_source']['previous_timestamp'] == 1.
    assert all(x.timestamp == current.timestamp for x in b.branches)
    assert assessment.to_json() == before
    with pytest.raises(ValueError):
        build(state, replace(assessment, assessment_kind='future_prediction'))


def test_deterministic_json_and_source_order():
    state = contact()
    first = state.relations[0]
    second = dict(first, derived_from=('second.source',))
    a = build(replace(state, relations=(first, second, first)))
    b = build(replace(state, relations=(second, first)))
    assert a.to_json() == b.to_json()
    assert len(a.branches) == 6
    assert json.loads(a.to_json()) == a.to_dict()
    assert replace(a, branches=tuple(reversed(a.branches)),
                   branch_relations=tuple(reversed(a.branch_relations))).to_json() == a.to_json()
    assert a.branches[0].to_dict() == json.loads(a.branches[0].to_json())


def test_different_context_changes_ids():
    state = moving()
    original = {b.future_id for b in build(state).branches}
    for changes in ({'scene_id': 'other'}, {'session_id': 'other'}, {'coordinate_frame_id': 'other'},
                    {'timestamp': 3.}, {'latent_state_id': 'other'}):
        assert original.isdisjoint(b.future_id for b in build(replace(state, **changes)).branches)


def test_no_probability_observation_or_actor_claims():
    state, constraints = collision()
    for b in build(state, constraints).branches:
        assert b.confidence is None and b.status in ('possible', 'indeterminate')
        for key in ('probability', 'prior', 'posterior', 'likelihood', 'rank', 'supports', 'contradicts', 'observed'):
            assert key not in b.to_dict()
        for a in b.assumptions:
            assert not any(word in a.statement for word in ('will happen', 'definitely', 'confirmed',
                'observed future', 'true branch', 'hidden person', 'teleportation'))
        assert b.provenance['truth_decision'] == 'not_performed'
        assert set(b.object_ids) <= {'o0', 'o1'}


def test_process_determinism_and_no_model_imports():
    script = '''
import sys
from evaluation.tests.test_multiple_futures import build
print(build().to_json())
assert not {'torch', 'transformers', 'ultralytics', 'cv2'} & set(sys.modules)
'''
    outputs = [subprocess.run([sys.executable, '-c', script], check=True, text=True, capture_output=True,
        env={**os.environ, 'PYTHONHASHSEED': seed}).stdout for seed in ('1', '42')]
    assert outputs[0] == outputs[1]


def test_ambiguous_and_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        build(HybridWorldState(moving()), B('scene2.0', 2.))
    with pytest.raises(ValueError):
        build('not a physical state')
    with pytest.raises(ValueError):
        build(consequences='not a consequence bundle')


@pytest.mark.parametrize('source_has_constraints', [False, True])
def test_optional_constraint_detail_does_not_break_consequence_binding(source_has_constraints):
    state, constraints = collision()
    source = CrossModalConsequenceBuilder().build(state, constraints if source_has_constraints else None)
    bundle = build(state, None if source_has_constraints else constraints, source)
    assert set(branch('contact_within_horizon', bundle).source_consequence_ids) == {
        c.consequence_id for c in source.candidates}


@pytest.mark.parametrize('provenance_change', [
    {'source_signal': 'unsupported'}, {'source_references': ('unrelated',)}])
def test_consequence_id_does_not_override_source_provenance(provenance_change):
    state = contact()
    source = CrossModalConsequenceBuilder().build(state)
    c = source.candidates[0]
    c = replace(c, provenance=dict(c.provenance, **provenance_change))
    with pytest.raises(ValueError):
        build(state, consequences=replace(source, candidates=(c,)))


def test_conflicting_predicted_changes_rejected():
    b = branch()
    with pytest.raises(ValueError):
        replace(b, predicted_changes=b.predicted_changes + (
            {'field': 'motion_state', 'value': 'unresolved', 'epistemic_status': 'branch_assumption'},))


def test_object_permutation_does_not_rank_branches():
    first = moving().objects[0]
    second = dict(first, physical_object_id='o1')
    state = replace(moving(), objects=(first, second))
    bundle = build(state)
    assert len(bundle.branches) == 4
    assert bundle.to_json() == build(replace(state, objects=(second, first))).to_json()
    assert all(r.relation == 'unresolved_relation' for r in bundle.branch_relations)


def test_consequence_permutation_does_not_duplicate_branches():
    state, constraints = collision()
    source = CrossModalConsequenceBuilder().build(state, constraints)
    a = build(state, constraints, source)
    b = build(state, constraints, replace(source, candidates=tuple(reversed(source.candidates))))
    assert len(a.branches) == 3 and a.to_json() == b.to_json()


def test_constraint_uncertainty_survives_no_eligible_source():
    _, constraints = collision()
    bundle = build(latent(), constraints)
    assert bundle.branches == () and 'upstream_uncertainty' in bundle.uncertainty


def test_constraint_frame_or_session_mismatch_rejected():
    state, constraints = collision()
    for key in ('session_id', 'coordinate_frame_id'):
        r = constraints.results[0]
        r = replace(r, provenance=dict(r.provenance, **{key: 'other'}))
        with pytest.raises(ValueError):
            build(state, replace(constraints, results=(r,)))


def test_untraceable_and_absent_object_signals_not_promoted():
    state, _ = collision()
    assert build(replace(state, active_constraints=())).branches == ()
    state, _ = discontinuity()
    source = dict(state.dynamics[0], object_ids=('hidden',))
    assert build(replace(state, dynamics=(source,))).branches == ()
