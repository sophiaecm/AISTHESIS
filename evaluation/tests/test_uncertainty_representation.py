"""Step 27A engineering contracts; synthetic sources are not scientific evidence."""
from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from fifth_layer.world_model.uncertainty_representation import (
    UncertaintyContext, UncertaintyEntry, UncertaintyBundle, UncertaintyRepresentationBuilder,
    adapt_physical_uncertainty, adapt_constraint_uncertainty, adapt_latent_uncertainty,
    adapt_cross_modal_uncertainty, adapt_future_uncertainty, adapt_belief_uncertainty,
    adapt_prediction_uncertainty, adapt_calibration_uncertainty, adapt_grounding_uncertainty,
    adapt_active_perception_uncertainty, adapt_common_evidence_uncertainty,
    adapt_reasoning_uncertainty, bundle_from_dict, CATEGORIES, STATUSES,
)
from fifth_layer.world_model.common_evidence_state import CommonEvidenceState, bounded_plain
from fifth_layer.world_model.evidence import EvidenceBundle, EvidenceItem
from fifth_layer.world_model.prediction_records import PredictionRecord
from fifth_layer.world_model.physics_constraints import PhysicsConstraintBundle
from fifth_layer.world_model.cross_modal_consequences import CrossModalConsequenceBuilder
from fifth_layer.world_model.bayesian_belief_state import BayesianBeliefStateBuilder
from fifth_layer.confidence_calibration import ConfidenceCalibrationMemory
from evaluation.tests.test_bayesian_belief_state import initialize, futures, evidence
from evaluation.tests.test_multiple_futures import moving
from evaluation.tests.test_cross_modal_consequences import snapshot, collision, constraint
from evaluation.tests.test_spatial_grounding import request, raw, plan, B as GROUNDING
from evaluation.tests.test_reasoning_connectome_v02 import route, full


CONTEXT = UncertaintyContext('scene', 2., 'session', 'pixels')


def entry(**changes):
    args = dict(context=CONTEXT, source_layer='test.source', source_type='TestSource', source_id='source',
        uncertainty_kind='physical', semantic_scope='test_field', status='available', value=.2,
        value_kind='numeric', original_field='test_field', source_references=('ref',),
        assumptions=('conditional',), provenance={'source_provenance': {'method': 'synthetic'}})
    args.update(changes)
    return UncertaintyEntry(**args)


def bundle(entries):
    return UncertaintyRepresentationBuilder().build(CONTEXT, entries)


def prediction(**changes):
    args = dict(prediction_id='p', scene_id='scene', hypothesis_id='h', hypothesis_type='continuation',
        source_timestamp=1., target_timestamp=2., horizon_seconds=1., confidence=.8, uncertainty=.3,
        evidence_for=('e1',), evidence_against=('e2',), provenance={'session_id': 'session', 'coordinate_frame_id': 'pixels'})
    args.update(changes)
    return PredictionRecord(**args)


def test_deterministic_ids_order_serialization_and_round_trip():
    a, b = entry(), entry(source_id='second', value=('conflict',), value_kind='categorical')
    first, second = bundle((a, b)), bundle((b, a))
    assert first == second
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()
    assert bundle_from_dict(json.loads(first.to_json())) == first
    assert first.source_lineage == (('test.source', 'TestSource', 'second'), ('test.source', 'TestSource', 'source'))


@pytest.mark.parametrize('change', [dict(value=.4), dict(source_id='other'), dict(original_field='other'),
    dict(assumptions=('other',)), dict(provenance={'method': 'other'}), dict(context=replace(CONTEXT, timestamp=1.))])
def test_identity_covers_semantic_content(change):
    assert entry(**change).uncertainty_id != entry().uncertainty_id


def test_deep_immutability_and_detachment():
    value = {'uncertainty': [1, 2]}
    provenance = {'method': ['original']}
    e = entry(value=value, value_kind='structured', provenance=provenance)
    output = bundle([e])
    value['uncertainty'].append(3)
    provenance['method'].append('changed')
    assert e.value['uncertainty'] == (1, 2)
    assert e.provenance['method'] == ('original',)
    with pytest.raises(TypeError): e.value['uncertainty'] = ()
    with pytest.raises(FrozenInstanceError): e.status = 'unavailable'
    with pytest.raises(FrozenInstanceError): output.entries = ()
    with pytest.raises(FrozenInstanceError): CONTEXT.timestamp = 10
    data = output.to_dict()
    data['entries'][0]['value']['uncertainty'].append(10)
    assert output.entries[0].value['uncertainty'] == (1, 2)


@pytest.mark.parametrize('category', CATEGORIES)
def test_explicit_categories(category):
    assert entry(uncertainty_kind=category).uncertainty_kind == category


@pytest.mark.parametrize('status', STATUSES)
def test_statuses_are_first_class(status):
    value = None if status in ('unavailable', 'not_applicable') else 'ambiguous'
    e = entry(status=status, value=value, value_kind='categorical')
    assert e.status == status and e.value == value


@pytest.mark.parametrize('kind', ['numeric', 'backend_score'])
@pytest.mark.parametrize('value', [-23., 42., 0.])
def test_unbounded_source_values_remain_unbounded(kind, value):
    assert entry(value_kind=kind, value=value).value == value


@pytest.mark.parametrize('kind', ['confidence', 'scalar_uncertainty', 'calibration_reliability'])
@pytest.mark.parametrize('value', [-.1, 1.1, True])
def test_only_explicit_unit_interval_contracts_bounded(kind, value):
    with pytest.raises(ValueError): entry(value_kind=kind, value=value)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('where', ['value', 'provenance', 'context'])
def test_nonfinite_rejected_recursively(value, where):
    with pytest.raises(ValueError):
        if where == 'value': entry(value={'nested': [value]}, value_kind='structured')
        elif where == 'provenance': entry(provenance={'nested': [value]})
        else: replace(CONTEXT, timestamp=value)


@pytest.mark.parametrize('change', [dict(status='certain'), dict(uncertainty_kind='global'),
    dict(value_kind='truth_probability'), dict(transformation='1-confidence'), dict(value=None),
    dict(status='unavailable', value=0), dict(status='not_applicable', value=.5),
    dict(source_id=''), dict(original_field=''), dict(source_references='ref'), dict(assumptions='assumption'),
    dict(provenance=[]), dict(context={}), dict(value_kind='categorical', value=1.)])
def test_invalid_entry_contract(change):
    with pytest.raises(ValueError): entry(**change)


@pytest.mark.parametrize('name,value', [('scene_id', 'other'), ('session_id', 'other'),
    ('coordinate_frame_id', 'other'), ('timestamp', 3.)])
def test_bundle_rejects_context_mismatch(name, value):
    with pytest.raises(ValueError, match='mismatch|timestamp'):
        bundle((entry(context=replace(CONTEXT, **{name: value})),))


@pytest.mark.parametrize('name', ['session_id', 'coordinate_frame_id'])
def test_unknown_bundle_identity_does_not_hide_conflicting_sources(name):
    a = entry()
    b = entry(source_id='second', context=replace(CONTEXT, **{name: 'other'}))
    with pytest.raises(ValueError, match='mismatch'):
        UncertaintyBundle(replace(CONTEXT, **{name: None}), (a, b))


def test_unknown_time_remains_unknown_and_history_keeps_source_time():
    unknown = entry(context=replace(CONTEXT, timestamp=None, session_id=None, coordinate_frame_id=None))
    old = entry(context=replace(CONTEXT, timestamp=1.), source_id='old')
    b = bundle((unknown, old))
    assert {e.context.timestamp for e in b.entries} == {None, 1.}
    assert unknown.context.session_id is None
    with pytest.raises(ValueError): UncertaintyBundle(replace(CONTEXT, timestamp=None), (old,))


def test_separate_values_and_no_aggregation():
    entries = adapt_prediction_uncertainty(prediction())
    b = bundle(entries)
    assert {e.original_field: e.value for e in b.entries} == {'confidence': .8, 'uncertainty': .3}
    assert all(e.transformation == 'none' for e in entries)
    assert not {'score', 'mean', 'global_uncertainty', 'overall_uncertainty', 'winner'} & b.to_dict().keys()
    assert all(e.source_references == ('e1', 'e2') for e in entries)


def test_missing_prediction_uncertainty_not_one_minus_confidence():
    entries = adapt_prediction_uncertainty(prediction(uncertainty=None))
    assert entries[0].value == .8
    assert entries[1].value is None and entries[1].status == 'unavailable'


def test_belief_distribution_assumptions_and_provenance_preserved_without_winner():
    source = initialize()
    before = source.to_json()
    e, = adapt_belief_uncertainty(source)
    assert dict(e.value) == {b.hypothesis_id: b.posterior_probability for b in source.beliefs}
    assert e.value_kind == 'branch_label_distribution'
    assert e.semantic_scope == 'categorical_branch_labels_not_event_frequencies'
    assert set(e.assumptions) == {a for b in source.beliefs for a in b.assumptions}
    assert e.provenance['source_provenance'] == source.provenance
    assert e.provenance['source_details']['beliefs'][0]['source_future']['assumptions']
    assert source.to_json() == before
    assert 'winner' not in e.to_dict() and 'confidence' not in e.to_dict()


def test_belief_unavailable_is_not_zero():
    source = BayesianBeliefStateBuilder().initialize(futures(), initialization_mode='unavailable')
    e, = adapt_belief_uncertainty(source)
    assert e.status == 'unavailable' and e.value is None
    assert all(b['posterior_probability'] is None for b in e.provenance['source_details']['beliefs'])


def test_belief_indeterminate_is_not_point_five():
    initial = initialize()
    source = BayesianBeliefStateBuilder().update(initial, evidence(initial, (0., 0.)))
    e, = adapt_belief_uncertainty(source)
    assert e.status == 'indeterminate' and e.value is None


@pytest.mark.parametrize('distribution', [{}, {'a': .9}, {'a': -.1, 'b': 1.1}, {'a': float('nan')}, {'a': None}])
def test_invalid_explicit_distribution(distribution):
    with pytest.raises(ValueError): entry(value_kind='branch_label_distribution', value=distribution)


def calibration(n=0):
    memory = ConfidenceCalibrationMemory()
    for i in range(n):
        memory.update(dict(prediction_id=str(i), status='correct', evaluated_timestamp=float(i),
            prediction=dict(source='temporal_live', prediction_type='position', class_name='object')))
    return memory.calibrate(.8, 'temporal_live', 'position', 'object', float(n))


@pytest.mark.parametrize('n', [0, 4, 5, 8])
def test_real_calibration_output_is_represented_without_recalibration(n):
    source = calibration(n)
    before = json.dumps(source)
    e, = adapt_calibration_uncertainty(source, context=CONTEXT, source_id='issuance')
    assert e.status == ('insufficient_evidence' if n < 5 else 'available')
    assert e.value == source['calibration_reliability']
    assert e.value_kind == 'calibration_reliability'
    assert e.provenance['source_output']['calibration_samples'] == n
    assert e.provenance['source_output']['raw_confidence'] == .8
    assert json.dumps(source) == before
    source['calibration_bucket'].append('mutated')
    assert 'mutated' not in e.provenance['source_output']['calibration_bucket']


@pytest.mark.parametrize('change', [dict(calibration_samples=-1), dict(calibration_samples=True),
    dict(calibration_reliability=1.1), dict(calibration_bucket=[]), dict(extra='unsupported'),
    dict(calibration_samples=5), dict(calibration_bucket=['empirical'])])
def test_calibration_exact_shape_and_consistency(change):
    with pytest.raises(ValueError):
        adapt_calibration_uncertainty(dict(calibration(), **change), context=CONTEXT, source_id='id')


@pytest.mark.parametrize('score', [-2., 0., .8, 12., None])
def test_grounding_backend_score_is_never_confidence(score):
    r = request()
    source = GROUNDING.integrate_observations(r, (raw(r, backend_score=score),))
    before = source.to_json()
    entries = adapt_grounding_uncertainty(source)
    e = next(e for e in entries if e.value_kind == 'backend_score')
    assert e.value == score
    assert e.status == ('unavailable' if score is None else 'available')
    assert 'backend_specific' in e.semantic_scope
    assert all(e.value_kind != 'confidence' for e in entries)
    assert source.to_json() == before


@pytest.mark.parametrize('status', ['no_candidate', 'unavailable', 'indeterminate'])
def test_empty_grounding_never_implies_absence(status):
    source = replace(GROUNDING.integrate_observations(request(), ()), grounding_status=status)
    e, = adapt_grounding_uncertainty(source)
    assert e.status == ('indeterminate' if status == 'indeterminate' else 'unavailable')
    assert e.provenance['source_details']['no_candidate_not_physical_absence'] is True
    if status != 'indeterminate': assert e.value is None


def test_multiple_grounding_candidates_remain_indeterminate():
    r = request()
    source = GROUNDING.integrate_observations(r, (raw(r), raw(r, coordinates=(2, 3, 10, 20))))
    entries = adapt_grounding_uncertainty(source)
    assert entries[0].status == 'indeterminate'
    assert len(entries[0].provenance['source_details']['regions']) == 2


def test_physical_conflict_stays_categorical():
    r = replace(constraint('inertia_consistency', 'abrupt_change_detected', 'violated', ('o0',)),
                uncertainty=('conflicting_constraints',))
    source = PhysicsConstraintBundle(r.scene_id, r.timestamp, (r,))
    e, = adapt_constraint_uncertainty(source)
    assert e.value == ('conflicting_constraints',)
    assert e.provenance['source_details']['status'] == 'violated'
    assert e.provenance['source_details']['finding'] == 'abrupt_change_detected'
    assert e.source_references == r.derived_from


@pytest.mark.parametrize('status', ['indeterminate', 'unsupported', 'not_applicable'])
def test_physical_status_mapping_preserves_original(status):
    r = replace(constraint('inertia_consistency', 'abrupt_change_detected', 'violated', ('o0',)), status=status)
    e, = adapt_constraint_uncertainty(PhysicsConstraintBundle(r.scene_id, r.timestamp, (r,)))
    assert e.status == {'unsupported': 'unavailable'}.get(status, status)
    assert e.provenance['source_details']['status'] == status


def test_cross_modal_unavailable_is_not_no_sound():
    source = CrossModalConsequenceBuilder().build(collision()[0])
    candidate = replace(source.candidates[0], status='unavailable')
    source = replace(source, candidates=(candidate,))
    entries = adapt_cross_modal_uncertainty(source)
    e = next(e for e in entries if e.source_id == candidate.consequence_id)
    assert e.status == 'unavailable' and e.value is None
    assert e.provenance['source_details']['modality'] == candidate.modality
    assert e.provenance['source_details']['uncertainty'] == candidate.uncertainty


def test_future_multiplicity_assumptions_and_unknown_horizons():
    source = futures()
    entries = adapt_future_uncertainty(source)
    branches = [e for e in entries if e.original_field == 'uncertainty']
    assert {e.source_id for e in branches} == {b.future_id for b in source.branches}
    assert all(e.assumptions and e.provenance['source_details']['assumptions'] for e in branches)
    horizons = [e for e in entries if e.uncertainty_kind == 'temporal']
    assert len(horizons) == len(branches)
    assert all(e.value is None and e.status == 'unavailable' for e in horizons)
    assert bounded_plain(entries[0].value) == bounded_plain(source.branch_relations)


def test_common_evidence_keeps_pixel_uncertainty_and_missing_modality():
    item = EvidenceItem('visual1', 'scene', 'visual', 'detector', 'frame',
        {'position_uncertainty': 25.}, 1., confidence=.8, epistemic_status='observed',
        provenance={'session_id': 'session', 'coordinate_frame_id': 'pixels'})
    source = CommonEvidenceState('scene', 'session', 2., EvidenceBundle('scene', (item,)), 'pixels')
    entries = adapt_common_evidence_uncertainty(source)
    e = next(e for e in entries if e.original_field == 'item.value.position_uncertainty')
    assert e.value == 25. and e.uncertainty_kind == 'observational'
    assert e.context.timestamp == 1.
    missing = next(e for e in entries if e.original_field == 'availability.sensory')
    assert missing.value is None and missing.status == 'unavailable'
    assert bundle(entries).entries


@pytest.mark.parametrize('factory,adapter', [
    (snapshot, adapt_physical_uncertainty), (moving, adapt_latent_uncertainty),
    (lambda: CrossModalConsequenceBuilder().build(collision()[0]), adapt_cross_modal_uncertainty),
    (futures, adapt_future_uncertainty), (initialize, adapt_belief_uncertainty),
    (plan, adapt_active_perception_uncertainty), (lambda: route(full()), adapt_reasoning_uncertainty),
])
def test_real_source_adapters_preserve_sources_and_round_trip(factory, adapter):
    source = factory()
    before = source.to_json()
    entries = adapter(source)
    assert entries
    context = entries[0].context
    output = UncertaintyBundle(context, entries)
    assert bundle_from_dict(output.to_dict()) == output
    assert source.to_json() == before
    assert all(e.original_field and e.source_id and e.provenance for e in entries)


@pytest.mark.parametrize('adapter', [adapt_physical_uncertainty, adapt_constraint_uncertainty,
    adapt_latent_uncertainty, adapt_cross_modal_uncertainty, adapt_future_uncertainty,
    adapt_belief_uncertainty, adapt_prediction_uncertainty, adapt_grounding_uncertainty,
    adapt_active_perception_uncertainty, adapt_common_evidence_uncertainty, adapt_reasoning_uncertainty])
def test_adapters_reject_arbitrary_dicts(adapter):
    with pytest.raises(ValueError): adapter({})


@pytest.mark.parametrize('field,value', [('schema_version', 'future'), ('bundle_id', 'forged'),
    ('source_lineage', []), ('extra', 'field')])
def test_round_trip_rejects_tampering(field, value):
    data = bundle((entry(),)).to_dict()
    data[field] = value
    with pytest.raises(ValueError): bundle_from_dict(data)


def test_round_trip_rejects_entry_tampering_and_duplicate_entries():
    data = bundle((entry(),)).to_dict()
    data['entries'][0]['value'] = .9
    with pytest.raises(ValueError): bundle_from_dict(data)
    with pytest.raises(ValueError): bundle((entry(), entry()))
    with pytest.raises(ValueError): bundle(({},))


@pytest.mark.parametrize('payload', [b'raw', {'tensor': [1]}, {'image': [1]}, {'nested': float('nan')}, 'a' * 16385])
def test_opaque_and_oversized_payloads_rejected(payload):
    with pytest.raises(ValueError): entry(value_kind='structured', value=payload)


def test_depth_and_cycle_limits():
    deep = {}
    for _ in range(26): deep = {'next': deep}
    with pytest.raises(ValueError): entry(provenance=deep)
    cyclic = {}
    cyclic['cycle'] = cyclic
    with pytest.raises(ValueError): entry(provenance=cyclic)


def test_source_collection_permutation():
    source = futures()
    other = replace(source, branches=source.branches[::-1], branch_relations=source.branch_relations[::-1])
    a, b = adapt_future_uncertainty(source), adapt_future_uncertainty(other)
    assert UncertaintyBundle(a[0].context, a).to_json() == UncertaintyBundle(b[0].context, b).to_json()


def test_no_heavy_imports_or_random_identity_across_processes():
    code = """
import sys
from fifth_layer.world_model.uncertainty_representation import UncertaintyContext, UncertaintyBundle
assert not {'torch', 'numpy', 'transformers', 'cv2', 'PIL'} & set(sys.modules)
print(UncertaintyBundle(UncertaintyContext('scene'), ()).to_json())
"""
    outputs = [subprocess.check_output([sys.executable, '-c', code], text=True,
        env=dict(os.environ, PYTHONHASHSEED=seed)) for seed in ('1', '42')]
    assert outputs[0] == outputs[1]


def test_production_has_no_io_or_aggregation_calls():
    import ast
    from fifth_layer.world_model import uncertainty_representation as module
    tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    forbidden = {'open', 'eval', 'exec', 'uuid4', 'random', 'time', 'mean', 'average', 'argmax', 'download', 'calibrate', 'update'}
    calls = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
             for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))}
    assert not forbidden & calls


def test_empty_constraint_inventory_is_not_zero_uncertainty():
    e, = adapt_constraint_uncertainty(PhysicsConstraintBundle('scene', None))
    assert e.value is None and e.status == 'unavailable'
    assert e.context.timestamp is None


def test_declared_source_provenance_conflict_rejected():
    with pytest.raises(ValueError, match='scene_id mismatch'):
        adapt_prediction_uncertainty(prediction(provenance={'scene_id': 'other'}))


def test_unsupported_source_schema_rejected():
    with pytest.raises(ValueError):
        adapt_physical_uncertainty(replace(snapshot(), schema_version='future'))


def test_grounding_permutation_retains_same_uncertainty_bundle():
    r = request()
    observations = (raw(r, backend_score=.2), raw(r, coordinates=(2, 3, 10, 20), backend_score=.9))
    a = adapt_grounding_uncertainty(GROUNDING.integrate_observations(r, observations))
    b = adapt_grounding_uncertainty(GROUNDING.integrate_observations(r, observations[::-1]))
    assert UncertaintyBundle(a[0].context, a).to_json() == UncertaintyBundle(b[0].context, b).to_json()


def test_metadata_mapping_order_does_not_change_identity():
    assert entry(provenance={'a': 1, 'b': 2}) == entry(provenance={'b': 2, 'a': 1})


def test_combined_real_layers_keep_separate_lineage_and_semantics():
    latent = moving()
    future = futures()
    belief = initialize(future)
    entries = (*adapt_latent_uncertainty(latent), *adapt_future_uncertainty(future),
               *adapt_belief_uncertainty(belief))
    result = UncertaintyBundle(entries[0].context, entries)
    assert len(result.entries) == len(entries)
    assert {e.uncertainty_kind for e in entries} == {'physical', 'hypothesis', 'temporal', 'belief'}
    assert len([e for e in entries if e.value_kind == 'branch_label_distribution']) == 1
