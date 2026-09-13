"""Engineering correctness only; no learned-model usefulness evaluation."""
from dataclasses import FrozenInstanceError, replace
import builtins
import json

import pytest

from fifth_layer.world_model.physical_state import PhysicalWorldState, PhysicalObjectState
from fifth_layer.world_model.latent_physical_state_builder import LatentPhysicalStateBuilder
from fifth_layer.world_model.physics_constraints import PhysicsConstraintBundle, PhysicsConstraintResult
from fifth_layer.world_model.physics_constraint_engine import PhysicsConstraintEngine
from fifth_layer.world_model.hybrid_world_state import HybridWorldState, LearnedRepresentationSignal
from fifth_layer.world_model.hybrid_world_state_builder import HybridWorldStateBuilder


@pytest.fixture
def sources():
    explicit = PhysicalWorldState('scene', 2., (PhysicalObjectState('ball', 2.),))
    result = PhysicsConstraintResult('c1', 'collision_possibility', 'scene', 2.,
        ('ball',), 'satisfied', 'collision_possible', 'Possibility only',
        derived_from=('geometry',), provenance={'session_id': 'session', 'coordinate_frame_id': 'pixels'},
        uncertainty=('actual_contact_unknown',))
    constraints = PhysicsConstraintBundle('scene', 2., (result,))
    physical = LatentPhysicalStateBuilder().build(explicit, session_id='session', coordinate_frame_id='pixels')
    signal = LearnedRepresentationSignal('vjepa2', 2., 'session', 'scene', 'rep1', 1024,
        {'producer': 'external', 'input_end_timestamp': 2.}, representation_reference='opaque://rep1')
    return physical, constraints, signal


def build(sources):
    return HybridWorldStateBuilder().build(*sources)


def test_valid_and_serializable(sources):
    state = build(sources)
    assert state.timestamp == 2. and state.session_id == 'session'
    assert json.loads(state.to_json()) == state.to_dict()
    assert state.learned_signal.epistemic_status == 'learned_signal'


def test_missing_signal(sources):
    state = build((*sources[:2], None))
    assert state.learned_signal is None
    assert state.availability['learned_representation'] == 'unavailable'


def test_signal_creates_no_physical_facts(sources):
    physical = sources[0]
    before = physical.to_json()
    state = build(sources)
    assert state.physical_state is physical
    assert state.physical_state.to_json() == before
    assert len(state.physical_state.objects) == 1
    assert state.physical_state.relations == ()
    assert state.physical_state.dynamics == ()


def test_collision_possible_is_not_confirmed(sources):
    result = build(sources).physics_constraints.results[0]
    assert result is sources[1].results[0]
    assert result.finding == 'collision_possible'
    assert result.uncertainty == ('actual_contact_unknown',)


def test_unknown_is_not_filled_by_metrics(sources):
    signal = replace(sources[2], pooled_norm=0., temporal_change_score=.8, reference_similarity=1.)
    state = build((*sources[:2], signal))
    for name in ('center', 'velocity', 'displacement'):
        attr = state.physical_state.objects[0]['attributes'][name]
        assert attr['value'] is None and attr['status'] == 'unknown'


@pytest.mark.parametrize('changes', [
    {'timestamp': 3.}, {'timestamp': 1.}, {'session_id': 'wrong'},
    {'scene_id': 'other-sequence'}, {'coordinate_frame_id': 'metric'},
    {'provenance': {'input_end_timestamp': 3.}},
    {'provenance': {'session_id': 'wrong'}},
])
def test_signal_alignment_rejected(sources, changes):
    with pytest.raises(ValueError):
        build((*sources[:2], replace(sources[2], **changes)))


def test_unknown_physical_time(sources):
    physical = replace(sources[0], timestamp=None)
    assert HybridWorldStateBuilder().build(physical).timestamp is None
    with pytest.raises(ValueError):
        HybridWorldStateBuilder().build(physical, learned_signal=sources[2])


def test_determinism(sources):
    assert build(sources) == build(sources)
    assert build(sources).to_json() == build(sources).to_json()


def test_provenance_and_references_preserved(sources):
    state = build(sources)
    assert state.provenance['explicit_physical'] == sources[0].provenance
    assert state.provenance['physics_constraint']['results']['c1'] == sources[1].results[0].provenance
    assert state.provenance['learned_representation'] == sources[2].provenance
    assert state.source_references['learned_representation']['representation_id'] == 'rep1'


def test_model_name_does_not_import_model(sources, monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if any(token in name.lower() for token in ('jepa', 'torch', 'transformers')):
            raise AssertionError('model import attempted')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    assert build(sources).learned_signal.source_model == 'vjepa2'


def test_large_representation_is_only_a_reference(sources):
    state = build((*sources[:2], replace(sources[2], feature_dimension=10**9)))
    assert len(state.to_json()) < 20000
    assert state.learned_signal.representation_reference == 'opaque://rep1'


@pytest.mark.parametrize('changes', [
    {'provenance': {'embedding': [1., 2.]}}, {'provenance': {'opaque': object()}},
    {'provenance': {'tensor': b'abc'}}, {'provenance': {'text': 'x'*1025}},
    {'provenance': {str(i): i for i in range(33)}},
    {'feature_dimension': 0}, {'feature_dimension': True},
    {'pooled_norm': -1.}, {'temporal_change_score': float('nan')},
    {'reference_similarity': float('inf')}, {'timestamp': None},
    {'representation_reference': 'x'*1025},
])
def test_invalid_payload_rejected(sources, changes):
    with pytest.raises(ValueError):
        replace(sources[2], **changes)


def test_nested_immutability_and_input_detachment(sources):
    metadata = {'producer': 'external'}
    signal = replace(sources[2], provenance=metadata)
    state = build((*sources[:2], signal))
    metadata['producer'] = 'changed'
    assert state.learned_signal.provenance['producer'] == 'external'
    with pytest.raises(FrozenInstanceError):
        state.timestamp = 9.
    with pytest.raises(TypeError):
        state.provenance['learned_representation']['producer'] = 'changed'
    with pytest.raises(TypeError):
        state.physical_state.objects[0]['attributes']['center']['value'] = 0
    before = json.dumps(state.to_dict()['physics_constraints'], sort_keys=True)
    exported = state.to_dict()
    exported['physics_constraints']['results'][0]['finding'] = 'collision'
    assert json.dumps(state.to_dict()['physics_constraints'], sort_keys=True) == before
    assert state.physics_constraints is sources[1]


@pytest.mark.parametrize('context', [{}, {'session_id': 'bad', 'coordinate_frame_id': 'pixels'},
    {'session_id': 'session', 'coordinate_frame_id': 'bad'},
    {'session_id': 'session', 'coordinate_frame_id': 'pixels', 'source_timestamp': 3.}])
def test_constraint_context_rejected(sources, context):
    result = replace(sources[1].results[0], provenance=context or {'producer': 'unknown'})
    with pytest.raises(ValueError):
        build((sources[0], replace(sources[1], results=(result,)), sources[2]))


def test_constraint_scene_and_time_rejected(sources):
    for bundle in (PhysicsConstraintBundle('other', 2.), PhysicsConstraintBundle('scene', 3.)):
        with pytest.raises(ValueError):
            build((sources[0], bundle, sources[2]))


def test_empty_constraints_differ_from_missing(sources):
    assert build((sources[0], None, None)).availability['physics_constraint'] == 'unavailable'
    assert build((sources[0], PhysicsConstraintBundle('scene', 2.), None)).availability['physics_constraint'] == 'available'


def test_engine_assessment_supported(sources):
    explicit = PhysicalWorldState('scene', 2.)
    assessment = PhysicsConstraintEngine().assess(explicit, session_id='session', coordinate_frame_id='pixels')
    state = build((sources[0], assessment, sources[2]))
    assert state.physics_constraints is assessment
    assert state.source_references['physics_constraint']['assessment_id'] == assessment.assessment_id


def test_direct_constructor_cannot_bypass_alignment(sources):
    with pytest.raises(ValueError):
        HybridWorldState(*sources[:2], replace(sources[2], timestamp=3.))


def test_coordinate_uncertainty_is_explicit(sources):
    assert 'learned_coordinate_frame_unspecified' in build(sources).uncertainty
    aligned = build((*sources[:2], replace(sources[2], coordinate_frame_id='pixels')))
    assert 'learned_coordinate_frame_unspecified' not in aligned.uncertainty
