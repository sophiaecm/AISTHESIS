"""Explicit synthetic models test mathematics/contracts, not empirical validity."""
from dataclasses import FrozenInstanceError, replace
from decimal import localcontext, ROUND_DOWN
import json
import os
import subprocess
import sys

import pytest

from fifth_layer.world_model.bayesian_belief_state import (
    BayesianBeliefStateBuilder, BeliefInitializationMode, LikelihoodEvidence,
    VERSION)
from fifth_layer.world_model.multiple_futures import MultipleFuturesBuilder
from fifth_layer.world_model.cross_modal_consequences import CrossModalConsequenceBuilder
from fifth_layer.world_model.hybrid_world_state import HybridWorldState, LearnedRepresentationSignal
from evaluation.tests.test_multiple_futures import moving, discontinuity
from evaluation.tests.test_cross_modal_consequences import collision, contact, latent, snapshot


def futures():
    return MultipleFuturesBuilder().build(moving())


def initialize(source=None, values=(.6, .4)):
    source = futures() if source is None else source
    ids = sorted(b.future_id for b in source.branches)
    return BayesianBeliefStateBuilder().initialize(source, priors=dict(zip(ids, values)),
        prior_provenance={'source_id': 'synthetic-prior', 'kind': 'test_fixture'})


def evidence(state, values=(.8, .2), **changes):
    args = dict(evidence_id='event1', scene_id=state.scene_id, timestamp=state.timestamp,
        session_id=state.session_id, coordinate_frame_id=state.coordinate_frame_id,
        likelihood_by_hypothesis=dict(zip(state.source_future_ids, values)),
        provenance={'source_id': 'synthetic-likelihood', 'kind': 'test_fixture'})
    args.update(changes)
    return LikelihoodEvidence(**args)


def update(state, source=None, **kwargs):
    return BayesianBeliefStateBuilder().update(state, evidence(state) if source is None else source, **kwargs)


def posterior(state):
    return tuple(b.posterior_probability for b in state.beliefs)


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan'), float('inf'), -float('inf'), True, None, '0.5'])
def test_invalid_prior_numbers(value):
    source = futures()
    ids = sorted(b.future_id for b in source.branches)
    with pytest.raises(ValueError):
        BayesianBeliefStateBuilder().initialize(source, priors={ids[0]: value, ids[1]: .5},
            prior_provenance={'source': 'test'})


@pytest.mark.parametrize('value', [-.1, 1.1, float('nan'), float('inf'), True, None, '0.5'])
def test_invalid_likelihood_numbers(value):
    with pytest.raises(ValueError):
        evidence(initialize(), (value, .5))


@pytest.mark.parametrize('field,value', [('belief_state_id', ''), ('scene_id', ''), ('session_id', ''),
    ('source_multiple_futures_id', ''), ('timestamp', -1), ('timestamp', float('nan')),
    ('initialization_mode', 'learned'), ('update_status', 'true'), ('normalization_constant', -1),
    ('update_index', -1), ('update_index', True), ('provenance', {})])
def test_state_contract_validation(field, value):
    with pytest.raises(ValueError):
        replace(initialize(), **{field: value})


@pytest.mark.parametrize('field,value', [('hypothesis_id', ''), ('hypothesis_id', 'unknown'),
    ('prior_probability', -.1), ('posterior_probability', 1.1), ('likelihood', float('inf')),
    ('confidence', 0), ('confidence', .5), ('source_future', 'wrong'),
    ('update_status', 'observed'), ('source_evidence_ids', ('same', 'same')), ('provenance', {})])
def test_hypothesis_contract_validation(field, value):
    with pytest.raises(ValueError):
        replace(initialize().beliefs[0], **{field: value})


@pytest.mark.parametrize('field,value', [('evidence_id', ''), ('scene_id', ''), ('session_id', ''),
    ('timestamp', -1), ('status', 'calibrated'), ('evidence_type', ''),
    ('likelihood_by_hypothesis', None), ('provenance', {}), ('provenance', {'tensor': [1]})])
def test_likelihood_contract_validation(field, value):
    with pytest.raises(ValueError):
        replace(evidence(initialize()), **{field: value})


def test_duplicate_and_unknown_hypotheses_rejected():
    state = initialize()
    with pytest.raises(ValueError):
        replace(state, beliefs=(state.beliefs[0], state.beliefs[0]))
    with pytest.raises(ValueError):
        replace(state, source_future_ids=('unknown', state.source_future_ids[0]))


def test_extra_missing_and_unknown_prior_ids():
    source = futures()
    ids = [b.future_id for b in source.branches]
    for priors in ({ids[0]: 1.}, {ids[0]: .5, 'unknown': .5}, {i: .5 for i in ids} | {'extra': 0.}):
        with pytest.raises(ValueError):
            BayesianBeliefStateBuilder().initialize(source, priors=priors, prior_provenance={'source': 'test'})


@pytest.mark.parametrize('values', [(0., 0.), (.2, .2), (.6, .6), (.5, .500000001)])
def test_unnormalized_prior_rejected(values):
    with pytest.raises(ValueError):
        initialize(values=values)


def test_normalized_prior_accepted_without_silent_repair():
    state = initialize(values=(.5, .5000000000001))
    assert posterior(state) == (.5, .5000000000001)
    assert state.update_status == 'initialized' and state.initialization_mode == 'explicit_prior'
    assert all(b.likelihood is None for b in state.beliefs)


def test_explicit_prior_requires_provenance():
    with pytest.raises(ValueError):
        BayesianBeliefStateBuilder().initialize(futures(), priors={b.future_id: .5 for b in futures().branches})


def test_no_prior_no_uniform_fallback():
    state = BayesianBeliefStateBuilder().initialize(futures())
    assert state.initialization_mode == 'unavailable' and state.update_status == 'unavailable'
    assert posterior(state) == (None, None)
    assert all(b.prior_probability is None for b in state.beliefs)
    after = update(state)
    assert after.update_status == 'unavailable' and posterior(after) == (None, None)


def test_uniform_is_explicit_uninformative():
    state = BayesianBeliefStateBuilder().initialize(futures(), initialization_mode='uniform_uninformative')
    assert posterior(state) == (.5, .5)
    assert state.initialization_mode == BeliefInitializationMode.UNIFORM_UNINFORMATIVE
    assert state.provenance['prior_source']['meaning'] == 'no label preferred by initialization; not empirical frequency'
    assert state.probability_scope == 'categorical_branch_labels_not_event_frequencies'


@pytest.mark.parametrize('mode', ['uniform_uninformative', 'unavailable'])
def test_ambiguous_initialization_options_rejected(mode):
    with pytest.raises(ValueError):
        BayesianBeliefStateBuilder().initialize(futures(), initialization_mode=mode,
            priors={b.future_id: .5 for b in futures().branches})


def test_empty_hypothesis_space_unavailable():
    empty = MultipleFuturesBuilder().build(latent(snapshot(positions=())))
    for mode in ('unavailable', 'uniform_uninformative'):
        state = BayesianBeliefStateBuilder().initialize(empty, initialization_mode=mode)
        assert state.beliefs == () and state.update_status == 'unavailable'


def test_two_hypothesis_bayes_math():
    state = initialize()
    result = update(state)
    assert posterior(result) == pytest.approx((6/7, 1/7))
    assert result.normalization_constant == pytest.approx(.56)
    assert result.update_status == 'updated'
    assert sum(posterior(result)) == pytest.approx(1.)
    assert tuple(b.prior_probability for b in result.beliefs) == (.6, .4)
    assert tuple(b.likelihood for b in result.beliefs) == (.8, .2)


def test_three_hypothesis_math():
    source = MultipleFuturesBuilder().build(contact())
    state = initialize(source, (.2, .3, .5))
    result = update(state, evidence(state, (.1, .4, .8)))
    assert posterior(result) == pytest.approx((.02/.54, .12/.54, .4/.54))
    assert result.normalization_constant == pytest.approx(.54)


@pytest.mark.parametrize('values', [(.2, .2), (1., 1.), (1e-300, 1e-300)])
def test_identical_likelihoods_preserve_ratios(values):
    state = initialize()
    assert posterior(update(state, evidence(state, values))) == pytest.approx((.6, .4))


def test_zero_and_one_likelihoods():
    state = initialize()
    result = update(state, evidence(state, (1., 0.)))
    assert posterior(result) == (1., 0.)
    assert all(b.confidence is None for b in result.beliefs)
    assert result.beliefs[1].source_future.status in ('possible', 'indeterminate')


def test_zero_normalization_no_posterior_fabricated():
    state = initialize()
    result = update(state, evidence(state, (0., 0.)))
    assert result.update_status == 'indeterminate'
    assert result.normalization_constant == 0.
    assert posterior(result) == (None, None)
    assert tuple(b.prior_probability for b in result.beliefs) == (.6, .4)
    assert result.update_index == 0 and result.evidence_lineage == ()
    assert 'zero_normalization' in result.uncertainty


def test_zero_prior_is_supplied_not_assumed():
    state = initialize(values=(0., 1.))
    assert posterior(update(state)) == (0., 1.)
    result = update(state, evidence(state, (1., 0.)))
    assert result.update_status == 'indeterminate'


def test_tiny_weights_not_silently_zeroed():
    state = initialize(values=(1e-300, 1.))
    result = update(state, evidence(state, (1e-300, 0.)))
    assert result.update_status == 'indeterminate'
    assert result.normalization_constant is None
    assert result.provenance['update_reason'] == 'normalization_not_representable'
    assert posterior(result) == (None, None)


def test_underflowing_posterior_not_silently_impossible():
    state = initialize(values=(1e-300, 1.))
    result = update(state, evidence(state, (1e-300, 1.)))
    assert result.update_status == 'indeterminate'
    assert result.provenance['update_reason'] == 'posterior_not_representable'


def test_missing_evidence_unchanged():
    state = initialize()
    result = BayesianBeliefStateBuilder().update(state)
    assert posterior(result) == posterior(state)
    assert result.update_status == 'unchanged'
    assert result.update_index == 0 and result.evidence_event_id is None
    assert result.normalization_constant is None


def test_missing_likelihood_never_zero():
    state = initialize()
    result = update(state, evidence(state, (.8,)))
    assert result.update_status == 'unavailable'
    assert result.beliefs[1].likelihood is None
    assert posterior(result) == (None, None)
    assert tuple(b.prior_probability for b in result.beliefs) == (.6, .4)
    assert result.normalization_constant is None and result.update_index == 0


def test_missing_zero_prior_likelihood_still_blocks_update():
    state = initialize(values=(1., 0.))
    result = update(state, evidence(state, (.8,)))
    assert result.update_status == 'unavailable'


@pytest.mark.parametrize('status,expected', [('unavailable', 'unchanged'), ('invalid', 'invalid_evidence')])
def test_explicit_unusable_evidence_status(status, expected):
    state = initialize()
    result = update(state, evidence(state, (), status=status))
    assert result.update_status == expected and result.update_index == 0
    assert tuple(b.prior_probability for b in result.beliefs) == (.6, .4)


def test_invalid_status_cannot_smuggle_likelihoods():
    with pytest.raises(ValueError):
        evidence(initialize(), status='invalid')


@pytest.mark.parametrize('status', ['possible', 'indeterminate', 'unsupported', 'unavailable'])
def test_all_branch_statuses_retained_without_zeroing(status):
    source = futures()
    source = replace(source, branches=tuple(replace(b, status=status) for b in source.branches))
    state = initialize(source)
    assert len(state.beliefs) == len(source.branches)
    assert posterior(state) == (.6, .4)
    assert all(b.source_future.status == status for b in state.beliefs)


def test_branch_provenance_and_consequences_remain_candidates():
    physical, constraints = collision()
    consequences = CrossModalConsequenceBuilder().build(physical, constraints)
    source = MultipleFuturesBuilder().build(physical, constraints, consequences)
    state = initialize(source, (.2, .3, .5))
    result = update(state, evidence(state, (.4, .5, .9)))
    originals = {b.future_id: b for b in source.branches}
    for belief in result.beliefs:
        original = originals[belief.source_future_id]
        assert belief.source_future.to_json() == original.to_json()
        assert belief.assumptions == tuple(a.assumption_id for a in original.assumptions)
        assert belief.source_future.object_ids == original.object_ids
        assert belief.source_future.source_constraint_ids == original.source_constraint_ids
        assert belief.source_future.source_consequence_ids == original.source_consequence_ids
        assert all(c['epistemic_status'] != 'observed' for c in belief.source_future.predicted_changes)
    assert state.source_multiple_futures_schema == 'multiple-futures-0.2'
    assert state.provenance['source_branch_relations']


def test_sequential_updates_use_posterior_and_preserve_lineage():
    initial = initialize()
    first = update(initial)
    second_evidence = evidence(first, (.5, .9), evidence_id='event2', timestamp=3.)
    second = update(first, second_evidence, as_of_timestamp=3.)
    assert tuple(b.prior_probability for b in second.beliefs) == posterior(first)
    assert posterior(second) == pytest.approx((.3/.39, .09/.39))
    assert second.previous_belief_state_id == first.belief_state_id
    assert second.update_index == 2
    assert [e.evidence_id for e in second.evidence_lineage] == ['event1', 'event2']
    assert all(b.source_evidence_ids == ('event1', 'event2') for b in second.beliefs)
    assert second.provenance['prior_source'] == initial.provenance['prior_source']
    assert second.evidence_lineage[0].provenance['source_id'] == 'synthetic-likelihood'


def test_reapplying_event_rejected_even_with_changed_numbers():
    first = update(initialize())
    for e in (evidence(first), evidence(first, (.1, .9))):
        with pytest.raises(ValueError):
            update(first, e)


def test_failed_update_does_not_consume_evidence_or_replace_prior():
    initial = initialize()
    failed = update(initial, evidence(initial, (.8,)))
    recovered = update(failed, evidence(failed))
    assert posterior(recovered) == pytest.approx((6/7, 1/7))
    assert recovered.update_index == 1
    assert recovered.previous_belief_state_id == failed.belief_state_id


def test_no_evidence_after_success_does_not_double_apply():
    first = update(initialize())
    after = BayesianBeliefStateBuilder().update(first)
    assert posterior(after) == posterior(first)
    assert after.update_index == 1 and after.evidence_lineage == first.evidence_lineage


@pytest.mark.parametrize('changes', [{'scene_id': 'other'}, {'session_id': 'other'},
    {'coordinate_frame_id': 'other'}, {'coordinate_frame_id': None}])
def test_evidence_context_mismatch(changes):
    state = initialize()
    with pytest.raises(ValueError):
        update(state, evidence(state, **changes))


@pytest.mark.parametrize('changes', [{'scene_id': 'other'}, {'session_id': 'other'},
    {'coordinate_frame_id': 'other'}, {'timestamp': 3.}])
def test_hypothesis_bundle_context_mismatch(changes):
    state = initialize()
    with pytest.raises(ValueError):
        b = replace(state.beliefs[0], **changes)
        replace(state, beliefs=(b, state.beliefs[1]))


def test_future_evidence_requires_explicit_cutoff():
    state = initialize()
    e = evidence(state, timestamp=3.)
    with pytest.raises(ValueError):
        update(state, e)
    result = update(state, e, as_of_timestamp=3.)
    assert result.timestamp == 3. and result.update_status == 'updated'
    assert all(b.source_future.timestamp == 2. for b in result.beliefs)


def test_out_of_order_evidence_and_cutoff_rejected():
    state = initialize()
    with pytest.raises(ValueError):
        update(state, evidence(state, timestamp=1.))
    with pytest.raises(ValueError):
        update(state, as_of_timestamp=1.)


def test_unknown_evidence_time_unavailable():
    state = initialize()
    result = update(state, evidence(state, timestamp=None))
    assert result.update_status == 'unavailable'
    assert 'temporal_alignment_unknown' in result.uncertainty


def test_unknown_source_time_does_not_turn_into_zero():
    source = MultipleFuturesBuilder().build(replace(moving(), timestamp=None))
    state = initialize(source)
    assert state.timestamp is None
    result = update(state, evidence(state, timestamp=3.), as_of_timestamp=3.)
    assert result.update_status == 'unavailable' and posterior(result) == (None, None)


def test_nested_future_likelihood_provenance_rejected():
    with pytest.raises(ValueError):
        evidence(initialize(), provenance={'source_timestamp': 3.})


def test_future_prior_provenance_rejected():
    source = futures()
    with pytest.raises(ValueError):
        BayesianBeliefStateBuilder().initialize(source, priors={b.future_id: .5 for b in source.branches},
            prior_provenance={'source_timestamp': 3.})


def test_unknown_likelihood_id_rejected():
    state = initialize()
    e = evidence(state, likelihood_by_hypothesis={'unknown': .5})
    with pytest.raises(ValueError):
        update(state, e)


def test_bounded_immutable_state_evidence_and_json():
    metadata = {'source': {'refs': ['a']}}
    state = initialize()
    e = evidence(state, provenance=metadata)
    metadata['source']['refs'].append('b')
    assert e.provenance['source']['refs'] == ('a',)
    result = update(state, e)
    with pytest.raises(FrozenInstanceError):
        result.update_status = 'true'
    with pytest.raises(TypeError):
        e.likelihood_by_hypothesis[state.source_future_ids[0]] = 0
    with pytest.raises(TypeError):
        result.provenance['prior_source']['source_id'] = 'changed'
    with pytest.raises(FrozenInstanceError):
        result.beliefs[0].posterior_probability = 1
    assert json.loads(result.to_json()) == result.to_dict()
    assert json.loads(e.to_json()) == e.to_dict()


@pytest.mark.parametrize('metadata', [{'x': 'a'*16385}, {'x': object()}, {'x': float('inf')}])
def test_bad_metadata_rejected(metadata):
    with pytest.raises(ValueError):
        evidence(initialize(), provenance=metadata)


def test_deterministic_ids_and_permutations():
    source = futures()
    state = initialize(source)
    permuted = replace(source, branches=tuple(reversed(source.branches)),
        branch_relations=tuple(reversed(source.branch_relations)))
    assert initialize(permuted).to_json() == state.to_json()
    first = update(state)
    e = evidence(state)
    e = replace(e, likelihood_by_hypothesis=dict(reversed(list(e.likelihood_by_hypothesis.items()))))
    assert update(state, e).to_json() == first.to_json()
    assert replace(first, beliefs=tuple(reversed(first.beliefs))).to_json() == first.to_json()
    assert update(state, replace(e, evidence_id='different')).belief_state_id != first.belief_state_id


def test_no_truth_winner_observation_or_new_actor():
    state, constraints = discontinuity()
    source = MultipleFuturesBuilder().build(state, constraints)
    result = update(initialize(source))
    for key in ('winner', 'best', 'truth', 'hidden_actor', 'observed', 'supports', 'contradicts'):
        assert key not in result.to_dict()
        assert all(key not in b.to_dict() for b in result.beliefs)
    assert set(result.source_future_ids) == {b.future_id for b in source.branches}
    assert all(b.source_future.object_ids == ('o0',) for b in result.beliefs)
    assert result.provenance['truth_decision'] == 'not_performed'


def test_learned_signal_and_labels_cannot_supply_probabilities():
    physical = moving()
    signal = LearnedRepresentationSignal('synthetic', 2., 'session', 'scene2.0', 'rep', 8,
        {'claim': 'high confidence hidden actor'}, temporal_change_score=100.)
    a = MultipleFuturesBuilder().build(physical)
    b = MultipleFuturesBuilder().build(HybridWorldState(physical, learned_signal=signal))
    assert initialize(a).to_json() == initialize(b).to_json()
    state = BayesianBeliefStateBuilder().initialize(b)
    assert posterior(state) == (None, None)


def test_process_determinism_and_standard_library_execution():
    script = '''
import sys
from evaluation.tests.test_bayesian_belief_state import initialize, update
print(update(initialize()).to_json())
assert not {'torch', 'transformers', 'numpy', 'pymc', 'pyro', 'cv2'} & set(sys.modules)
'''
    outputs = [subprocess.run([sys.executable, '-c', script], check=True, capture_output=True,
        text=True, env={**os.environ, 'PYTHONHASHSEED': seed}).stdout for seed in ('1', '42')]
    assert outputs[0] == outputs[1]


def test_caller_decimal_context_does_not_change_results():
    state = initialize()
    expected = update(state).to_json()
    with localcontext() as ctx:
        ctx.prec = 4
        ctx.rounding = ROUND_DOWN
        ctx.Emin = -9
        ctx.Emax = 9
        assert update(state).to_json() == expected


def test_unknown_source_time_cannot_be_repaired_by_advancing_cutoff():
    source = MultipleFuturesBuilder().build(replace(moving(), timestamp=None))
    state = initialize(source)
    state = BayesianBeliefStateBuilder().update(state, as_of_timestamp=3.)
    result = update(state, evidence(state, timestamp=4.), as_of_timestamp=4.)
    assert result.update_status == 'unavailable'
    assert result.provenance['update_reason'] == 'temporal_alignment_unknown'


def test_direct_updated_state_rejects_inconsistent_math():
    state = update(initialize())
    with pytest.raises(ValueError):
        replace(state, normalization_constant=.9)
    wrong = tuple(replace(b, posterior_probability=.5) for b in state.beliefs)
    with pytest.raises(ValueError):
        replace(state, beliefs=wrong)
    wrong = tuple(replace(b, likelihood=.5) for b in state.beliefs)
    with pytest.raises(ValueError):
        replace(state, beliefs=wrong)


def test_partial_distribution_rejected():
    state = initialize()
    bad = replace(state.beliefs[0], prior_probability=None, posterior_probability=None,
                  update_status='unavailable')
    with pytest.raises(ValueError):
        replace(state, update_status='unavailable', beliefs=(bad, replace(state.beliefs[1],
            posterior_probability=None, update_status='unavailable')))


def test_duplicate_and_reversed_event_lineage_rejected():
    first = update(initialize())
    second = update(first, evidence(first, evidence_id='event2', timestamp=3.), as_of_timestamp=3.)
    with pytest.raises(ValueError):
        replace(second, evidence_lineage=tuple(reversed(second.evidence_lineage)))
    with pytest.raises(ValueError):
        replace(second, evidence_lineage=(second.evidence_lineage[0],)*2)


def test_both_missing_frames_follow_exact_context_policy():
    source = futures()
    # Use minimal provenance because this fixture otherwise declares the old frame recursively.
    branches = tuple(replace(b, coordinate_frame_id=None, provenance={'source': 'undesignated-frame'})
                     for b in source.branches)
    source = replace(source, coordinate_frame_id=None, branches=branches, provenance={'source': 'test'})
    state = initialize(source)
    assert update(state).coordinate_frame_id is None


def test_tiny_positive_likelihood_normalization_stays_positive():
    state = initialize(values=(.5, .5))
    result = update(state, evidence(state, (5e-324, 5e-324)))
    assert result.normalization_constant == 5e-324
    assert posterior(result) == (.5, .5)


def test_source_and_probability_inputs_not_mutated():
    source = futures()
    before = source.to_json()
    priors = {i: .5 for i in sorted(b.future_id for b in source.branches)}
    metadata = {'source': {'id': 'caller'}}
    state = BayesianBeliefStateBuilder().initialize(source, priors=priors, prior_provenance=metadata)
    metadata['source']['id'] = 'changed'
    priors[next(iter(priors))] = 0
    assert posterior(state) == (.5, .5)
    assert state.provenance['prior_source']['source']['id'] == 'caller'
    before_state = state.to_json()
    update(state)
    assert state.to_json() == before_state and source.to_json() == before


def test_no_semantic_label_probability_shortcut():
    source = futures()
    modified = replace(source, branches=tuple(replace(b, provenance={'class_name': 'glass',
        'description': 'impact definitely implies a hidden person', 'confidence': 1.}) for b in source.branches))
    result = BayesianBeliefStateBuilder().initialize(modified)
    assert posterior(result) == (None, None)
