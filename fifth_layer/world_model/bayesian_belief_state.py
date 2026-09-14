"""Caller-supplied categorical belief updates, not calibrated physical truth."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from enum import Enum
from math import fsum, isclose
import json
from types import SimpleNamespace

from ._structured import freeze, identifier, number
from .common_evidence_state import bounded_plain
from .evidence import stable_id
from .hybrid_world_state import _alignment
from .multiple_futures import MultipleFutureBundle, FutureStateCandidate


VERSION = 'bayesian-belief-state-v0.1'
TOLERANCE = 1e-12
_CONTEXT = ('scene_id', 'session_id', 'coordinate_frame_id')
_LIMITS = ('posterior_not_truth', 'normalization_not_calibration',
           'branch_labels_not_disjoint_real_world_events', 'supplied_models_not_verified')


class BeliefInitializationMode(str, Enum):
    EXPLICIT_PRIOR = 'explicit_prior'
    UNIFORM_UNINFORMATIVE = 'uniform_uninformative'
    UNAVAILABLE = 'unavailable'


class BeliefUpdateStatus(str, Enum):
    INITIALIZED = 'initialized'
    UPDATED = 'updated'
    UNCHANGED = 'unchanged'
    UNAVAILABLE = 'unavailable'
    INDETERMINATE = 'indeterminate'
    INVALID_EVIDENCE = 'invalid_evidence'


class LikelihoodEvidenceStatus(str, Enum):
    SUPPLIED = 'supplied'
    UNAVAILABLE = 'unavailable'
    INVALID = 'invalid'


def _enum(cls, value):
    try:
        return cls(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'invalid {cls.__name__}') from exc


def _strings(values, name):
    if not isinstance(values, (tuple, list)):
        raise ValueError(f'{name} requires an ordered sequence')
    for value in values:
        identifier(value, name)
    if len(set(values)) != len(values):
        raise ValueError(f'duplicate {name}')
    return tuple(sorted(values))


def _context(record):
    for name in _CONTEXT:
        identifier(getattr(record, name), name, optional=name == 'coordinate_frame_id')
    if record.timestamp is not None:
        number(record.timestamp, 'timestamp', nonnegative=True)


def _metadata(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError('nonempty structured provenance required')
    return freeze(bounded_plain(value))


def _probabilities(values):
    if not isinstance(values, Mapping):
        raise ValueError('probabilities require a mapping')
    result = {}
    for key, value in values.items():
        identifier(key, 'hypothesis ID')
        number(value, 'probability', unit=True)
        result[key] = float(value)
    return freeze(dict(sorted(result.items())))


def _distribution(values):
    if not values or not isclose(fsum(values), 1., rel_tol=0., abs_tol=TOLERANCE):
        raise ValueError('distribution must sum to one within 1e-12')


def _bayes(prior, likelihoods):
    # Fixed local context protects tiny products and isolates caller Decimal settings.
    with localcontext(Context(prec=80, rounding=ROUND_HALF_EVEN)):
        weights = {i: Decimal.from_float(prior[i]) * Decimal.from_float(likelihoods[i]) for i in sorted(prior)}
        total = sum(weights.values(), Decimal(0))
        normalizer = float(total)
        if total == 0:
            return {}, 0., 'zero_normalization'
        if normalizer == 0:
            return {}, None, 'normalization_not_representable'
        posterior = {i: float(weights[i] / total) for i in weights}
        if any(weights[i] > 0 and posterior[i] == 0 for i in weights):
            return {}, normalizer, 'posterior_not_representable'
        return posterior, normalizer, 'explicit_bayesian_update'


class _Serializable:
    def to_dict(self):
        return bounded_plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class LikelihoodEvidence(_Serializable):
    evidence_id: str
    scene_id: str
    timestamp: float | None
    session_id: str
    coordinate_frame_id: str | None
    likelihood_by_hypothesis: Mapping
    provenance: Mapping
    evidence_type: str = 'caller_supplied_likelihood_model'
    status: LikelihoodEvidenceStatus = LikelihoodEvidenceStatus.SUPPLIED
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        _context(self)
        identifier(self.evidence_id, 'evidence_id')
        identifier(self.evidence_type, 'evidence_type')
        object.__setattr__(self, 'status', _enum(LikelihoodEvidenceStatus, self.status))
        object.__setattr__(self, 'likelihood_by_hypothesis', _probabilities(self.likelihood_by_hypothesis))
        if self.status != 'supplied' and self.likelihood_by_hypothesis:
            raise ValueError('unavailable/invalid evidence cannot contain usable likelihoods')
        object.__setattr__(self, 'provenance', _metadata(self.provenance))
        _alignment(bounded_plain(self.provenance), self)
        bounded_plain(self)


@dataclass(frozen=True)
class HypothesisBelief(_Serializable):
    hypothesis_id: str
    scene_id: str
    timestamp: float | None
    session_id: str
    coordinate_frame_id: str | None
    source_future: FutureStateCandidate
    prior_probability: float | None
    posterior_probability: float | None
    likelihood: float | None
    update_status: BeliefUpdateStatus
    source_evidence_ids: tuple[str, ...]
    provenance: Mapping
    confidence: float | None = None
    source_future_id: str = field(init=False)
    assumptions: tuple[str, ...] = field(init=False)
    schema_version: str = field(default=VERSION, init=False)
    rule_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        _context(self)
        identifier(self.hypothesis_id, 'hypothesis_id')
        if not isinstance(self.source_future, FutureStateCandidate):
            raise ValueError('hypothesis requires a Step 18 future')
        if self.hypothesis_id != self.source_future.future_id:
            raise ValueError('hypothesis ID must equal source future ID')
        if any(getattr(self.source_future, n) != getattr(self, n) for n in _CONTEXT):
            raise ValueError('future and belief context mismatch')
        _alignment(bounded_plain(self.source_future), self)
        object.__setattr__(self, 'source_future_id', self.source_future.future_id)
        object.__setattr__(self, 'assumptions', tuple(a.assumption_id for a in self.source_future.assumptions))
        object.__setattr__(self, 'update_status', _enum(BeliefUpdateStatus, self.update_status))
        for name in ('prior_probability', 'posterior_probability', 'likelihood'):
            value = getattr(self, name)
            if value is not None:
                number(value, name, unit=True)
                object.__setattr__(self, name, float(value))
        if self.confidence is not None:
            raise ValueError('confidence is not a probability surrogate')
        if self.update_status == 'updated' and any(getattr(self, n) is None for n in
                ('prior_probability', 'posterior_probability', 'likelihood')):
            raise ValueError('updated beliefs require all Bayesian terms')
        if self.update_status in ('initialized', 'unchanged') and (
                self.prior_probability is None or self.posterior_probability != self.prior_probability):
            raise ValueError('initialized/unchanged beliefs must retain their prior')
        if self.update_status in ('unavailable', 'indeterminate', 'invalid_evidence') and self.posterior_probability is not None:
            raise ValueError('unsuccessful update cannot fabricate a posterior')
        object.__setattr__(self, 'source_evidence_ids', _strings(self.source_evidence_ids, 'source_evidence_ids'))
        object.__setattr__(self, 'provenance', _metadata(self.provenance))
        bounded_plain(self)


@dataclass(frozen=True)
class BayesianBeliefState(_Serializable):
    belief_state_id: str
    scene_id: str
    timestamp: float | None
    session_id: str
    coordinate_frame_id: str | None
    source_multiple_futures_id: str
    source_future_ids: tuple[str, ...]
    beliefs: tuple[HypothesisBelief, ...]
    initialization_mode: BeliefInitializationMode
    update_status: BeliefUpdateStatus
    normalization_constant: float | None
    evidence_event_id: str | None
    previous_belief_state_id: str | None
    evidence_lineage: tuple[LikelihoodEvidence, ...]
    update_index: int
    uncertainty: tuple[str, ...]
    provenance: Mapping
    source_multiple_futures_schema: str = field(default='multiple-futures-0.2', init=False)
    probability_scope: str = field(default='categorical_branch_labels_not_event_frequencies', init=False)
    schema_version: str = field(default=VERSION, init=False)
    rule_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        _context(self)
        for name in ('belief_state_id', 'source_multiple_futures_id', 'evidence_event_id', 'previous_belief_state_id'):
            identifier(getattr(self, name), name, optional=name in ('evidence_event_id', 'previous_belief_state_id'))
        for name, cls in (('initialization_mode', BeliefInitializationMode), ('update_status', BeliefUpdateStatus)):
            object.__setattr__(self, name, _enum(cls, getattr(self, name)))
        object.__setattr__(self, 'source_future_ids', _strings(self.source_future_ids, 'source_future_ids'))
        if not isinstance(self.beliefs, (tuple, list)) or any(not isinstance(b, HypothesisBelief) for b in self.beliefs):
            raise ValueError('beliefs requires HypothesisBelief records')
        ids = [b.hypothesis_id for b in self.beliefs]
        if len(set(ids)) != len(ids) or set(ids) != set(self.source_future_ids):
            raise ValueError('duplicate or unknown/missing future hypothesis IDs')
        for b in self.beliefs:
            if any(getattr(b, n) != getattr(self, n) for n in _CONTEXT + ('timestamp', 'update_status')):
                raise ValueError('hypothesis must match belief state context and status')
        object.__setattr__(self, 'beliefs', tuple(sorted(self.beliefs, key=lambda b: b.hypothesis_id)))
        for name in ('prior_probability', 'posterior_probability'):
            values = [getattr(b, name) for b in self.beliefs]
            if any(v is not None for v in values):
                if any(v is None for v in values):
                    raise ValueError('partial probability distribution')
                _distribution(values)
        if not self.beliefs and self.update_status != 'unavailable':
            raise ValueError('empty hypothesis space is unavailable')
        if self.initialization_mode == 'unavailable' and any(b.prior_probability is not None for b in self.beliefs):
            raise ValueError('uninitialized belief cannot contain probabilities')
        if self.normalization_constant is not None:
            number(self.normalization_constant, 'normalization_constant', nonnegative=True)
        if self.update_status == 'updated':
            if self.normalization_constant is None or self.normalization_constant <= 0 or self.evidence_event_id is None:
                raise ValueError('updated state requires positive normalization and evidence event')
        if type(self.update_index) is not int or self.update_index < 0:
            raise ValueError('update_index must be a nonnegative integer')
        if not isinstance(self.evidence_lineage, (tuple, list)) or any(
                not isinstance(e, LikelihoodEvidence) for e in self.evidence_lineage):
            raise ValueError('evidence_lineage requires explicit likelihood records')
        events = [e.evidence_id for e in self.evidence_lineage]
        if len(set(events)) != len(events) or len(events) != self.update_index:
            raise ValueError('duplicate evidence event or inconsistent update index')
        previous_time = None
        for e in self.evidence_lineage:
            if e.status != 'supplied' or set(e.likelihood_by_hypothesis) != set(ids):
                raise ValueError('applied evidence requires complete supplied likelihoods')
            if any(getattr(e, n) != getattr(self, n) for n in _CONTEXT):
                raise ValueError('evidence lineage context mismatch')
            if e.timestamp is None or (previous_time is not None and e.timestamp < previous_time):
                raise ValueError('evidence lineage must have ordered known timestamps')
            previous_time = e.timestamp
            _alignment(e.to_dict(), self)
        if any(set(b.source_evidence_ids) != set(events) for b in self.beliefs):
            raise ValueError('hypothesis evidence references must match applied lineage')
        if self.update_status == 'updated' and (not events or events[-1] != self.evidence_event_id):
            raise ValueError('updated state must end with the applied event')
        if self.update_status == 'updated':
            last = self.evidence_lineage[-1]
            if any(b.likelihood != last.likelihood_by_hypothesis[b.hypothesis_id] for b in self.beliefs):
                raise ValueError('likelihood terms must match applied evidence')
            expected, normalizer, _ = _bayes({b.hypothesis_id: b.prior_probability for b in self.beliefs},
                                             last.likelihood_by_hypothesis)
            if normalizer != self.normalization_constant or any(
                    b.posterior_probability != expected.get(b.hypothesis_id) for b in self.beliefs):
                raise ValueError('posterior or normalization does not match supplied Bayesian terms')
        object.__setattr__(self, 'evidence_lineage', tuple(self.evidence_lineage))
        object.__setattr__(self, 'uncertainty', _strings(self.uncertainty, 'uncertainty'))
        object.__setattr__(self, 'provenance', _metadata(self.provenance))
        _alignment(bounded_plain(self.provenance), self)
        bounded_plain(self)


def _state(**values):
    return BayesianBeliefState(stable_id('belief-state', VERSION, bounded_plain(values)), **values)


class BayesianBeliefStateBuilder:
    """All branch statuses remain eligible. Probabilities concern categorical labels.

    Numerical priors/likelihoods are supplied models, never inferred from content.
    """

    def initialize(self, futures, *, priors=None, prior_provenance=None, initialization_mode=None):
        if not isinstance(futures, MultipleFutureBundle) or futures.schema_version != 'multiple-futures-0.2':
            raise ValueError('expected MultipleFutureBundle v0.2')
        data = bounded_plain(futures)
        _alignment(data, futures)
        ids = tuple(sorted(b.future_id for b in futures.branches))
        mode = _enum(BeliefInitializationMode, initialization_mode if initialization_mode is not None else
                     ('explicit_prior' if priors is not None else 'unavailable'))
        if mode == 'explicit_prior':
            priors = _probabilities(priors)
            if set(priors) != set(ids):
                raise ValueError('explicit prior must cover exactly all source future IDs')
            _distribution(list(priors.values()))
            prior_source = _metadata(prior_provenance)
        else:
            if priors is not None or prior_provenance is not None:
                raise ValueError('prior mapping/provenance requires explicit_prior mode')
            priors = {i: 1. / len(ids) for i in ids} if mode == 'uniform_uninformative' else {}
            prior_source = {'mode': mode.value, 'meaning': 'no label preferred by initialization; not empirical frequency'}
        status = 'initialized' if mode != 'unavailable' and ids else 'unavailable'
        context = {n: getattr(futures, n) for n in _CONTEXT + ('timestamp',)}
        beliefs = tuple(HypothesisBelief(b.future_id, **context, source_future=b,
            prior_probability=priors.get(b.future_id), posterior_probability=priors.get(b.future_id),
            likelihood=None, update_status=status, source_evidence_ids=(),
            provenance={'prior_source': prior_source, 'source_future_schema': b.schema_version}) for b in futures.branches)
        return _state(**context, source_multiple_futures_id=stable_id('multiple-futures-source', data),
            source_future_ids=ids, beliefs=beliefs, initialization_mode=mode, update_status=status,
            normalization_constant=None, evidence_event_id=None, previous_belief_state_id=None,
            evidence_lineage=(), update_index=0, uncertainty=tuple(sorted(set(futures.uncertainty) | set(_LIMITS)
                | ({'prior_unavailable'} if status == 'unavailable' else set()))),
            provenance={'prior_source': prior_source, 'initial_prior_by_hypothesis': priors,
                'source_bundle_provenance': futures.provenance,
                'source_branch_relations': bounded_plain(futures.branch_relations),
                'source_timestamp': futures.timestamp, 'eligibility': 'all_branch_statuses_retained',
                'probability_scope': 'categorical_branch_labels_not_event_frequencies',
                'truth_decision': 'not_performed', 'rule_version': VERSION})

    def update(self, previous, evidence=None, *, as_of_timestamp=None):
        if not isinstance(previous, BayesianBeliefState):
            raise ValueError('expected BayesianBeliefState v0.1')
        cutoff = previous.timestamp if as_of_timestamp is None else as_of_timestamp
        if cutoff is not None:
            number(cutoff, 'as_of_timestamp', nonnegative=True)
        if previous.timestamp is not None and (cutoff is None or cutoff < previous.timestamp):
            raise ValueError('update cutoff cannot precede previous belief state')
        ids = previous.source_future_ids
        prior = {b.hypothesis_id: b.posterior_probability if b.posterior_probability is not None else b.prior_probability
                 for b in previous.beliefs}
        known = bool(prior) and all(p is not None for p in prior.values())
        status, reason = ('unchanged', 'missing_evidence') if known else ('unavailable', 'prior_unavailable')
        likelihoods, posterior, normalizer = {}, dict(prior) if known else {}, None
        lineage = previous.evidence_lineage
        if evidence is not None:
            if not isinstance(evidence, LikelihoodEvidence):
                raise ValueError('expected LikelihoodEvidence')
            if any(getattr(evidence, n) != getattr(previous, n) for n in _CONTEXT):
                raise ValueError('likelihood evidence context mismatch')
            if evidence.evidence_id in {e.evidence_id for e in lineage}:
                raise ValueError('evidence event already applied; supply a new event')
            if set(evidence.likelihood_by_hypothesis) - set(ids):
                raise ValueError('unknown likelihood hypothesis ID')
            _alignment(evidence.to_dict(), SimpleNamespace(timestamp=cutoff,
                session_id=previous.session_id, coordinate_frame_id=previous.coordinate_frame_id))
            if evidence.timestamp is not None and previous.timestamp is not None and evidence.timestamp < previous.timestamp:
                raise ValueError('evidence predates previous belief cutoff')
            likelihoods = evidence.likelihood_by_hypothesis
            posterior = {}
            if evidence.status == 'invalid':
                status, reason = 'invalid_evidence', 'caller_marked_invalid'
            elif evidence.status == 'unavailable':
                status, reason = ('unchanged', 'evidence_unavailable') if known else ('unavailable', 'prior_unavailable')
                posterior = dict(prior) if known else {}
            elif not known:
                status, reason = 'unavailable', 'prior_unavailable'
            elif evidence.timestamp is None or previous.timestamp is None or previous.provenance.get('source_timestamp') is None:
                status, reason = 'unavailable', 'temporal_alignment_unknown'
            elif set(likelihoods) != set(ids):
                status, reason = 'unavailable', 'incomplete_likelihoods_no_update'
            else:
                posterior, normalizer, reason = _bayes(prior, likelihoods)
                status = 'updated' if posterior else 'indeterminate'
                if posterior:
                    lineage += (evidence,)
        context = {n: getattr(previous, n) for n in _CONTEXT}
        context['timestamp'] = cutoff
        beliefs = tuple(HypothesisBelief(b.hypothesis_id, **context, source_future=b.source_future,
            prior_probability=prior[b.hypothesis_id], posterior_probability=posterior.get(b.hypothesis_id),
            likelihood=likelihoods.get(b.hypothesis_id), update_status=status,
            source_evidence_ids=tuple(e.evidence_id for e in lineage),
            provenance={'prior_source': {'previous_belief_state_id': previous.belief_state_id,
                'probability_field': 'posterior_probability' if b.posterior_probability is not None else 'prior_probability'},
                'likelihood_source': None if evidence is None else evidence.to_dict(),
                'update_reason': reason}) for b in previous.beliefs)
        return _state(**context, source_multiple_futures_id=previous.source_multiple_futures_id,
            source_future_ids=ids, beliefs=beliefs, initialization_mode=previous.initialization_mode,
            update_status=status, normalization_constant=normalizer,
            evidence_event_id=None if evidence is None else evidence.evidence_id,
            previous_belief_state_id=previous.belief_state_id, evidence_lineage=lineage,
            update_index=len(lineage), uncertainty=tuple(sorted(set(previous.uncertainty) | set(_LIMITS)
                | ({reason} if status not in ('initialized', 'updated', 'unchanged') else set()))),
            provenance={**previous.provenance, 'previous_belief_state_id': previous.belief_state_id,
                'update_reason': reason, 'likelihood_source': None if evidence is None else evidence.to_dict(),
                'likelihood_interpretation': 'caller_supplied_conditional_on_previously_applied_evidence',
                'missing_likelihood_policy': 'require_all_hypotheses', 'rule_version': VERSION})
