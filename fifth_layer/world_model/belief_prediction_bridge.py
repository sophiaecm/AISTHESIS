"""Explicit branch issuance and plumbing into the frozen observation evaluator."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isclose
import json
from types import SimpleNamespace

from ._structured import freeze, identifier, number
from .bayesian_belief_state import BayesianBeliefState, HypothesisBelief
from .common_evidence_state import bounded_plain
from .evidence import stable_id
from .experience_memory import ExperienceEpisode
from .hybrid_world_state import _alignment
from .latent_physical_state import LatentPhysicalState
from .prediction_records import PredictionRecord, OutcomeRecord, PredictionEvaluation
from . import prediction_observation as observation
from .scene_state import SceneState


VERSION = 'belief-prediction-bridge-v0.1'


def _metadata(value):
    if not isinstance(value, Mapping) or not value:
        raise ValueError('nonempty structured provenance required')
    return freeze(bounded_plain(value))


def _fields(values):
    if not isinstance(values, (tuple, list)):
        raise ValueError('field names require an ordered sequence')
    for value in values:
        identifier(value, 'field')
    return tuple(sorted(set(values)))


def _mapping(future):
    predicted, unavailable = {}, set()
    for change in future.predicted_changes:
        if (change['field'], change['value'], change['epistemic_status']) == (
                'motion_state', 'moving', 'branch_assumption'):
            predicted['motion_state'] = 'moving'
        else:
            unavailable.add(change['field'])
    return predicted, tuple(sorted(unavailable))


class _Serializable:
    def to_dict(self):
        return bounded_plain(self)

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class BeliefPredictionIssue(_Serializable):
    issue_id: str
    belief_state_id: str
    source_belief: HypothesisBelief
    prediction: PredictionRecord | None
    issue_status: str
    evaluable_fields: tuple[str, ...]
    unevaluable_fields: tuple[str, ...]
    target_timestamp: float | None
    provenance: Mapping
    hypothesis_id: str = field(init=False)
    source_future_id: str = field(init=False)
    source_posterior_probability: float | None = field(init=False)
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        identifier(self.issue_id, 'issue_id')
        identifier(self.belief_state_id, 'belief_state_id')
        if not isinstance(self.source_belief, HypothesisBelief):
            raise ValueError('issue requires HypothesisBelief')
        source = self.source_belief
        object.__setattr__(self, 'hypothesis_id', source.hypothesis_id)
        object.__setattr__(self, 'source_future_id', source.source_future_id)
        object.__setattr__(self, 'source_posterior_probability', source.posterior_probability)
        if self.issue_status not in ('issued', 'unavailable', 'insufficient_prediction_content'):
            raise ValueError('invalid issue status')
        if self.target_timestamp is not None:
            number(self.target_timestamp, 'target_timestamp', nonnegative=True)
            if source.timestamp is not None and self.target_timestamp <= source.timestamp:
                raise ValueError('target must follow the belief source time')
        for name in ('evaluable_fields', 'unevaluable_fields'):
            object.__setattr__(self, name, _fields(getattr(self, name)))
        mapped, unavailable = _mapping(source.source_future)
        if self.evaluable_fields != tuple(sorted(mapped)) or self.unevaluable_fields != unavailable:
            raise ValueError('field inventory must reflect the actual source changes')
        if (self.prediction is not None) != (self.issue_status == 'issued'):
            raise ValueError('only issued records contain predictions')
        if self.prediction is not None:
            p = self.prediction
            if not isinstance(p, PredictionRecord):
                raise ValueError('expected existing PredictionRecord')
            if (not mapped or source.timestamp is None or self.target_timestamp is None or
                    p.hypothesis_id != self.hypothesis_id or p.scene_id != source.scene_id or
                    p.source_timestamp != source.timestamp or p.target_timestamp != self.target_timestamp or
                    p.horizon_seconds <= 0 or dict(p.predicted_state) != mapped):
                raise ValueError('prediction does not match issued source, fields or times')
            if p.confidence is not None or p.uncertainty is not None or p.evidence_for or p.evidence_against:
                raise ValueError('bridge cannot promote probabilities or assumptions to confidence/evidence')
            if p.provenance.get('issue_id') != self.issue_id or p.provenance.get('belief_state_id') != self.belief_state_id:
                raise ValueError('prediction provenance must link to issue and belief')
        object.__setattr__(self, 'provenance', _metadata(self.provenance))
        if self.prediction is not None:
            p = self.prediction
            expected_track = self.provenance.get('association', {}).get('track_id')
            if type(p.track_id) is not type(expected_track) or p.track_id != expected_track or p.object_id is not None:
                raise ValueError('prediction must retain attributed track binding only')
            if p.hypothesis_type != 'continued_motion' or p.trajectory_id is not None or p.image_size is not None:
                raise ValueError('bridge does not issue additional event or geometry claims')
            _alignment(bounded_plain(p.provenance), source)
        _alignment(bounded_plain(source), source)
        _alignment(bounded_plain(self.provenance), source)
        bounded_plain(self)


def issue_prediction(belief_state, *, hypothesis_id, target_timestamp=None,
                     horizon_seconds=None, physical_state=None):
    """Select only the named branch. Numeric timing is explicitly caller supplied.

    Optional original latent state provides an attributed tracker identity. Opaque
    physical object IDs are never parsed or used as frame-local observation IDs.
    """
    if not isinstance(belief_state, BayesianBeliefState) or belief_state.schema_version != 'bayesian-belief-state-v0.1':
        raise ValueError('expected BayesianBeliefState v0.1')
    identifier(hypothesis_id, 'hypothesis_id')
    source = next((b for b in belief_state.beliefs if b.hypothesis_id == hypothesis_id), None)
    if source is None:
        raise ValueError('unknown hypothesis_id')
    _alignment(bounded_plain(belief_state), belief_state)
    future = source.source_future
    stamp = belief_state.timestamp
    for name, value in (('target_timestamp', target_timestamp), ('horizon_seconds', horizon_seconds)):
        if value is not None:
            number(value, name, nonnegative=True)
    if horizon_seconds is not None and horizon_seconds <= 0:
        raise ValueError('horizon_seconds must be positive')
    target = target_timestamp
    if stamp is not None:
        if target is None and horizon_seconds is not None:
            target = stamp + horizon_seconds
            number(target, 'target_timestamp', nonnegative=True)
        if target is not None and target <= stamp:
            raise ValueError('target must be strictly later than source')
        if target is not None and horizon_seconds is not None and not isclose(
                stamp + horizon_seconds, target, rel_tol=0., abs_tol=1e-9):
            raise ValueError('explicit target and horizon disagree')
    track_id, association = None, {'policy': 'unbound_without_attributed_track_identity'}
    if physical_state is not None:
        if not isinstance(physical_state, LatentPhysicalState) or physical_state.schema_version != 'latent-physical-state-0.1':
            raise ValueError('expected original LatentPhysicalState v0.1')
        if (physical_state.latent_state_id != future.source_physical_state_id or
                physical_state.timestamp != future.timestamp or any(getattr(physical_state, n) != getattr(source, n)
                for n in ('scene_id', 'session_id', 'coordinate_frame_id'))):
            raise ValueError('association physical state must match original future source')
        _alignment(bounded_plain(physical_state), source)
        objects = {o['physical_object_id']: o for o in physical_state.objects}
        if not set(future.object_ids) <= objects.keys():
            raise ValueError('branch physical endpoints absent from source state')
        if len(future.object_ids) == 1:
            attr = objects[future.object_ids[0]].get('attributes', {}).get('track_id', {})
            value = attr.get('value')
            if (type(value) in (int, str) and attr.get('status') in ('observed', 'estimated')
                    and attr.get('derived_from')):
                # Do not select a tracker ID shared by multiple physical objects.
                same = [o for o in physical_state.objects if type(o.get('attributes', {}).get('track_id', {}).get('value'))
                        is type(value) and o['attributes']['track_id']['value'] == value]
                if len(same) == 1:
                    track_id = value
                    association = {'policy': 'attributed_track_id_from_original_latent_object',
                        'physical_object_id': future.object_ids[0], 'track_id': value,
                        'source_timestamp': physical_state.timestamp, 'source_refs': attr['derived_from']}
                else:
                    association = {'policy': 'unbound_ambiguous_source_track_identity'}
    mapped, unavailable = _mapping(future)
    if stamp is None or future.timestamp is None or target is None:
        status, reason = 'unavailable', 'known_source_and_explicit_numeric_target_required'
    elif not mapped or len(future.object_ids) != 1:
        status, reason = 'insufficient_prediction_content', 'no_supported_single_object_prediction'
    else:
        status, reason = 'issued', 'explicit_branch_and_numeric_time'
    provenance = {'source_belief_provenance': source.provenance,
        'belief_state_provenance': belief_state.provenance,
        'belief_schema_version': belief_state.schema_version, 'future_schema_version': future.schema_version,
        'belief_update_status': belief_state.update_status.value, 'evidence_lineage': bounded_plain(belief_state.evidence_lineage),
        'source_multiple_futures_id': belief_state.source_multiple_futures_id,
        'association': association, 'reason': reason, 'selection_policy': 'explicit_hypothesis_id_only',
        'horizon_policy': 'caller_numeric_time_not_qualitative_horizon_conversion',
        'truth_decision': 'not_performed', 'rule_version': VERSION}
    identity = stable_id('belief-prediction-issue', VERSION, belief_state.belief_state_id,
        bounded_plain(source), target, mapped, unavailable, provenance)
    prediction = None
    if status == 'issued':
        # Legacy records validate scalar uncertainty. Preserve structured source
        # records in the bridge, and project compatible reference metadata here.
        links = {'issue_id': identity, 'belief_state_id': belief_state.belief_state_id,
            'hypothesis_id': hypothesis_id, 'source_future_id': future.future_id,
            'source_physical_state_id': future.source_physical_state_id,
            'session_id': source.session_id, 'coordinate_frame_id': source.coordinate_frame_id,
            'source_branch_status': future.status.value, 'source_branch_family': future.branch_family,
            'source_assumptions': bounded_plain(future.assumptions),
            'source_predicted_changes': bounded_plain(future.predicted_changes),
            'source_field_references': future.source_field_references,
            'source_constraint_ids': future.source_constraint_ids, 'source_consequence_ids': future.source_consequence_ids,
            'source_prior_probability': source.prior_probability, 'source_posterior_probability': source.posterior_probability,
            'source_likelihood': source.likelihood, 'evidence_lineage_ids': tuple(e.evidence_id for e in belief_state.evidence_lineage),
            'source_belief_schema': belief_state.schema_version, 'source_future_schema': future.schema_version,
            'source_horizon_kind': future.horizon_kind.value, 'source_horizon_value': future.horizon_value,
            'unevaluable_fields': unavailable, 'association': association,
            'horizon_policy': provenance['horizon_policy'], 'selection_policy': provenance['selection_policy'],
            'truth_decision': 'not_performed', 'bridge_schema_version': VERSION}
        values = dict(scene_id=source.scene_id, hypothesis_id=hypothesis_id, hypothesis_type='continued_motion',
            source_timestamp=stamp, target_timestamp=target, horizon_seconds=target-stamp,
            predicted_state=mapped, track_id=track_id, provenance=links)
        prediction = PredictionRecord(stable_id('prediction', identity, values), **values)
    return BeliefPredictionIssue(identity, belief_state.belief_state_id, source, prediction, status,
        tuple(sorted(mapped)), unavailable, target, provenance)


@dataclass(frozen=True)
class BeliefPredictionOutcome(_Serializable):
    issue: BeliefPredictionIssue
    outcome: OutcomeRecord
    evaluation: PredictionEvaluation
    experience_episode: ExperienceEpisode
    schema_version: str = field(default=VERSION, init=False)

    def __post_init__(self):
        if not isinstance(self.issue, BeliefPredictionIssue) or self.issue.prediction is None:
            raise ValueError('outcome requires an issued prediction')
        if not isinstance(self.outcome, OutcomeRecord) or not isinstance(self.evaluation, PredictionEvaluation):
            raise ValueError('existing outcome/evaluation records required')
        if not isinstance(self.experience_episode, ExperienceEpisode):
            raise ValueError('existing ExperienceEpisode required')
        p, o, e = self.issue.prediction, self.outcome, self.evaluation
        if o.prediction_id != p.prediction_id or e.prediction_id != p.prediction_id or e.outcome_id != o.outcome_id:
            raise ValueError('prediction/outcome/evaluation links must agree')
        if e.evaluated_timestamp != o.observation_timestamp:
            raise ValueError('evaluation must retain the observed timestamp')
        if o.observation_timestamp is not None and o.observation_timestamp <= p.source_timestamp:
            raise ValueError('observation must be later than prediction source')
        episode = self.experience_episode
        if (episode.prediction_summary != freeze(p) or episode.observation_summary != freeze(o)
                or episode.prediction_error != freeze(e) or episode.evaluation_status != e.status):
            raise ValueError('episode must preserve exact existing evaluation records')
        bounded_plain(self)

    @property
    def issue_id(self):
        return self.issue.issue_id

    @property
    def prediction(self):
        return self.issue.prediction

    @property
    def status(self):
        return self.evaluation.status

    @property
    def prediction_error(self):
        # The legacy container is mutable: return a detached instance on demand.
        return self.evaluation.as_prediction_error()


def evaluate_issue(issue, scene, *, session_id, coordinate_frame_id,
                   time_tolerance_seconds=0., position_tolerance_pixels=0.):
    """Reuse observation/evaluation/episode functions; no Bayesian or memory writes."""
    if not isinstance(issue, BeliefPredictionIssue) or issue.prediction is None:
        raise ValueError('only issued predictions can be evaluated')
    source = issue.source_belief
    if session_id != source.session_id or coordinate_frame_id != source.coordinate_frame_id:
        raise ValueError('outcome session/frame must match issuance')
    if scene is not None:
        if not isinstance(scene, SceneState) or scene.schema_version != '0.1':
            raise ValueError('expected later observed SceneState v0.1 or None')
        if scene.timestamp is None or scene.timestamp <= issue.prediction.source_timestamp:
            raise ValueError('new observation time must strictly follow prediction source')
        context = SimpleNamespace(timestamp=scene.timestamp, session_id=session_id, coordinate_frame_id=coordinate_frame_id)
        _alignment(bounded_plain(scene.provenance), context)
    outcome = observation.outcome_from_scene(issue.prediction, scene)
    if scene is not None:
        # Only admitted observed sources are checked; predicted_tracks remain unused.
        _alignment(bounded_plain(outcome), context)
    evaluation = observation.evaluate_prediction(issue.prediction, outcome,
        time_tolerance_seconds=time_tolerance_seconds, position_tolerance_pixels=position_tolerance_pixels)
    episode = observation.experience_episode(issue.prediction, outcome, evaluation)
    return BeliefPredictionOutcome(issue, outcome, evaluation, episode)
