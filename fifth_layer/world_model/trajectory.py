"""Future trajectory representation in existing center / xyxy bbox conventions."""
from dataclasses import dataclass, field
from typing import Any, Mapping

from ._structured import freeze_fields, geometry, identifier, number, timestamps
from .hypothesis import HypothesisSet


@dataclass(frozen=True)
class FutureTrajectory:
    trajectory_id: str
    hypothesis_id: str
    track_id: int | str | None
    start_timestamp: float
    horizon_seconds: float
    predicted_states: tuple = ()
    predicted_centers: tuple = ()
    predicted_bboxes: tuple = ()
    expected_event: str | None = None
    uncertainty_by_step: tuple = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        identifier(self.trajectory_id, 'trajectory_id')
        identifier(self.hypothesis_id, 'hypothesis_id')
        if self.track_id is not None and type(self.track_id) not in (int, str):
            raise ValueError('track_id must be an integer, string or None')
        if self.expected_event is not None:
            identifier(self.expected_event, 'expected_event')
        timestamps(self, ('start_timestamp', 'horizon_seconds'))
        freeze_fields(self, ('predicted_states', 'predicted_centers', 'predicted_bboxes',
                            'uncertainty_by_step', 'provenance'))
        for name in ('predicted_states', 'predicted_centers', 'predicted_bboxes', 'uncertainty_by_step'):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(f'{name} must be an ordered sequence')
        for center in self.predicted_centers:
            geometry(center, 'predicted_centers', 2)
        if any(not isinstance(state, Mapping) for state in self.predicted_states):
            raise ValueError('predicted_states must contain structured state summaries')
        for bbox in self.predicted_bboxes:
            geometry(bbox, 'predicted_bboxes', 4)
        for uncertainty in self.uncertainty_by_step:
            number(uncertainty, 'uncertainty_by_step', unit=True)
        lengths = {len(values) for values in (self.predicted_states, self.predicted_centers,
                   self.predicted_bboxes, self.uncertainty_by_step) if values}
        if len(lengths) > 1:
            raise ValueError('non-empty trajectory step sequences must have matching lengths')


def validate_trajectories(trajectories, hypotheses: HypothesisSet):
    """Return an immutable collection after validating IDs and local linkage."""
    result = tuple(trajectories)
    seen = set()
    known = {h.hypothesis_id for h in hypotheses.hypotheses}
    for trajectory in result:
        if not isinstance(trajectory, FutureTrajectory):
            raise ValueError('trajectories must contain FutureTrajectory instances')
        if trajectory.trajectory_id in seen:
            raise ValueError(f'duplicate trajectory_id: {trajectory.trajectory_id}')
        if trajectory.hypothesis_id not in known:
            raise ValueError(f'unknown hypothesis_id: {trajectory.hypothesis_id}')
        seen.add(trajectory.trajectory_id)
    return result
