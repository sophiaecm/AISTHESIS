"""Read-only sensory adapters using the existing EvidenceItem contract.

Only auditory expected-consequence outputs exist in this repository. No sensor
measurements are manufactured. Other modalities are coverage records only.
"""
from collections.abc import Mapping

from fifth_layer.expected_consequences import ExpectedConsequences
from .evidence import EvidenceItem, stable_id


MODALITIES = ('auditory', 'tactile', 'thermal', 'kinesthetic')


def unavailable_modality(scene, modality, *, timestamp=None):
    """Observation availability, independent of whether consequences are inferred."""
    stamp = scene.timestamp if timestamp is None else timestamp
    return EvidenceItem(
        stable_id('sensory-unavailable', scene.scene_id, modality, stamp),
        scene.scene_id, 'sensory', __name__, 'modality_availability',
        {'observation_available': False}, stamp,
        provenance={'reason': 'no sensory measurement supplied',
                    'adapter_version': 'sensory-0.1'},
        modality=modality, epistemic_status='unavailable')


class AuditoryEvidenceProvider:
    """Adapt ExpectedConsequences.predictions or its exact mapping projection.

    The legacy probability is retained as an uncalibrated heuristic score in
    value, never converted to confidence. Optional explicit confidence and
    metadata are preserved. Class names are not object/track identities.
    Latent/future summaries omit candidate details and are not substituted.
    """
    component = 'fifth_layer.reasoners.auditory.AuditoryReasoner'

    def provide(self, scene, output=None, *, timestamp=None):
        if output is None:
            return (unavailable_modality(scene, 'auditory', timestamp=timestamp),)
        origin = 'mapping'
        if isinstance(output, ExpectedConsequences):
            output = output.predictions
            origin = 'ExpectedConsequences.predictions'
        if not isinstance(output, Mapping):
            raise ValueError('auditory output must be expected consequences or their mapping')
        if output.get('auditory_modality_observed', False) is not False:
            raise ValueError('AuditoryReasoner does not produce observations')
        stamp = output.get('timestamp', scene.timestamp if timestamp is None else timestamp)
        items = [unavailable_modality(scene, 'auditory', timestamp=stamp)]
        records = output.get('auditory_consequences', ())
        if not isinstance(records, (list, tuple)):
            raise ValueError('auditory_consequences must be a sequence')
        for record in records:
            if not isinstance(record, Mapping) or not isinstance(record.get('consequence'), str):
                raise ValueError('auditory candidate requires a consequence')
            if record.get('observed', False) is not False or record.get(
                    'epistemic_status', 'expected') not in ('expected', 'inferred'):
                raise ValueError('auditory consequences cannot be observations')
            value = dict(record)
            value['observed'] = False
            # Evidence labels are an unordered set; preserve trajectories and other values.
            for key in ('evidence', 'opposing_evidence', 'supports', 'contradicts',
                        'supporting_evidence_ids', 'opposing_evidence_ids'):
                if key in value:
                    labels = value[key]
                    if not isinstance(labels, (list, tuple)) or any(not isinstance(x, str) for x in labels):
                        raise ValueError(f'{key} must be a sequence of source labels')
                    value[key] = tuple(sorted(set(labels)))
            candidate_stamp = record.get('timestamp', stamp)
            provenance = dict(source_field=f'{origin}.auditory_consequences[*]',
                source_timestamp=candidate_stamp, source_provenance=record.get('provenance', {}),
                scene_provenance=scene.provenance, adapter_version='sensory-0.1',
                confidence_policy='explicit source confidence only',
                probability_policy='legacy heuristic score; not calibrated',
                physics_policy='unverified consequence; requires physical anchors before hypothesis use')
            identity = stable_id('sensory', scene.scene_id, self.component, value, candidate_stamp, provenance)
            # No automatic links from a possible sound to a physical event.
            items.append(EvidenceItem(identity, scene.scene_id, 'sensory', self.component,
                'auditory_consequence', value, candidate_stamp, record.get('confidence'),
                supports=tuple(sorted(set(record.get('supports', ())))),
                contradicts=tuple(sorted(set(record.get('contradicts', ())))),
                track_id=record.get('track_id'), object_id=record.get('object_id'),
                provenance=provenance, modality='auditory',
                epistemic_status=record.get('epistemic_status', 'expected'),
                supporting_evidence_ids=tuple(sorted(set(record.get('supporting_evidence_ids', ())))),
                opposing_evidence_ids=tuple(sorted(set(record.get('opposing_evidence_ids', ()))))))
        return tuple(sorted({x.evidence_id: x for x in items}.values(), key=lambda x: x.evidence_id))


def collect_sensory_evidence(scene, *, auditory=None, timestamp=None):
    return (*AuditoryEvidenceProvider().provide(scene, auditory, timestamp=timestamp),
            *(unavailable_modality(scene, modality, timestamp=timestamp) for modality in MODALITIES[1:]))


def sensory_applies(item, kind, physical_items, *, opposing=False):
    """Only explicitly bound sensory evidence may annotate an existing candidate.

    Physical anchor IDs must resolve in the same association and issuance time.
    Inference cannot override explicit physical contradiction. Observed opposing
    evidence is accepted only when the caller explicitly supplies a hypothesis
    link; missing measurements have no links and never enter this function.
    """
    if item.epistemic_status == 'unavailable':
        return False
    links = item.contradicts if opposing else item.supports
    if kind not in links:
        return False
    if not any(x.timestamp == item.timestamp for x in physical_items):
        return False
    physical = [x for x in physical_items if x.source_type in ('physics', 'motion', 'temporal')
                and x.timestamp == item.timestamp]
    if opposing and item.epistemic_status == 'observed':
        return True
    if not opposing and any(kind in x.contradicts for x in physical):
        return False
    ids = item.opposing_evidence_ids if opposing else item.supporting_evidence_ids
    return any(x.evidence_id in ids and kind in (x.contradicts if opposing else x.supports)
               for x in physical)
