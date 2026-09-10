"""Isolated sensory contract and adversarial epistemic regression tests."""
from dataclasses import replace
from itertools import permutations
import unittest

from fifth_layer.world_state import WorldState
from fifth_layer.reasoners.auditory import AuditoryReasoner
from fifth_layer.world_model import (
    SceneState, EvidenceItem, EvidenceBundle, AuditoryEvidenceProvider,
    collect_evidence, MultiHypothesisGenerator, unavailable_modality,
)


def scene(**changes):
    return SceneState(scene_id='s', timestamp=10., **changes)


def sensory(**changes):
    values = dict(evidence_id='sensory', scene_id='s', source_type='sensory',
        source_component='test.sensor', evidence_type='measurement',
        value={'claim': 'explicit source claim'}, timestamp=10., confidence=.1,
        track_id=7, modality='auditory', epistemic_status='observed')
    values.update(changes)
    return EvidenceItem(**values)


def generate(current, *extra):
    base = collect_evidence(current)
    return MultiHypothesisGenerator().generate(current, EvidenceBundle('s', (*base.items, *extra)))


class SensoryTests(unittest.TestCase):
    def setUp(self):
        self.scene = scene(motion_evidence=[dict(track_id=7, motion_state='moving_right', confidence=.4)])
        self.anchor = collect_evidence(self.scene).items[0]
        self.provider = AuditoryEvidenceProvider()

    def candidate(self, **changes):
        record = dict(consequence='footsteps_possible', probability=.12,
                      uncertainty=.88, observed=False, evidence=['human_motion'])
        record.update(changes)
        return next(x for x in self.provider.provide(self.scene,
            {'auditory_consequences': [record]}) if x.epistemic_status == 'expected')

    def test_actual_auditory_output(self):
        output = AuditoryReasoner().infer_expected_consequences(WorldState(10., dict(
            motion_evidence=[dict(class_name='person', motion_state='moving_right')], detections=[])))
        items = self.provider.provide(self.scene, output)
        candidates = [x for x in items if x.epistemic_status == 'expected']
        self.assertEqual({x.value['consequence'] for x in candidates},
                         {'footsteps_possible', 'clothing_rustle_possible'})
        self.assertTrue(all(x.value['observed'] is False for x in candidates))

    def test_missing_modalities(self):
        items = [x for x in collect_evidence(scene(), include_sensory=True).items]
        self.assertEqual({x.modality for x in items}, {'auditory', 'tactile', 'thermal', 'kinesthetic'})
        for item in items:
            self.assertEqual(item.epistemic_status, 'unavailable')
            self.assertEqual(dict(item.value), {'observation_available': False})
            self.assertEqual(item.supports + item.contradicts, ())
            self.assertIsNone(item.confidence)

    def test_unavailable_is_not_silence_contact_or_heat_absence(self):
        for modality in ('auditory', 'tactile', 'thermal', 'kinesthetic'):
            item = unavailable_modality(self.scene, modality)
            self.assertEqual(generate(self.scene), generate(self.scene, item))
            for field in ('silence', 'no_contact', 'no_heat', 'no_motion'):
                self.assertNotIn(field, item.value)

    def test_inferred_contract_for_all_modalities_without_faking_reasoners(self):
        for modality in ('auditory', 'tactile', 'thermal', 'kinesthetic'):
            item = sensory(modality=modality, epistemic_status='inferred', value={'observed': False})
            self.assertEqual(item.epistemic_status, 'inferred')
            self.assertEqual(item.modality, modality)

    def test_status_required(self):
        with self.assertRaises(ValueError):
            sensory(epistemic_status=None)

    def test_observed_conflict_rejected(self):
        for status, flag in (('observed', False), ('inferred', True), ('expected', True), ('unavailable', True)):
            with self.assertRaises(ValueError):
                sensory(epistemic_status=status, value={'observed': flag})

    def test_provider_rejects_observed_record(self):
        with self.assertRaises(ValueError):
            self.candidate(observed=True)

    def test_provider_rejects_observed_envelope(self):
        with self.assertRaises(ValueError):
            self.provider.provide(self.scene, {'auditory_modality_observed': True})

    def test_provider_rejects_status_spoof(self):
        with self.assertRaises(ValueError):
            self.candidate(epistemic_status='observed')

    def test_inferred_status_preserved(self):
        items = self.provider.provide(self.scene, {'auditory_consequences': [
            {'consequence': 'possible', 'epistemic_status': 'inferred'}]})
        self.assertEqual(next(x for x in items if x.evidence_type == 'auditory_consequence').epistemic_status,
                         'inferred')

    def test_unavailable_rejects_negative_claim(self):
        with self.assertRaises(ValueError):
            replace(unavailable_modality(self.scene, 'auditory'), value={'silence': True})

    def test_stale_observed_opposition_ignored(self):
        self.assertEqual(generate(self.scene), generate(self.scene,
            sensory(timestamp=9., contradicts=('continued_motion',))))

    def test_provider_preserves_explicit_links(self):
        item = self.candidate(track_id=7, supports=['continued_motion'],
                              supporting_evidence_ids=[self.anchor.evidence_id])
        result = next(x for x in generate(self.scene, item).hypotheses if x.hypothesis_type == 'continued_motion')
        self.assertIn(item.evidence_id, result.evidence_for)

    def test_missing_cannot_link(self):
        for changes in ({'supports': ('continued_motion',)}, {'contradicts': ('continued_motion',)},
                        {'confidence': .1}, {'supporting_evidence_ids': ('p',)}):
            with self.assertRaises(ValueError):
                replace(unavailable_modality(self.scene, 'auditory'), **changes)

    def test_low_confidence(self):
        self.assertEqual(self.candidate(confidence=.02).confidence, .02)

    def test_heuristic_is_not_confidence(self):
        item = self.candidate()
        self.assertIsNone(item.confidence)
        self.assertEqual(item.value['probability'], .12)
        self.assertIn('not calibrated', item.provenance['probability_policy'])

    def test_provenance_timestamp_association(self):
        item = self.candidate(timestamp=8., track_id=7, object_id='obj', provenance={'source': 'fixture'})
        self.assertEqual((item.timestamp, item.track_id, item.object_id), (8., 7, 'obj'))
        self.assertEqual(item.provenance['source_provenance']['source'], 'fixture')
        self.assertIn('auditory_consequences', item.provenance['source_field'])

    def test_class_name_is_not_identity(self):
        item = self.candidate(source_object='person')
        self.assertIsNone(item.track_id)
        self.assertIsNone(item.object_id)

    def test_source_timestamp_fallback(self):
        items = self.provider.provide(self.scene, {'auditory_consequences': []}, timestamp=3.)
        self.assertEqual(items[0].timestamp, 3.)

    def test_deterministic_and_label_permutation(self):
        self.assertEqual(self.candidate(evidence=['a', 'b']), self.candidate(evidence=['b', 'a']))

    def test_record_permutation(self):
        records = [dict(consequence=x, observed=False) for x in ('a', 'b')]
        first = self.provider.provide(self.scene, {'auditory_consequences': records})
        second = self.provider.provide(self.scene, {'auditory_consequences': records[::-1]})
        self.assertEqual(first, second)

    def test_duplicate_does_not_accumulate(self):
        record = dict(consequence='a', observed=False)
        self.assertEqual(self.provider.provide(self.scene, {'auditory_consequences': [record]}),
                         self.provider.provide(self.scene, {'auditory_consequences': [record, record]}))

    def test_no_sensory_is_backward_compatible(self):
        generator = MultiHypothesisGenerator()
        self.assertEqual(generator.generate(self.scene, collect_evidence(self.scene)),
                         generator.generate(self.scene, collect_evidence(self.scene, include_sensory=True)))

    def test_unlinked_consequence_is_backward_compatible(self):
        self.assertEqual(generate(self.scene), generate(self.scene, self.candidate(track_id=7)))

    def test_sensory_alone_cannot_create_candidates(self):
        for status in ('observed', 'inferred', 'expected'):
            current = scene()
            extra = sensory(epistemic_status=status, supports=('continued_motion', 'hidden_actor', 'collision'))
            self.assertEqual(generate(current), generate(current, extra))

    def test_semantic_alone_cannot_create_actor(self):
        current = scene(semantic_evidence={'semantic_evidence': {'mentioned_objects': ['ball', 'person']}})
        self.assertEqual({x.hypothesis_type for x in generate(current).hypotheses}, {'unknown_dynamic_event'})

    def test_occlusion_semantics_and_absence_do_not_create_actor(self):
        current = scene(occlusion_evidence={'occlusion_evidence': [dict(has_overlap_evidence=True)]},
                        semantic_evidence={'semantic_evidence': {'mentioned_objects': ['ball']}})
        result = generate(current, unavailable_modality(current, 'auditory'))
        self.assertFalse(any('actor' in x.hypothesis_type for x in result.hypotheses))

    def test_explicit_observed_opposition(self):
        item = sensory(contradicts=('continued_motion',))
        before = {x.hypothesis_type: x for x in generate(self.scene).hypotheses}
        after = {x.hypothesis_type: x for x in generate(self.scene, item).hypotheses}
        self.assertEqual(set(before), set(after))
        self.assertEqual(after['continued_motion'].posterior_probability,
                         before['continued_motion'].posterior_probability * .5)
        self.assertEqual(after['continued_motion'].evidence_against, ('sensory',))

    def test_expected_support_requires_anchor(self):
        item = sensory(epistemic_status='expected', supports=('continued_motion',))
        self.assertEqual(generate(self.scene), generate(self.scene, item))
        linked = replace(item, supporting_evidence_ids=(self.anchor.evidence_id,))
        result = next(x for x in generate(self.scene, linked).hypotheses if x.hypothesis_type == 'continued_motion')
        self.assertIn(linked.evidence_id, result.evidence_for)
        self.assertEqual(result.confidence, .4)
        self.assertEqual(result.posterior_probability, 1.)
        self.assertEqual(next(x for x in result.provenance['score_inputs']
                             if x['evidence_id'] == 'sensory')['epistemic_status'], 'expected')

    def test_stale_wrong_identity_and_dangling_anchor_ignored(self):
        item = sensory(epistemic_status='expected', supports=('continued_motion',),
                       supporting_evidence_ids=(self.anchor.evidence_id,))
        for changes in ({'timestamp': 9.}, {'track_id': 8}, {'supporting_evidence_ids': ('missing',)}):
            self.assertEqual(generate(self.scene), generate(self.scene, replace(item, **changes)))

    def test_physical_contradiction_blocks_sensory_support(self):
        item = sensory(epistemic_status='expected', supports=('object_stops',),
                       supporting_evidence_ids=(self.anchor.evidence_id,))
        self.assertEqual(generate(self.scene), generate(self.scene, item))

    def test_inferred_opposition_needs_explicit_opposing_anchor(self):
        item = sensory(epistemic_status='inferred', contradicts=('object_stops',))
        self.assertEqual(generate(self.scene), generate(self.scene, item))
        linked = replace(item, opposing_evidence_ids=(self.anchor.evidence_id,))
        result = next(x for x in generate(self.scene, linked).hypotheses if x.hypothesis_type == 'object_stops')
        self.assertIn('sensory', result.evidence_against)

    def test_input_permutation(self):
        items = [self.anchor, sensory(contradicts=('continued_motion',)),
                 unavailable_modality(self.scene, 'thermal')]
        results = [MultiHypothesisGenerator().generate(self.scene, EvidenceBundle('s', order))
                   for order in permutations(items)]
        self.assertTrue(all(x == results[0] for x in results))

    def test_semantic_impact_stays_unverified(self):
        output = AuditoryReasoner().infer_expected_consequences(WorldState(10., {'scene_description': 'ball hits ground'}))
        evidence = collect_evidence(self.scene, auditory=output)
        self.assertEqual(MultiHypothesisGenerator().generate(self.scene, evidence), generate(self.scene))
        self.assertTrue(any(x.value.get('consequence') == 'contact_or_impact_sound_possible' for x in evidence.items))


if __name__ == '__main__':
    unittest.main()
