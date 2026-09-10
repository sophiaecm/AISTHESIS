"""Deterministic retrieval, leakage, evidence and bounded-context tests."""
from dataclasses import replace
from unittest.mock import patch
import json
import unittest

from evaluation.sensory import _plain
from fifth_layer.world_model import (SceneState, ExperienceMemory, EvidenceItem, EvidenceBundle,
    collect_evidence, MultiHypothesisGenerator, prediction_from_hypothesis, outcome_from_scene,
    evaluate_prediction, experience_episode, adapt_temporal_trajectories)
from fifth_layer.world_model.experience_learning import (
    ExperienceQuery, ExperienceRetriever, ExperienceEvidenceProvider, experience_context,
)


def current(time=10., name='current', motion='moving_left', category='ball'):
    obj = dict(track_id=7, box_xyxy=[0, 0, 10, 10])
    if category is not None:
        obj['class_name'] = category
    scene = SceneState(name, time, image_width=100, image_height=100,
        observed_objects=[obj], motion_evidence=[dict(track_id=7, motion_state=motion)])
    evidence = collect_evidence(scene, include_sensory=True)
    hypotheses = MultiHypothesisGenerator().generate(scene, evidence)
    return scene, evidence, hypotheses


def historical(start=1., name='past', status='supported', category='ball', motion='moving_left', sensory=False):
    scene, evidence, hypotheses = current(start, name, motion, category)
    h = next(h for h in hypotheses.hypotheses if h.hypothesis_type == 'continued_motion')
    if sensory:
        provenance = dict(h.provenance)
        provenance['score_inputs'] = (*provenance['score_inputs'], dict(evidence_id='historical-sound',
            value={'consequence': 'footsteps_possible'}, modality='auditory', epistemic_status='expected'))
        h = replace(h, provenance=provenance)
    prediction = prediction_from_hypothesis(scene, h, evidence)
    after, _, _ = current(start + 1., name + '-outcome',
                          'moving_right' if status == 'contradicted' else motion, category)
    outcome = outcome_from_scene(prediction, after)
    evaluation = evaluate_prediction(prediction, outcome)
    evaluation = replace(evaluation, status=status)
    return experience_episode(prediction, outcome, evaluation)


def memory(*episodes, at=0., capacity=512):
    result = ExperienceMemory(clock=lambda: at, capacity=capacity)
    for episode in episodes:
        result.add(episode)
    return result


def query_for(scene, evidence, hypotheses):
    h = next(h for h in hypotheses.hypotheses if h.hypothesis_type == 'continued_motion')
    return ExperienceQuery.from_current(scene, evidence, h)


class ExperienceLearningTests(unittest.TestCase):
    def setUp(self):
        self.scene, self.evidence, self.hypotheses = current()
        self.query = query_for(self.scene, self.evidence, self.hypotheses)

    def retrieve(self, *episodes, **policy):
        return ExperienceRetriever(**policy).retrieve(self.query, memory(*episodes), current_time=10.)

    def context(self, *episodes, **options):
        return experience_context(self.scene, self.evidence, self.hypotheses, memory(*episodes),
                                  current_time=10., enabled=True, **options)

    def continuation(self, context):
        h = next(h for h in self.hypotheses.hypotheses if h.hypothesis_type == 'continued_motion')
        return next(r for r in context['ranking'] if r['hypothesis_id'] == h.hypothesis_id)

    def test_similar_supported_retrieved(self):
        episode = historical()
        records, audit = self.retrieve(episode)
        self.assertEqual(records[0].episode_id, episode.episode_id)
        self.assertEqual(records[0].similarity_score, 1.)
        self.assertIn('motion_state', records[0].matched_features)
        self.assertTrue(audit[0]['eligible'])

    def test_unrelated_not_retrieved(self):
        records, _ = self.retrieve(historical(category='car', motion='moving_right'))
        self.assertEqual(records, ())

    def test_future_episode_never_retrieved(self):
        self.assertEqual(self.retrieve(historical(start=11.))[0], ())

    def test_equal_timestamp_is_not_past(self):
        self.assertEqual(self.retrieve(historical(start=9.))[0], ())

    def test_evaluation_after_current_not_retrieved(self):
        e = historical()
        error = dict(e.prediction_error, evaluated_timestamp=12.)
        self.assertEqual(self.retrieve(replace(e, evaluated_timestamp=12., prediction_error=error))[0], ())

    def test_target_after_current_not_retrieved(self):
        e = historical()
        p = dict(e.prediction_summary, target_timestamp=12., horizon_seconds=11.)
        self.assertEqual(self.retrieve(replace(e, prediction_summary=p))[0], ())

    def test_nested_future_observation_rejected(self):
        e = historical()
        o = dict(e.observation_summary, observation_timestamp=12.)
        self.assertEqual(self.retrieve(replace(e, observation_summary=o))[0], ())

    def test_pending_expired_and_unevaluable_rejected(self):
        for status in ('pending', 'expired', 'unevaluable'):
            self.assertEqual(self.retrieve(replace(historical(), evaluation_status=status))[0], ())

    def test_incomplete_legacy_record_rejected(self):
        self.assertEqual(self.retrieve(replace(historical(), prediction_summary={}))[0], ())

    def test_current_prediction_excluded(self):
        e = historical(name='current')
        self.assertEqual(self.retrieve(e)[0], ())

    def test_query_time_cannot_be_advanced_to_leak(self):
        with self.assertRaises(ValueError):
            ExperienceRetriever().retrieve(self.query, memory(historical()), current_time=20.)

    def test_unobservable_is_not_negative(self):
        row = self.continuation(self.context(historical(status='unobservable')))
        self.assertEqual(row['experience_contribution'], 0.)
        self.assertEqual(row['retrieved'][0]['evaluation_status'], 'unobservable')

    def test_insufficient_evidence_not_negative(self):
        self.assertEqual(self.continuation(self.context(historical(status='insufficient_evidence')))
                         ['experience_contribution'], 0.)

    def test_partial_is_context_only(self):
        self.assertEqual(self.continuation(self.context(historical(status='partially_supported')))
                         ['experience_contribution'], 0.)

    def test_historical_contradiction_never_deletes(self):
        result = self.context(historical(status='contradicted'))
        self.assertIs(result['hypotheses'], self.hypotheses)
        self.assertLess(self.continuation(result)['experience_contribution'], 0.)

    def test_missing_feature_not_mismatch(self):
        record = self.retrieve(historical(category=None))[0][0]
        self.assertIn('object_class', record.unavailable_features)
        self.assertNotIn('object_class', record.conflicting_features)
        self.assertEqual(record.similarity_score, 1.)

    def test_unavailable_sensory_not_match_or_conflict(self):
        query = replace(self.query, features={**self.query.features, 'sensory.auditory.status': 'unavailable'})
        retrieved, _ = ExperienceRetriever().retrieve(query, memory(historical(sensory=True)), current_time=10.)
        self.assertIn('sensory.auditory.status', retrieved[0].unavailable_features)
        self.assertNotIn('sensory.auditory.status', retrieved[0].conflicting_features)

    def test_top_k_deterministic_order(self):
        episodes = [historical(name=f'past-{i}') for i in range(6)]
        records, _ = self.retrieve(*episodes)
        self.assertEqual([r.episode_id for r in records], sorted(e.episode_id for e in episodes)[:3])

    def test_episode_permutation_same_retrieval(self):
        episodes = [historical(name=f'past-{i}') for i in range(4)]
        self.assertEqual(self.retrieve(*episodes), self.retrieve(*episodes[::-1]))

    def test_byte_determinism(self):
        def encoded():
            return json.dumps(_plain(self.context(historical())), sort_keys=True, separators=(',', ':'))
        self.assertEqual(encoded(), encoded())

    def test_read_does_not_mutate_or_call_clock(self):
        m = memory(historical())
        before = tuple(m._episodes.items())
        with patch.object(m, '_clock', side_effect=AssertionError('hidden clock')):
            ExperienceRetriever().retrieve(self.query, m, current_time=10.)
        self.assertEqual(tuple(m._episodes.items()), before)

    def test_episode_unchanged(self):
        e = historical()
        before = json.dumps(_plain(e), sort_keys=True)
        self.context(e)
        self.assertEqual(json.dumps(_plain(e), sort_keys=True), before)

    def test_disabled_exact_base_hypotheses(self):
        m = memory(historical())
        with patch.object(m, 'snapshot', side_effect=AssertionError('retrieval when disabled')):
            result = experience_context(self.scene, self.evidence, self.hypotheses, m, current_time=10.)
        self.assertIs(result['hypotheses'], self.hypotheses)
        self.assertTrue(all(r['experience_contribution'] == 0. for r in result['ranking']))
        self.assertEqual(result['experience_evidence'].items, ())

    def test_default_generator_and_trajectories_unchanged(self):
        result = self.context(historical())
        combined = EvidenceBundle(self.scene.scene_id, (*self.evidence.items, *result['experience_evidence'].items))
        self.assertEqual(MultiHypothesisGenerator().generate(self.scene, combined), self.hypotheses)
        self.assertEqual(adapt_temporal_trajectories(self.scene, combined, self.hypotheses),
                         adapt_temporal_trajectories(self.scene, self.evidence, self.hypotheses))

    def test_absolute_and_relative_bound(self):
        row = self.continuation(self.context(*(historical(name=f'e{i}') for i in range(8))))
        self.assertLessEqual(abs(row['experience_contribution']), .05)
        self.assertLessEqual(abs(row['experience_contribution']), .05 * row['original_heuristic_score'])

    def test_configured_bound(self):
        self.assertAlmostEqual(self.continuation(self.context(historical(), max_contribution=.01))
                               ['experience_contribution'], .01)
        with self.assertRaises(ValueError):
            self.context(historical(), max_contribution=.5)

    def test_current_contradiction_blocks_support(self):
        contrary = EvidenceItem('opposition', 'current', 'physics', 'fixture', 'physical_constraint', {}, 10.,
                                track_id=7, contradicts=('continued_motion',))
        evidence = EvidenceBundle('current', (*self.evidence.items, contrary))
        hypotheses = MultiHypothesisGenerator().generate(self.scene, evidence)
        result = experience_context(self.scene, evidence, hypotheses, memory(historical()), current_time=10., enabled=True)
        row = self.continuation(result)
        self.assertEqual(row['experience_contribution'], 0.)
        self.assertEqual(row['reason'], 'current_contradiction')

    def test_current_physical_support_required(self):
        result = experience_context(self.scene, EvidenceBundle('current'), self.hypotheses,
                                    memory(historical()), current_time=10., enabled=True)
        self.assertTrue(all(r['experience_contribution'] == 0. for r in result['ranking']))

    def test_competing_histories_coexist_and_cancel(self):
        result = self.context(historical(name='yes'), historical(name='no', status='contradicted'))
        row = self.continuation(result)
        self.assertEqual({r['evaluation_status'] for r in row['retrieved']}, {'supported', 'contradicted'})
        self.assertEqual(row['experience_contribution'], 0.)
        self.assertFalse(result['provenance']['winner_selected'])

    def test_heuristic_not_probability(self):
        result = self.context(historical())
        self.assertFalse(result['provenance']['calibrated'])
        self.assertEqual(self.continuation(result)['original_heuristic_score'], 1.)
        self.assertEqual(self.continuation(result)['final_heuristic_score'], 1.05)

    def test_hidden_actor_history_cannot_create_actor(self):
        e = historical()
        e = replace(e, prediction_summary=dict(e.prediction_summary, hypothesis_type='hidden_human'))
        result = self.context(e)
        self.assertEqual(result['hypotheses'], self.hypotheses)
        self.assertTrue(all(r['experience_contribution'] == 0. for r in result['ranking']))

    def test_experience_alone_cannot_create_candidate(self):
        item = EvidenceItem('historic', 'empty', 'experience', 'fixture', 'context', {}, 10.,
                            track_id=7, supports=('continued_motion', 'hidden_human'))
        scene = SceneState('empty', 10.)
        generator = MultiHypothesisGenerator()
        self.assertEqual(generator.generate(scene, EvidenceBundle('empty', (item,))),
                         generator.generate(scene, EvidenceBundle('empty')))

    def test_expected_auditory_stays_historical(self):
        result = self.context(historical(sensory=True))
        item = result['experience_evidence'].items[0]
        self.assertEqual(item.source_type, 'experience')
        self.assertFalse(item.value['current_observation'])
        self.assertIsNone(item.epistemic_status)
        self.assertEqual(item.provenance['historical_features']['sensory.auditory.status'], 'expected')
        self.assertEqual(item.supports + item.contradicts, ())
        self.assertIsNone(item.confidence)

    def test_evidence_provenance(self):
        e = historical()
        item = self.context(e)['experience_evidence'].items[0]
        self.assertEqual(item.value['source_episode_id'], e.episode_id)
        self.assertEqual(item.provenance['historical_timestamps']['evaluated_timestamp'], e.evaluated_timestamp)

    def test_ttl_read_only_and_capacity_unchanged(self):
        m = memory(historical(), at=0., capacity=1)
        self.assertEqual(m.snapshot(at_time=60.), ())
        self.assertEqual(len(m._episodes), 1)
        second = historical(name='second')
        m.add(second)
        self.assertEqual([e.episode_id for e in m.snapshot(at_time=1.)], [second.episode_id])

    def test_future_admission_not_visible(self):
        m = memory(historical(), at=20.)
        self.assertEqual(ExperienceRetriever().retrieve(self.query, m, current_time=10.)[0], ())

    def test_top_k_limit(self):
        for value in (0, 9, True):
            with self.assertRaises(ValueError):
                ExperienceRetriever(top_k=value)

    def test_provider_rejects_future_context(self):
        record = self.retrieve(historical())[0][0]
        record = replace(record, historical_timestamps=dict(record.historical_timestamps, evaluated_timestamp=20.))
        h = next(h for h in self.hypotheses.hypotheses if h.hypothesis_type == 'continued_motion')
        with self.assertRaises(ValueError):
            ExperienceEvidenceProvider().provide(self.scene, h, self.query, (record,))

    def test_position_only_contradiction_not_event_penalty(self):
        e = historical(status='contradicted')
        error = dict(e.prediction_error, metrics={'position_match': False, 'motion_state_match': None})
        e = replace(e, prediction_error=error)
        row = self.continuation(self.context(e))
        self.assertTrue(row['retrieved'])
        self.assertEqual(row['experience_contribution'], 0.)

    def test_future_observed_event_time_rejected(self):
        e = historical()
        outcome = dict(e.observation_summary, observed_state=dict(e.observation_summary['observed_state'], event_timestamp=20.))
        self.assertEqual(self.retrieve(replace(e, observation_summary=outcome))[0], ())

    def test_conflicting_structure_retained_without_influence(self):
        row = self.continuation(self.context(historical(category='car')))
        self.assertTrue(row['retrieved'])
        self.assertIn('object_class', row['retrieved'][0]['conflicting_features'])
        self.assertEqual(row['experience_contribution'], 0.)


if __name__ == '__main__':
    unittest.main()
