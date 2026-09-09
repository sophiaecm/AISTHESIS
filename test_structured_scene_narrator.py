import unittest
from copy import deepcopy
from fifth_layer.world_state import WorldState
from fifth_layer.perception.structured_scene_narrator import StructuredSceneNarrator, FastSceneState


def item(label='person', x=150, y=150, track=1, **extra):
    return dict(class_name=label, track_id=track, object_id=track, confidence=.9,
                box_xyxy=[x-10,y-10,x+10,y+10], observation_state='observed', **extra)


def world(items=None, **extra):
    return WorldState(100, dict(accepted_detections=[item()] if items is None else items,
                               image_width=300, image_height=300, **extra))


class NarratorTests(unittest.TestCase):
    def narrate(self, state=None, **kwargs):
        return StructuredSceneNarrator(**kwargs).narrate(state or world())

    def test_single_person_and_no_fabricated_semantics(self):
        text = self.narrate()['description']
        self.assertIn('One person is visible near the center', text)
        for forbidden in ('woman', 'man ', 'young', 'hair', 'shirt', 'blue', 'sitting', 'intends'):
            self.assertNotIn(forbidden, text)

    def test_plural_and_confidence_filter(self):
        state = world([item(), item(track=2), dict(item(track=3), confidence=.1)])
        result = self.narrate(state)
        self.assertIn('Two people are visible', result['description'])
        self.assertEqual(result['accepted_object_count'], 2)

    def test_position_grid(self):
        for x,h in ((30,'left'),(150,'center'),(270,'right')):
            for y,v in ((30,'upper'),(150,'middle'),(270,'lower')):
                with self.subTest(x=x,y=y):
                    text = self.narrate(world([item(x=x,y=y)]))['description']
                    if v != 'middle':
                        self.assertIn(v, text)
                    if h != 'center' or v == 'middle':
                        self.assertIn(h, text)

    def test_stable_motion_and_jitter(self):
        for movement, normalized, expected in [('moving_right',.1,'is moving right'),
                ('stationary',0,'is stationary'),('moving_left',.001,'No reliable motion')]:
            text = self.narrate(world(stable_motion_evidence=[dict(track_id=1,
                motion_state=movement, normalized_motion=normalized)]))['description']
            self.assertIn(expected, text)

    def test_unstable_motion_not_narrated(self):
        text = self.narrate(world(motion_evidence=[dict(track_id=1,motion_state='moving_left')]))['description']
        self.assertNotIn('moving left', text)

    def test_predicted_not_visible_and_expiration(self):
        p = dict(track_id=9,class_name='phone',is_predicted=True,observation_state='predicted',
                 predicted_bbox=[200,200,240,240],prediction_valid_until=101)
        result = self.narrate(world([],predicted_tracks=[p]))
        self.assertEqual(result['accepted_object_count'],0)
        self.assertIn('previously observed phone may', result['description'])
        self.assertNotIn('phone is visible', result['description'])
        p['prediction_valid_until'] = 99
        self.assertEqual(self.narrate(world([],predicted_tracks=[p]))['predicted_track_count'],0)

    def test_uncertain_rejected_predicted_are_not_objects(self):
        items = [item(uncertain=True),item(rejected=True),item(is_predicted=True)]
        self.assertEqual(self.narrate(world(items))['accepted_object_count'],0)

    def test_occlusion_no_depth_claim(self):
        result = self.narrate(world(occlusion_evidence=[dict(possible_occlusion_evidence=True)]))
        self.assertIn('Possible occlusion',result['description'])
        self.assertNotIn('behind',result['description'])

    def test_high_risk_first_all_modes_and_levels(self):
        for mode in ('access','adas'):
            for detail in ('concise','detailed','accessibility_detailed'):
                self.assertTrue(self.narrate(world(risk='HIGH'),mode=mode,detail_level=detail)['description'].startswith('High risk'))

    def test_unknown_uncertainty_and_insufficient(self):
        text=self.narrate()['description']
        self.assertIn('cannot yet be determined',text)
        self.assertIn('insufficient evidence',text)
        self.assertIn('remains uncertain',text)
        self.assertNotIn('no risk',text)

    def test_levels_and_modes(self):
        concise=self.narrate(detail_level='concise')
        detailed=self.narrate(detail_level='detailed')
        self.assertLess(len(concise['description']),len(detailed['description']))
        self.assertLessEqual(concise['sentence_count'],3)
        self.assertEqual(self.narrate(mode='access')['detail_level'],'accessibility_detailed')
        text=self.narrate(world([item('vase'),item('car',track=2)]),mode='adas')['description']
        self.assertTrue(text.startswith('One car'))

    def test_evidence_types_and_latent(self):
        result=self.narrate(world(latent='trajectory_toward_occlusion'))
        self.assertIn('observed',result['evidence_types'])
        self.assertIn('inferred',result['evidence_types'])
        self.assertIn('movement pattern suggests',result['description'])

    def test_spatial_relation_uses_confirmed_ids(self):
        result=self.narrate(world([item(),item('phone',track=2)],scene_relations=[
            dict(first_object_id=1,second_object_id=2,relation='near')]))
        self.assertIn('The person is near the phone in the image.',result['spatial_summary'])

    def test_limits_and_legacy_and_no_mutation(self):
        state=world([item(track=i) for i in range(100)])
        before=deepcopy(state)
        result=self.narrate(state,max_characters=300,max_sentences=4)
        self.assertLessEqual(len(result['description']),300)
        self.assertLessEqual(result['sentence_count'],4)
        self.assertEqual(state,before)
        self.assertEqual(self.narrate(WorldState())['accepted_object_count'],0)
        self.assertEqual(StructuredSceneNarrator().narrate({'detections':[item()]})['accepted_object_count'],0)

    def test_latency(self):
        times=[self.narrate(world([item(track=i) for i in range(100)]),detail_level='detailed')['latency_ms'] for _ in range(100)]
        self.assertLess(max(times),100)
        print(f'NARRATION BENCHMARK 100 objects / 100 runs: mean={sum(times)/len(times):.3f}ms max={max(times):.3f}ms')

    def test_debounce_jitter_and_risk_override(self):
        now=[0.]
        state=FastSceneState(clock=lambda:now[0],wall_clock=lambda:100+now[0])
        first=state.update(world())['description']
        now[0]=.1
        next_state=world([item(x=151)])
        next_state.timestamp=100.1
        self.assertEqual(state.update(next_state)['description'],first)
        next_state.data['risk']='HIGH'
        self.assertTrue(state.update(next_state)['description'].startswith('High risk'))
        now[0]=4
        self.assertNotIn('Waiting',state.description())

    def test_fast_deep_lifetimes_independent(self):
        from fifth_layer.perception.deep_analysis import DeepAnalysisState
        from types import SimpleNamespace
        clock=[0.]
        fast=FastSceneState(clock=lambda:clock[0],wall_clock=lambda:100+clock[0])
        deep=DeepAnalysisState(clock=lambda:clock[0],wall_clock=lambda:100+clock[0])
        job=deep.begin(SimpleNamespace(timestamp=100))
        deep.apply_scene(job,'visual details')
        fast.update(world())
        self.assertEqual(deep.scene_description(),'visual details')
        clock[0]=26
        fresh=world(); fresh.timestamp=126
        fast.update(fresh)
        self.assertIsNone(deep.current())
        self.assertEqual(deep.scene_description(),'Analyzing scene...')
        self.assertIn('One person',fast.description())

    def test_live_update_and_two_overlay_sections(self):
        from test_analysis_connections import live_functions
        from unittest.mock import Mock
        import numpy as np
        env=live_functions('draw_overlay')
        env.update(fast_scene_state=FastSceneState(wall_clock=lambda:100),
            latest_stable_motion_evidence=[], latest_reasoning={},
            latest_description='Deep visual details', latest_motion_summary='stationary',
            get_display_reasoning=lambda:{}, cv2=Mock(), draw_tracked_detections=lambda f,d:f)
        state=world()
        env['update_fast_scene'](state)
        self.assertIn('One person',state.data['fast_scene_description'])
        env['draw_overlay'](np.zeros((480,640,3)),[])
        labels=' '.join(c.args[1] for c in env['cv2'].putText.call_args_list)
        self.assertIn('FAST STRUCTURED SCENE',labels)
        self.assertIn('DEEP VISUAL DESCRIPTION',labels)
        self.assertIn('One person',labels)
        self.assertIn('Deep visual details',labels)
