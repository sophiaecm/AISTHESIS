"""Deterministic synthetic localization evidence; no real backend inference."""
from dataclasses import FrozenInstanceError, replace
import json
import socket
from unittest.mock import patch

import pytest

from fifth_layer.world_model.spatial_grounding import (
    GroundingFrame, GroundingRequest, RawGroundingObservation, GroundedRegion,
    GroundedObservationTarget, SpatialGroundingAdapter, VERSION, POLICY)
from evaluation.tests.test_active_perception import B as AP, inputs

B = SpatialGroundingAdapter()


def plan():
    return AP.build(inputs([], [(0, 1)])[0])


def frame(p=None, width=640, height=480):
    p = p or plan()
    return GroundingFrame('image2', width, height, p.timestamp, p.session_id, p.coordinate_frame_id, p.scene_id)


def request(outputs=('box', 'point')):
    p = plan()
    return B.create_request(p, p.targets[0].target_id, frame(p), outputs)


def raw(r=None, kind='box', coordinates=(0, 0, 1000, 1000), space='normalized_0_1000', **kwargs):
    r = r or request()
    return RawGroundingObservation(r.request_id, r.frame.frame_id, 'synthetic', 'v1', kind, coordinates, space, **kwargs)


def test_request_exact_symbolic_lineage():
    p = plan()
    target = p.targets[0]
    result = B.create_request(p, target.target_id, frame(p))
    assert result.source_plan_id == p.plan_id
    assert result.source_target_id == target.target_id
    assert result.object_ids == target.object_ids
    assert result.relation_types == target.relation_types
    assert result.provenance['source_cue_ids'] == target.cue_ids
    assert result.provenance['source_ids'] == target.source_ids
    assert result.provenance['source_topological_state_id'] == p.source_topological_state_id
    assert result.provenance['source_priority_band'] == target.priority_band
    assert result.symbolic_query == dict(target_scope=target.target_scope, object_ids=target.object_ids,
                                         relation_types=target.relation_types, semantic_query_status='unavailable')
    assert 'semantic_query_unavailable' in result.uncertainty


@pytest.mark.parametrize('name', ['width', 'height'])
@pytest.mark.parametrize('value', [0, -1, 2.5, True, '640', None])
def test_invalid_dimensions(name, value):
    with pytest.raises(ValueError):
        replace(frame(), **{name: value})


@pytest.mark.parametrize('value', [None, -1, float('nan'), float('inf'), True, '2'])
def test_invalid_timestamp(value):
    with pytest.raises(ValueError):
        replace(frame(), timestamp=value)


@pytest.mark.parametrize('name', ['image_id', 'session_id', 'coordinate_frame_id', 'scene_id'])
@pytest.mark.parametrize('value', ['', None])
def test_frame_ids_required(name, value):
    with pytest.raises(ValueError):
        replace(frame(), **{name: value})


@pytest.mark.parametrize('name,value', [('timestamp', 3.), ('scene_id', 'other'), ('session_id', 'other'), ('coordinate_frame_id', 'other')])
def test_frame_plan_alignment(name, value):
    p = plan()
    with pytest.raises(ValueError):
        B.create_request(p, p.targets[0].target_id, replace(frame(p), **{name: value}))


@pytest.mark.parametrize('value', [None, {}, 'plan', 2])
def test_wrong_plan_type(value):
    with pytest.raises(ValueError):
        B.create_request(value, 'target', frame())


def test_wrong_target_id():
    with pytest.raises(ValueError):
        B.create_request(plan(), 'missing', frame())


@pytest.mark.parametrize('name,value', [('schema_version', 'wrong'), ('plan_id', 'wrong')])
def test_corrupt_plan(name, value):
    p = plan()
    object.__setattr__(p, name, value)
    with pytest.raises(ValueError):
        B.create_request(p, p.targets[0].target_id, frame())


def test_corrupt_nested_target():
    p = plan()
    object.__setattr__(p.targets[0], 'schema_version', 'wrong')
    p = replace(p)
    with pytest.raises(ValueError):
        B.create_request(p, p.targets[0].target_id, frame())


@pytest.mark.parametrize('outputs', [(), ('mask',), ('region_reference',), ('box', 'unknown')])
def test_only_supported_request_outputs(outputs):
    with pytest.raises(ValueError):
        request(outputs)


@pytest.mark.parametrize('space,kind,coords,expected', [
    ('normalized_0_1000', 'box', (0, 250, 1000, 750), (0, 120, 640, 360)),
    ('normalized_0_1000', 'point', (500, 500), (320, 240)),
    ('normalized_0_1', 'box', (0, .25, 1, .75), (0, 120, 640, 360)),
    ('normalized_0_1', 'point', (.5, .5), (320, 240)),
    ('pixel', 'box', (0, 120, 640, 360), (0, 120, 640, 360)),
    ('pixel', 'point', (320, 240), (320, 240)),
])
def test_coordinate_conversions(space, kind, coords, expected):
    r = request()
    result = B.integrate_observations(r, (raw(r, kind, coords, space),))
    region, = result.regions
    assert region.pixel_coordinates == expected
    assert region.normalized_coordinates == tuple(v / (640 if i % 2 == 0 else 480) for i, v in enumerate(expected))
    assert result.grounding_status == 'grounded_candidate'


@pytest.mark.parametrize('space,extent', [('normalized_0_1000', 1000), ('normalized_0_1', 1)])
@pytest.mark.parametrize('kind', ['box', 'point'])
def test_extent_and_zero_boundary(space, extent, kind):
    r = request()
    coords = (0, 0, extent, extent) if kind == 'box' else (extent, extent)
    region, = B.integrate_observations(r, (raw(r, kind, coords, space),)).regions
    assert region.pixel_coordinates[-2:] == (640, 480)


def test_fractional_pixels_not_rounded():
    r = request()
    region, = B.integrate_observations(r, (raw(r, 'point', (1, 1)),)).regions
    assert region.pixel_coordinates == (.64, .48)


@pytest.mark.parametrize('kind,coords', [('box', (0, 0)), ('box', (0, 0, 1)), ('point', (0,)),
                                      ('point', (0, 0, 1)), ('box', '0,0,1,1'), ('mask', (0, 0))])
def test_bad_coordinate_arity(kind, coords):
    with pytest.raises(ValueError):
        raw(kind=kind, coordinates=coords)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), -1, True, None, '1'])
@pytest.mark.parametrize('space', ['normalized_0_1000', 'normalized_0_1', 'pixel'])
def test_invalid_coordinates(value, space):
    with pytest.raises(ValueError):
        raw(kind='point', coordinates=(value, 0), space=space)


@pytest.mark.parametrize('space,coords', [('normalized_0_1000', (1001, 0)), ('normalized_0_1', (1.01, 0))])
def test_normalized_out_of_bounds(space, coords):
    with pytest.raises(ValueError):
        raw(kind='point', coordinates=coords, space=space)


@pytest.mark.parametrize('coords', [(641, 0), (0, 481)])
def test_pixel_out_of_bounds_at_integration(coords):
    r = request()
    observation = raw(r, 'point', coords, 'pixel')
    with pytest.raises(ValueError):
        B.integrate_observations(r, (observation,))


@pytest.mark.parametrize('coords', [(2, 0, 1, 1), (0, 2, 1, 1)])
@pytest.mark.parametrize('space', ['normalized_0_1000', 'pixel'])
def test_reversed_boxes_rejected(coords, space):
    with pytest.raises(ValueError):
        raw(coordinates=coords, space=space)


def test_zero_area_box_allowed():
    r = request()
    result = B.integrate_observations(r, (raw(r, coordinates=(500, 500, 500, 500)),))
    assert result.regions[0].pixel_coordinates == (320, 240, 320, 240)


@pytest.mark.parametrize('name', ['request_id', 'frame_id'])
def test_wrong_raw_lineage(name):
    r = request()
    observation = replace(raw(r), **{name: 'different'})
    with pytest.raises(ValueError):
        B.integrate_observations(r, (observation,))


def test_cross_image_reuse_rejected():
    p = plan()
    r = request()
    other = B.create_request(p, p.targets[0].target_id, replace(frame(p), image_id='different_image'))
    with pytest.raises(ValueError):
        B.integrate_observations(other, (raw(r),))


def test_unrequested_output_rejected():
    r = request(('box',))
    with pytest.raises(ValueError):
        B.integrate_observations(r, (raw(r, 'point', (0, 0)),))


@pytest.mark.parametrize('score', [None, -9., 0., .25, 9.])
def test_score_only_backend_metadata(score):
    r = request()
    result = B.integrate_observations(r, (raw(r, backend_score=score, backend_label='unverified label'),))
    assert result.backend_metadata[0]['backend_score'] == score
    assert result.backend_metadata[0]['backend_label'] == 'unverified label'
    assert result.provenance['source_priority_band'] == r.provenance['source_priority_band']
    assert result.provenance['backend_score_promoted'] is False
    assert not {'confidence', 'posterior', 'risk', 'priority_band'} & result.to_dict().keys()


@pytest.mark.parametrize('score', [float('nan'), float('inf'), True, '0.5'])
def test_invalid_score(score):
    with pytest.raises(ValueError):
        raw(backend_score=score)


def test_multiple_candidates_keep_low_and_high_scores():
    r = request()
    a, b = raw(r, backend_score=.01), raw(r, 'point', (500, 500), backend_score=.99)
    result = B.integrate_observations(r, (a, b))
    assert result.grounding_status == 'multiple_candidates'
    assert len(result.regions) == 2
    assert {m['backend_score'] for m in result.backend_metadata} == {.01, .99}
    assert 'multiple_candidate_ambiguity' in result.uncertainty
    assert result.provenance['winner_selection'] is False
    assert result.to_json() == B.integrate_observations(r, (b, a)).to_json()


def test_exact_duplicates_coalesce_distinct_observations_do_not():
    r = request()
    a = raw(r)
    assert len(B.integrate_observations(r, (a, a)).regions) == 1
    b = replace(a, raw_reference='other backend record')
    assert len(B.integrate_observations(r, (a, b)).regions) == 2


def test_empty_completed_response_not_absence():
    result = B.integrate_observations(request(), ())
    assert result.grounding_status == 'no_candidate' and not result.regions
    assert 'no_candidate_not_physical_absence' in result.uncertainty
    assert result.provenance['physical_truth_claim'] is False


@pytest.mark.parametrize('status', ['unavailable', 'indeterminate'])
def test_noncompleted_responses(status):
    r = request()
    result = B.integrate_observations(r, (), response_status=status)
    assert result.grounding_status == status
    assert result.provenance['external_grounding_evidence'] is False
    with pytest.raises(ValueError):
        B.integrate_observations(r, (raw(r),), response_status=status)


@pytest.mark.parametrize('observations', [None, {}, 'no results'])
def test_response_must_be_explicit_sequence(observations):
    with pytest.raises(ValueError):
        B.integrate_observations(request(), observations)


def test_malformed_mixed_response_fail_fast():
    r = request()
    valid, bad = raw(r), raw(r, 'point', (9999, 0), 'pixel')
    with pytest.raises(ValueError):
        B.integrate_observations(r, (valid, bad))


def test_uncertainty_and_lineage():
    r = request()
    observation = raw(r, uncertainty=('backend_limit',))
    result = B.integrate_observations(r, (observation,))
    assert set(r.uncertainty) | {'backend_limit'} <= set(result.uncertainty)
    assert result.source_plan_id == r.source_plan_id
    assert result.source_target_id == r.source_target_id
    assert result.source_request_id == r.request_id
    assert result.frame == r.frame
    assert result.regions[0].source_observation_id == observation.observation_id
    assert result.provenance['source_topological_state_id'] == r.provenance['source_topological_state_id']


def test_all_contracts_frozen_detached_deterministic():
    p, r = plan(), request()
    o = raw(r, provenance={'audit': {'kind': 'synthetic'}})
    result = B.integrate_observations(r, (o,))
    for record in (r.frame, r, o, result.regions[0], result):
        assert record.schema_version == VERSION
        with pytest.raises(FrozenInstanceError):
            record.schema_version = 'bad'
        with pytest.raises(TypeError):
            record.provenance['x'] = 1
        data = record.to_dict()
        data['provenance']['x'] = 1
        assert 'x' not in record.provenance
        assert record.to_dict() == json.loads(record.to_json())
        assert record.to_json() == replace(record).to_json()
    before = p.to_json(), r.to_json(), o.to_json()
    B.integrate_observations(r, (o,))
    assert before == (p.to_json(), r.to_json(), o.to_json())


@pytest.mark.parametrize('record_kind', ['frame', 'request', 'observation'])
def test_wrong_schema_rejected(record_kind):
    r = request()
    if record_kind == 'frame':
        f = frame()
        object.__setattr__(f, 'schema_version', 'wrong')
        with pytest.raises(ValueError): B.create_request(plan(), plan().targets[0].target_id, f)
    elif record_kind == 'request':
        object.__setattr__(r, 'schema_version', 'wrong')
        with pytest.raises(ValueError): B.integrate_observations(r, ())
    else:
        o = raw(r)
        object.__setattr__(o, 'schema_version', 'wrong')
        with pytest.raises(ValueError): B.integrate_observations(r, (o,))


def test_payloads_and_backend_instances_rejected():
    for metadata in ({'raw_image': 'encoded'}, {'opaque': object()}, {'payload': b'bytes'}):
        with pytest.raises(ValueError): raw(provenance=metadata)


def test_query_cannot_invent_semantic_phrase():
    with pytest.raises(ValueError):
        replace(request(), symbolic_query={'text': 'person behind car'})


def test_no_network_filesystem_or_backend_calls():
    r, o = request(), raw()
    with patch.object(socket, 'socket', side_effect=AssertionError('network')), patch('builtins.open', side_effect=AssertionError('filesystem')):
        result = B.integrate_observations(r, (o,))
    assert all(result.provenance[k] == v for k, v in POLICY.items())
    assert result.provenance['external_grounding_evidence'] is True
    assert not {'command', 'action', 'pan', 'tilt', 'trajectory', 'pose', 'depth'} & result.to_dict().keys()


@pytest.mark.parametrize('change', [dict(normalized_coordinates=(0, 0, .5, .5)),
                                  dict(pixel_coordinates=(0, 0, 641, 480)),
                                  dict(frame_width=0)])
def test_inconsistent_region_rejected(change):
    r = request()
    region = B.integrate_observations(r, (raw(r),)).regions[0]
    with pytest.raises(ValueError):
        replace(region, **change)


def test_result_backend_lineage_required():
    r = request()
    result = B.integrate_observations(r, (raw(r),))
    with pytest.raises(ValueError):
        replace(result, backend_metadata=())


@pytest.mark.parametrize('status', ['confirmed', 'hidden_actor_found', 'multiple_candidates', 'no_candidate'])
def test_result_status_must_match_candidate_count(status):
    r = request()
    result = B.integrate_observations(r, (raw(r),))
    with pytest.raises(ValueError):
        replace(result, grounding_status=status)


def test_request_creation_preserves_inputs_and_output_type_order():
    p, f = plan(), frame()
    before = p.to_json(), f.to_json()
    a = B.create_request(p, p.targets[0].target_id, f, ('point', 'box'))
    b = B.create_request(p, p.targets[0].target_id, f, ('box', 'point'))
    assert a.to_json() == b.to_json()
    assert before == (p.to_json(), f.to_json())


def test_unknown_coordinate_space_rejected():
    with pytest.raises(ValueError):
        raw(space='world_xyz')
