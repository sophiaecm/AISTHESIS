"""Step 26C tests for the injected LocateAnything backend bridge.

These tests use only synthetic GroundingRequest records and an injected fake
runtime. They do not load a model, access a network, or perform real visual
grounding.
"""
from dataclasses import FrozenInstanceError, replace
import math

import pytest

from fifth_layer.backends.locateanything_backend import (
    BACKEND,
    MODEL,
    VERSION,
    LocateAnythingBackendBridge,
    LocateAnythingInvocation,
    LocateAnythingOutputParser,
    LocateAnythingRawResponse,
)
from fifth_layer.world_model.spatial_grounding import (
    GroundingFrame,
    GroundingRequest,
    SpatialGroundingAdapter,
)


def make_frame():
    return GroundingFrame(
        image_id="image-001",
        width=640,
        height=480,
        timestamp=10.0,
        session_id="session-001",
        coordinate_frame_id="camera-001",
        scene_id="scene-001",
        provenance={"source": "synthetic-step26c-test"},
    )


def make_request(outputs=("box",), *, scope="object"):
    frame = make_frame()
    object_ids = ("o1",) if scope == "object" else ("o1", "o2")
    relation_types = () if scope != "relation" else ("contact_candidate",)
    if scope == "relation":
        object_ids = ("o1", "o2")
    symbolic_query = {
        "object_ids": tuple(sorted(object_ids)),
        "relation_types": tuple(sorted(relation_types)),
        "target_scope": scope,
        "semantic_query_status": "unavailable",
    }
    return GroundingRequest(
        source_plan_id="plan-001",
        source_target_id="target-001",
        target_scope=scope,
        object_ids=object_ids,
        relation_types=relation_types,
        symbolic_query=symbolic_query,
        frame=frame,
        requested_output_types=outputs,
        uncertainty=("synthetic_input",),
        provenance={
            "source": "synthetic-step26c-test",
            "semantic_query_status": "unavailable",
        },
    )


def make_response(invocation, *, text=None, payload=None, metadata=None):
    return LocateAnythingRawResponse(
        invocation_id=invocation.invocation_id,
        request_id=invocation.request_id,
        frame_id=invocation.frame_id,
        response_text=text,
        structured_payload=payload,
        backend_name=BACKEND,
        backend_model=MODEL,
        backend_version="synthetic-runtime-v0",
        backend_metadata=metadata or {"runtime": "fake"},
        uncertainty=("synthetic_backend",),
        provenance={"source": "fake-runtime"},
    )


class FakeRuntime:
    def __init__(self, response_factory):
        self.response_factory = response_factory
        self.calls = 0
        self.invocations = []

    def infer(self, invocation):
        self.calls += 1
        self.invocations.append(invocation)
        return self.response_factory(invocation)


def test_invocation_is_immutable_and_schema_is_correct():
    request = make_request()
    invocation = LocateAnythingBackendBridge().create_invocation(
        request, "local-image-reference", "red cup"
    )
    assert invocation.schema_version == VERSION
    assert invocation.backend_name == BACKEND
    assert invocation.backend_model == MODEL
    with pytest.raises(FrozenInstanceError):
        invocation.query = "changed"


def test_response_is_immutable_and_schema_is_correct():
    request = make_request()
    invocation = LocateAnythingBackendBridge().create_invocation(
        request, "image-ref", "red cup"
    )
    response = make_response(invocation, text="<box><1><2><3><4></box>")
    assert response.schema_version == VERSION
    with pytest.raises(FrozenInstanceError):
        response.response_text = "changed"


def test_invocation_identity_is_deterministic():
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    a = bridge.create_invocation(request, "image-ref", "red cup")
    b = bridge.create_invocation(request, "image-ref", "red cup")
    assert a.invocation_id == b.invocation_id
    assert a.to_dict() == b.to_dict()
    assert a.to_json() == b.to_json()


def test_serialized_output_is_detached():
    request = make_request()
    invocation = LocateAnythingBackendBridge().create_invocation(
        request, "image-ref", "red cup"
    )
    data = invocation.to_dict()
    data["provenance"]["query_source"] = "tampered"
    assert invocation.provenance["query_source"] == "caller_provided"


def test_request_is_not_mutated_by_invocation_creation():
    request = make_request()
    before = request.to_json()
    LocateAnythingBackendBridge().create_invocation(request, "image-ref", "red cup")
    assert request.to_json() == before


@pytest.mark.parametrize(
    ("outputs", "task"),
    [
        (("box",), "grounding"),
        (("point",), "pointing"),
        (("box", "point"), "grounding_and_pointing"),
    ],
)
def test_requested_output_types_are_preserved(outputs, task):
    request = make_request(outputs)
    invocation = LocateAnythingBackendBridge().create_invocation(
        request, "image-ref", "explicit query"
    )
    assert invocation.requested_output_types == tuple(sorted(outputs))
    assert invocation.task_type == task


def test_explicit_query_preserved_exactly_and_marked_caller_provided():
    request = make_request()
    query = "the leftmost chair"
    invocation = LocateAnythingBackendBridge().create_invocation(
        request, "image-ref", query
    )
    assert invocation.query == query
    assert invocation.provenance["query_source"] == "caller_provided"


@pytest.mark.parametrize("query", [None, "", "   "])
def test_missing_or_invalid_query_is_rejected_before_runtime(query):
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    runtime = FakeRuntime(
        lambda inv: make_response(inv, text="<box><1><2><3><4></box>")
    )
    with pytest.raises((ValueError, TypeError)):
        bridge.ground(request, "image-ref", runtime, query)
    assert runtime.calls == 0


def test_symbolic_ids_are_not_expanded_into_semantics():
    request = make_request()
    invocation = LocateAnythingBackendBridge().create_invocation(
        request, "image-ref", "explicit caller wording"
    )
    assert invocation.query == "explicit caller wording"
    assert "o1" not in invocation.query


def test_image_reference_is_preserved_as_opaque_reference(tmp_path):
    request = make_request()
    missing = tmp_path / "does-not-exist.png"
    invocation = LocateAnythingBackendBridge().create_invocation(
        request, str(missing), "red cup"
    )
    assert invocation.image_reference == str(missing)
    assert not missing.exists()


def test_runtime_called_exactly_once():
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    runtime = FakeRuntime(
        lambda inv: make_response(inv, text="<box><10><20><30><40></box>")
    )
    observations = bridge.ground(request, "image-ref", runtime, "red cup")
    assert runtime.calls == 1
    assert len(observations) == 1


def test_invalid_runtime_is_rejected():
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "red cup")
    with pytest.raises(ValueError):
        bridge.run(invocation, object())


def test_wrong_runtime_response_type_is_rejected():
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "red cup")

    class WrongRuntime:
        def infer(self, invocation):
            return {"not": "a response"}

    with pytest.raises((ValueError, TypeError)):
        bridge.run(invocation, WrongRuntime())


@pytest.mark.parametrize("field", ["request_id", "frame_id", "invocation_id"])
def test_response_lineage_mismatch_rejected(field):
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "red cup")
    response = make_response(invocation, text="<box><1><2><3><4></box>")
    bad = replace(response, **{field: "mismatch"})
    runtime = FakeRuntime(lambda _: bad)
    with pytest.raises(ValueError):
        bridge.run(invocation, runtime)
    assert runtime.calls == 1


def test_invocation_request_lineage_mismatch_rejected_during_parse():
    request = make_request()
    other = make_request(("point",))
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(other, "image-ref", "red cup")
    response = make_response(invocation, text="<box><1><2><3><4></box>")
    with pytest.raises(ValueError):
        bridge.parse(request, invocation, response)


def test_valid_text_box_parsed_as_normalized_observation():
    request = make_request(("box",))
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "red cup")
    response = make_response(invocation, text="<box><10><20><900><1000></box>")
    (obs,) = bridge.parse(request, invocation, response)
    assert obs.output_type == "box"
    assert obs.coordinates == (10, 20, 900, 1000)
    assert obs.coordinate_space == "normalized_0_1000"
    assert obs.backend_score is None


def test_valid_text_point_uses_upstream_box_wrapper_with_two_coordinates():
    request = make_request(("point",))
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "red cup")
    response = make_response(invocation, text="<box><0><1000></box>")
    (obs,) = bridge.parse(request, invocation, response)
    assert obs.output_type == "point"
    assert obs.coordinates == (0, 1000)


def test_boundary_coordinates_zero_and_1000_are_accepted():
    request = make_request(("box",))
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "red cup")
    response = make_response(invocation, text="<box><0><0><1000><1000></box>")
    (obs,) = bridge.parse(request, invocation, response)
    assert obs.coordinates == (0, 0, 1000, 1000)


@pytest.mark.parametrize(
    "text",
    [
        "<box><1><2><3></box>",
        "<box><1><2><3><4><5></box>",
        "<box><-1><2><3><4></box>",
        "<box><1><2><3><1001></box>",
        "<box><10><20><5><40></box>",
        "<box><10><20><30><5></box>",
        "<box><nan><20><30><40></box>",
        "<box><inf><20><30><40></box>",
        "<box>1,2,3,4</box>",
        "<point><1><2></point>",
    ],
)
def test_malformed_or_unsupported_text_coordinates_rejected(text):
    request = make_request(("box", "point"))
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(invocation, text=text)
    with pytest.raises(ValueError):
        bridge.parse(request, invocation, response)


@pytest.mark.parametrize(
    "text",
    [
        "there are 4 objects and confidence is 800",
        "the object is around 300 pixels wide",
        "coordinates might be 10,20,30,40",
    ],
)
def test_arbitrary_prose_numbers_are_not_parsed(text):
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(invocation, text=text)
    with pytest.raises(ValueError):
        bridge.parse(request, invocation, response)


def test_prose_may_surround_complete_explicit_grounding_record():
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(
        invocation, text="candidate: <box><10><20><30><40></box> done"
    )
    (obs,) = bridge.parse(request, invocation, response)
    assert obs.coordinates == (10, 20, 30, 40)


def test_multiple_text_boxes_are_preserved_without_winner_selection():
    request = make_request(("box",))
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(
        invocation,
        text="<box><10><20><30><40></box><box><50><60><70><80></box>",
    )
    observations = bridge.parse(request, invocation, response)
    assert len(observations) == 2
    assert {o.coordinates for o in observations} == {
        (10, 20, 30, 40),
        (50, 60, 70, 80),
    }


def test_structured_box_and_point_are_both_preserved_when_requested():
    request = make_request(("box", "point"))
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(
        invocation,
        payload={
            "status": "completed",
            "results": [
                {"type": "box", "coordinates": [10, 20, 30, 40]},
                {"type": "point", "coordinates": [500, 600]},
            ],
        },
    )
    observations = bridge.parse(request, invocation, response)
    assert {o.output_type for o in observations} == {"box", "point"}


def test_unrequested_structured_output_type_is_rejected():
    request = make_request(("box",))
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(
        invocation,
        payload={
            "status": "completed",
            "results": [{"type": "point", "coordinates": [10, 20]}],
        },
    )
    with pytest.raises(ValueError):
        bridge.parse(request, invocation, response)


@pytest.mark.parametrize(
    "coordinates",
    [
        [1, 2, 3],
        [-1, 2, 3, 4],
        [1, 2, 3, 1001],
        [10, 20, 5, 40],
        [10, 20, 30, 5],
    ],
)
def test_invalid_structured_coordinates_are_rejected(coordinates):
    request = make_request(("box",))
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(
        invocation,
        payload={
            "status": "completed",
            "results": [{"type": "box", "coordinates": coordinates}],
        },
    )
    with pytest.raises((ValueError, TypeError)):
        bridge.parse(request, invocation, response)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_structured_coordinates_rejected_at_raw_response_boundary(value):
    request = make_request(("box",))
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    with pytest.raises(ValueError, match="finite number"):
        make_response(
            invocation,
            payload={
                "status": "completed",
                "results": [{"type": "box", "coordinates": [value, 20, 30, 40]}],
            },
        )


def test_malformed_mixed_structured_response_fails_whole_parse():
    request = make_request(("box",))
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(
        invocation,
        payload={
            "status": "completed",
            "results": [
                {"type": "box", "coordinates": [10, 20, 30, 40]},
                {"type": "box", "coordinates": [100, 100, 50, 200]},
            ],
        },
    )
    with pytest.raises(ValueError):
        bridge.parse(request, invocation, response)


def test_structured_score_is_metadata_only_and_does_not_choose_winner():
    request = make_request(("box",))
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(
        invocation,
        payload={
            "status": "completed",
            "results": [
                {
                    "type": "box",
                    "coordinates": [10, 20, 30, 40],
                    "score": 0.1,
                    "label": "candidate-a",
                },
                {
                    "type": "box",
                    "coordinates": [50, 60, 70, 80],
                    "score": 0.99,
                    "label": "candidate-b",
                },
            ],
        },
    )
    observations = bridge.parse(request, invocation, response)
    assert len(observations) == 2
    assert {o.backend_score for o in observations} == {0.1, 0.99}
    for obs in observations:
        assert obs.provenance["backend_score_promoted"] is False
        assert "confidence" not in obs.provenance
        assert "posterior" not in obs.provenance
        assert "risk" not in obs.provenance


def test_absent_score_stays_absent():
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(invocation, text="<box><1><2><3><4></box>")
    (obs,) = bridge.parse(request, invocation, response)
    assert obs.backend_score is None


def test_explicit_text_no_candidate_returns_empty_tuple():
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(invocation, text="<no_candidate/>")
    assert bridge.parse(request, invocation, response) == ()


def test_structured_no_candidate_returns_empty_tuple():
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(
        invocation, payload={"status": "no_candidate", "results": []}
    )
    assert bridge.parse(request, invocation, response) == ()


@pytest.mark.parametrize("text", [None, "", "ordinary prose without grounding"])
def test_missing_or_nonexplicit_output_is_not_no_candidate(text):
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(invocation, text=text)
    with pytest.raises(ValueError):
        bridge.parse(request, invocation, response)


def test_empty_completed_structured_payload_is_not_no_candidate():
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(
        invocation, payload={"status": "completed", "results": []}
    )
    with pytest.raises(ValueError):
        bridge.parse(request, invocation, response)


def test_no_candidate_with_results_is_rejected():
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(
        invocation,
        payload={
            "status": "no_candidate",
            "results": [{"type": "box", "coordinates": [1, 2, 3, 4]}],
        },
    )
    with pytest.raises(ValueError):
        bridge.parse(request, invocation, response)


def test_text_and_structured_coordinate_representations_cannot_compete():
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(
        invocation,
        text="<box><1><2><3><4></box>",
        payload={
            "status": "completed",
            "results": [{"type": "box", "coordinates": [1, 2, 3, 4]}],
        },
    )
    with pytest.raises(ValueError):
        bridge.parse(request, invocation, response)


def test_observation_lineage_and_safety_provenance_are_preserved():
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(invocation, text="<box><1><2><3><4></box>")
    (obs,) = bridge.parse(request, invocation, response)
    assert obs.request_id == request.request_id
    assert obs.frame_id == request.frame.frame_id
    assert obs.provenance["invocation_id"] == invocation.invocation_id
    assert obs.provenance["response_id"] == response.response_id
    assert obs.provenance["parsed_from_external_backend"] is True
    assert obs.provenance["physical_truth_claim"] is False
    assert obs.provenance["object_identity_verified"] is False
    assert obs.provenance["hidden_actor_inference"] is False
    assert obs.provenance["control"] is False
    assert obs.provenance["Bayesian_feedback"] is False
    assert obs.provenance["ExperienceLearning_feedback"] is False
    assert obs.provenance["branch_selection"] is False
    assert obs.provenance["world_state_mutation"] is False
    assert obs.provenance["task_level_validation"] is False


def test_uncertainty_contains_external_backend_limitations():
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(invocation, text="<box><1><2><3><4></box>")
    (obs,) = bridge.parse(request, invocation, response)
    expected = {
        "external_backend_output_unverified",
        "grounding_not_physical_truth",
        "backend_label_not_verified_identity",
        "backend_score_not_aisthesis_confidence",
        "no_candidate_not_physical_absence",
        "normalized_2d_not_metric_3d",
        "grounding_not_control",
        "real_backend_not_smoke_tested",
    }
    assert expected <= set(obs.uncertainty)


def test_step26b_integrates_normalized_box_and_owns_pixel_conversion():
    request = make_request(("box",))
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(
        invocation, text="<box><250><250><750><750></box>"
    )
    observations = bridge.parse(request, invocation, response)
    grounded = SpatialGroundingAdapter().integrate_observations(
        request, observations
    )
    assert grounded.grounding_status == "grounded_candidate"
    assert len(grounded.regions) == 1
    region = grounded.regions[0]
    assert region.pixel_coordinates == (160.0, 120.0, 480.0, 360.0)
    assert region.normalized_coordinates == (0.25, 0.25, 0.75, 0.75)


def test_step26b_preserves_multiple_candidates_without_winner():
    request = make_request(("box",))
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(
        invocation,
        payload={
            "status": "completed",
            "results": [
                {"type": "box", "coordinates": [100, 100, 200, 200], "score": 0.9},
                {"type": "box", "coordinates": [300, 300, 400, 400], "score": 0.1},
            ],
        },
    )
    observations = bridge.parse(request, invocation, response)
    grounded = SpatialGroundingAdapter().integrate_observations(
        request, observations
    )
    assert grounded.grounding_status == "multiple_candidates"
    assert len(grounded.regions) == 2
    assert grounded.provenance["winner_selection"] is False


def test_step26b_explicit_no_candidate_does_not_claim_absence():
    request = make_request()
    bridge = LocateAnythingBackendBridge()
    invocation = bridge.create_invocation(request, "image-ref", "query")
    response = make_response(invocation, text="<no_candidate/>")
    observations = bridge.parse(request, invocation, response)
    grounded = SpatialGroundingAdapter().integrate_observations(
        request, observations
    )
    assert grounded.grounding_status == "no_candidate"
    assert "no_candidate_not_physical_absence" in grounded.uncertainty
    assert grounded.provenance["physical_truth_claim"] is False


def test_no_transformers_or_huggingface_dependency_in_backend_source():
    from pathlib import Path
    import fifth_layer.backends.locateanything_backend as module

    source = Path(module.__file__).read_text(encoding="utf-8-sig")
    assert "import transformers" not in source
    assert "from transformers" not in source
    assert "import huggingface_hub" not in source
    assert "from huggingface_hub" not in source


def test_no_network_subprocess_or_model_loader_symbols_in_backend_source():
    from pathlib import Path
    import fifth_layer.backends.locateanything_backend as module

    source = Path(module.__file__).read_text(encoding="utf-8-sig")
    forbidden = (
        "requests.",
        "urllib.",
        "httpx.",
        "subprocess.",
        "os.system",
        "from_pretrained",
        "snapshot_download",
        "hf_hub_download",
        "git clone",
    )
    for token in forbidden:
        assert token not in source


def test_bridge_has_no_control_or_world_model_mutation_api():
    public = {
        name
        for name in dir(LocateAnythingBackendBridge)
        if not name.startswith("_")
    }
    forbidden = {
        "pan",
        "tilt",
        "zoom",
        "move",
        "navigate",
        "actuate",
        "update_belief",
        "update_experience",
        "select_branch",
        "mutate_world_state",
    }
    assert not (public & forbidden)
