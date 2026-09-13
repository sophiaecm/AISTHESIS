import pytest

from fifth_layer.scientific_document import (
    DocumentElement,
    ScientificDocumentState,
    ScientificPage,
)
from fifth_layer.scientific_evidence_bridge import (
    ScientificEvidenceBridge,
)
from fifth_layer.scientific_relationships import (
    ScientificGraphNodeRef,
    ScientificNodeKind,
    ScientificRelationshipCandidate,
    ScientificRelationshipGraph,
    ScientificRelationshipStatus,
    ScientificRelationshipType,
)
from fifth_layer.scientific_structure import (
    ScientificStructureCandidate,
    ScientificStructureState,
    ScientificStructureStatus,
    ScientificStructureType,
)
from fifth_layer.world_model.common_evidence_state import (
    CommonEvidenceState,
)
from fifth_layer.world_model.evidence import (
    EvidenceSource,
)


def make_document(document_id="doc-001"):
    element = DocumentElement(
        element_id="element-001",
        element_type="text_block",
        page_index=0,
        bbox=(0.0, 0.0, 100.0, 20.0),
        text_content="Scientific text.",
        source="test",
    )

    page = ScientificPage(
        page_index=0,
        width=100.0,
        height=100.0,
        elements=(element,),
    )

    return ScientificDocumentState(
        document_id=document_id,
        source_metadata={
            "source_reference": "synthetic.pdf",
        },
        pages=(page,),
    )


def make_candidate(
    candidate_id,
    structure_type,
    *,
    document_id="doc-001",
    text=None,
):
    if text is None:
        text = f"Example {structure_type.value}."

    return ScientificStructureCandidate(
        candidate_id=candidate_id,
        structure_type=structure_type,
        status=ScientificStructureStatus.HEURISTIC,
        text=text,
        document_id=document_id,
        source_element_ids=("element-001",),
        page_number=1,
        confidence=None,
        extraction_method="test",
    )


def make_relationship(
    *,
    document_id="doc-001",
    relationship_type=ScientificRelationshipType.SUPPORTS,
):
    return ScientificRelationshipCandidate(
        relationship_id="relationship-001",
        relationship_type=relationship_type,
        status=ScientificRelationshipStatus.HEURISTIC,
        source=ScientificGraphNodeRef(
            node_id="result-001",
            node_kind=ScientificNodeKind.STRUCTURE_CANDIDATE,
            document_id=document_id,
        ),
        target=ScientificGraphNodeRef(
            node_id="hypothesis-001",
            node_kind=ScientificNodeKind.STRUCTURE_CANDIDATE,
            document_id=document_id,
        ),
        document_id=document_id,
        source_element_ids=("element-001",),
        confidence=None,
        inference_method="test_rule",
    )


def build_bridge_result(
    *,
    include_relationship=False,
    timestamp=10.0,
):
    document = make_document()

    result_candidate = make_candidate(
        "result-001",
        ScientificStructureType.RESULT,
        text="X improved Y.",
    )

    hypothesis_candidate = make_candidate(
        "hypothesis-001",
        ScientificStructureType.HYPOTHESIS,
        text="X may improve Y.",
    )

    claim_candidate = make_candidate(
        "claim-001",
        ScientificStructureType.CLAIM,
        text="The authors conclude that X improves Y.",
    )

    structures = ScientificStructureState(
        document_id=document.document_id,
        candidates=(
            result_candidate,
            hypothesis_candidate,
            claim_candidate,
        ),
    )

    relationships = ()

    if include_relationship:
        relationships = (
            make_relationship(),
        )

    graph = ScientificRelationshipGraph(
        document_id=document.document_id,
        relationships=relationships,
    )

    return ScientificEvidenceBridge().build(
        document,
        structures,
        graph,
        scene_id="scientific-scene-001",
        session_id="scientific-session-001",
        timestamp=timestamp,
    )


def make_common_state(
    bridge_result,
    *,
    timestamp=10.0,
):
    return CommonEvidenceState(
        scene_id=bridge_result.scene_id,
        session_id="scientific-session-001",
        timestamp=timestamp,
        evidence=bridge_result.evidence_bundle,
        coordinate_frame_id=None,
        present_families=("documentary",),
        provenance={
            "document_id": bridge_result.document_id,
            "session_id": "scientific-session-001",
            "timestamp": timestamp,
            "source_component": (
                "scientific_evidence_bridge_v0.1"
            ),
            "truth_decision": "not_performed",
        },
    )


def test_bridge_bundle_enters_common_evidence_state():
    bridge_result = build_bridge_result()

    state = make_common_state(bridge_result)

    assert isinstance(state, CommonEvidenceState)


def test_documentary_family_is_present():
    state = make_common_state(
        build_bridge_result()
    )

    assert "documentary" in state.present_families


def test_documentary_family_is_available():
    state = make_common_state(
        build_bridge_result()
    )

    assert (
        state.availability["documentary"]
        == "available"
    )


def test_bridge_items_survive_common_state():
    bridge_result = build_bridge_result()

    state = make_common_state(bridge_result)

    assert (
        len(state.evidence_items)
        == len(bridge_result.evidence_items)
    )


def test_all_items_remain_documentary():
    state = make_common_state(
        build_bridge_result()
    )

    assert all(
        item.source_type == EvidenceSource.DOCUMENTARY
        for item in state.evidence_items
    )


def test_reported_result_survives_common_state():
    state = make_common_state(
        build_bridge_result()
    )

    statuses = {
        item.epistemic_status
        for item in state.evidence_items
    }

    assert "reported_result" in statuses


def test_author_claim_survives_common_state():
    state = make_common_state(
        build_bridge_result()
    )

    statuses = {
        item.epistemic_status
        for item in state.evidence_items
    }

    assert "author_claim" in statuses


def test_hypothesis_remains_unknown():
    state = make_common_state(
        build_bridge_result()
    )

    hypothesis_items = [
        item
        for item in state.evidence_items
        if item.value.get("structure_type")
        == "hypothesis"
    ]

    assert len(hypothesis_items) == 1

    assert (
        hypothesis_items[0].epistemic_status
        == "unknown"
    )


def test_common_state_does_not_create_truth_confidence():
    state = make_common_state(
        build_bridge_result()
    )

    assert all(
        item.confidence is None
        for item in state.evidence_items
    )


def test_common_state_preserves_no_truth_decision():
    state = make_common_state(
        build_bridge_result()
    )

    assert (
        state.integrity["truth_decision"]
        == "not_performed"
    )


def test_relationship_survives_as_aisthesis_inference():
    state = make_common_state(
        build_bridge_result(
            include_relationship=True
        )
    )

    inferred_items = [
        item
        for item in state.evidence_items
        if item.epistemic_status
        == "aisthesis_inference"
    ]

    assert len(inferred_items) == 1


def test_supports_edge_does_not_become_support_semantics():
    state = make_common_state(
        build_bridge_result(
            include_relationship=True
        )
    )

    inferred_items = [
        item
        for item in state.evidence_items
        if item.epistemic_status
        == "aisthesis_inference"
    ]

    assert len(inferred_items) == 1

    item = inferred_items[0]

    assert (
        item.value["relationship_type"]
        == "supports"
    )

    assert item.supports == ()
    assert item.supporting_evidence_ids == ()


def test_relationship_does_not_increase_confidence():
    state = make_common_state(
        build_bridge_result(
            include_relationship=True
        )
    )

    inferred_items = [
        item
        for item in state.evidence_items
        if item.epistemic_status
        == "aisthesis_inference"
    ]

    assert inferred_items[0].confidence is None


def test_source_references_preserve_documentary_family():
    state = make_common_state(
        build_bridge_result()
    )

    assert state.source_references

    assert all(
        reference["source_family"]
        == "documentary"
        for reference
        in state.source_references.values()
    )


def test_source_references_preserve_epistemic_status():
    state = make_common_state(
        build_bridge_result()
    )

    statuses = {
        reference["epistemic_status"]
        for reference
        in state.source_references.values()
    }

    assert "reported_result" in statuses
    assert "author_claim" in statuses
    assert "unknown" in statuses


def test_common_state_preserves_document_provenance():
    state = make_common_state(
        build_bridge_result()
    )

    assert (
        state.provenance["document_id"]
        == "doc-001"
    )


def test_common_state_preserves_session_context():
    state = make_common_state(
        build_bridge_result()
    )

    assert (
        state.session_id
        == "scientific-session-001"
    )


def test_common_state_preserves_timestamp():
    state = make_common_state(
        build_bridge_result()
    )

    assert state.timestamp == 10.0


def test_future_documentary_evidence_is_rejected():
    bridge_result = build_bridge_result(
        timestamp=20.0
    )

    with pytest.raises(ValueError):
        make_common_state(
            bridge_result,
            timestamp=10.0,
        )


def test_documentary_evidence_needs_no_coordinate_frame():
    state = make_common_state(
        build_bridge_result()
    )

    assert state.coordinate_frame_id is None


def test_documentary_input_does_not_create_frame_warning():
    state = make_common_state(
        build_bridge_result()
    )

    integration_issues = set(
        state.uncertainty["integration"]
    )

    assert (
        "coordinate_frame_unspecified"
        not in integration_issues
    )


def test_bridge_and_common_state_are_deterministic():
    first_bridge = build_bridge_result()
    second_bridge = build_bridge_result()

    first = make_common_state(first_bridge)
    second = make_common_state(second_bridge)

    assert first.to_json() == second.to_json()


def test_common_state_serializes_documentary_evidence():
    state = make_common_state(
        build_bridge_result(
            include_relationship=True
        )
    )

    serialized = state.to_json()

    assert '"documentary"' in serialized
    assert '"reported_result"' in serialized
    assert '"author_claim"' in serialized
    assert '"aisthesis_inference"' in serialized


def test_common_state_does_not_convert_claim_to_result():
    state = make_common_state(
        build_bridge_result()
    )

    claim_items = [
        item
        for item in state.evidence_items
        if item.value.get("structure_type")
        == "claim"
    ]

    assert len(claim_items) == 1

    assert (
        claim_items[0].epistemic_status
        == "author_claim"
    )


def test_common_state_does_not_convert_result_to_observation():
    state = make_common_state(
        build_bridge_result()
    )

    result_items = [
        item
        for item in state.evidence_items
        if item.value.get("structure_type")
        == "result"
    ]

    assert len(result_items) == 1

    assert (
        result_items[0].epistemic_status
        == "reported_result"
    )

    assert (
        result_items[0].epistemic_status
        != "observed"
    )


def test_common_state_preserves_relationship_verification_status():
    state = make_common_state(
        build_bridge_result(
            include_relationship=True
        )
    )

    relationship_items = [
        item
        for item in state.evidence_items
        if item.epistemic_status
        == "aisthesis_inference"
    ]

    assert (
        relationship_items[0]
        .value["verification_status"]
        == "not_verified"
    )


def test_no_item_is_automatically_supporting_evidence():
    state = make_common_state(
        build_bridge_result(
            include_relationship=True
        )
    )

    assert all(
        item.supporting_evidence_ids == ()
        for item in state.evidence_items
    )


def test_no_item_is_automatically_opposing_evidence():
    state = make_common_state(
        build_bridge_result(
            include_relationship=True
        )
    )

    assert all(
        item.opposing_evidence_ids == ()
        for item in state.evidence_items
    )


def test_common_state_remains_evidence_inventory_not_belief():
    state = make_common_state(
        build_bridge_result(
            include_relationship=True
        )
    )

    assert (
        state.integrity["truth_decision"]
        == "not_performed"
    )

    assert all(
        item.confidence is None
        for item in state.evidence_items
    )