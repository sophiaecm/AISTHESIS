"""End-to-end tests for the AISTHESIS Mini-Lab scientific pipeline.

Pipeline under test:

ScientificDocumentState
    -> ScientificStructureBuilder
    -> ScientificStructureState
    -> ScientificRelationshipBuilder
    -> ScientificRelationshipGraph
    -> ScientificEvidenceBridge
    -> EvidenceBundle
    -> CommonEvidenceState

These tests validate engineering integration and epistemic
preservation.

They do not validate scientific truth or real-world scientific
understanding accuracy.
"""

from fifth_layer.scientific_document import (
    DocumentElement,
    ScientificDocumentState,
    ScientificPage,
)
from fifth_layer.scientific_structure import (
    ScientificStructureBuilder,
    ScientificStructureType,
)
from fifth_layer.scientific_relationships import (
    ScientificRelationshipBuilder,
)
from fifth_layer.scientific_evidence_bridge import (
    ScientificEvidenceBridge,
)
from fifth_layer.world_model.common_evidence_state import (
    CommonEvidenceState,
)
from fifth_layer.world_model.evidence import (
    EvidenceSource,
)


DOCUMENT_ID = "minilab-e2e-document-001"
SCENE_ID = "minilab-e2e-scene-001"
SESSION_ID = "minilab-e2e-session-001"
TIMESTAMP = 100.0


def make_element(
    element_id,
    text,
    *,
    page_index=0,
    element_type="text_block",
):
    return DocumentElement(
        element_id=element_id,
        element_type=element_type,
        page_index=page_index,
        bbox=(0.0, 0.0, 500.0, 40.0),
        text_content=text,
        source="synthetic_end_to_end_test",
    )


def make_scientific_document():
    """Create a controlled synthetic scientific document.

    The text intentionally contains lexical signals already handled
    by ScientificStructureBuilder v0.1.

    The test does not introduce new semantic rules.
    """

    elements = (
        make_element(
            "element-hypothesis",
            (
                "We hypothesize that increased visual occlusion "
                "will increase prediction error."
            ),
        ),
        make_element(
            "element-method",
            (
                "We used a controlled experiment to compare "
                "prediction performance under visible and "
                "occluded conditions."
            ),
        ),
        make_element(
            "element-variable",
            (
                "The measured prediction error was the primary "
                "dependent variable."
            ),
        ),
        make_element(
            "element-dataset",
            (
                "The dataset contained controlled video sequences "
                "of object interactions."
            ),
        ),
        make_element(
            "element-result",
            (
                "The results showed that prediction error increased "
                "under visual occlusion."
            ),
        ),
        make_element(
            "element-claim",
            (
                "These findings suggest that partial observability "
                "affects predictive performance."
            ),
        ),
        make_element(
            "element-limitation",
            (
                "A limitation is the restricted sample size."
            ),
        ),
    )

    page = ScientificPage(
        page_index=0,
        width=600.0,
        height=800.0,
        elements=elements,
    )

    return ScientificDocumentState(
        document_id=DOCUMENT_ID,
        source_metadata={
            "source_reference": "synthetic_minilab_e2e.pdf",
            "source_kind": "synthetic_test_document",
        },
        pages=(page,),
    )


def run_minilab_pipeline():
    """Run the complete currently implemented Mini-Lab pipeline."""

    document = make_scientific_document()

    structure_builder = ScientificStructureBuilder()
    structures = structure_builder.build(document)

    relationship_builder = ScientificRelationshipBuilder()
    relationships = relationship_builder.build(
        document,
        structures,
    )

    bridge = ScientificEvidenceBridge()
    bridge_result = bridge.build(
        document,
        structures,
        relationships,
        scene_id=SCENE_ID,
        session_id=SESSION_ID,
        timestamp=TIMESTAMP,
    )

    common_state = CommonEvidenceState(
        scene_id=SCENE_ID,
        session_id=SESSION_ID,
        timestamp=TIMESTAMP,
        evidence=bridge_result.evidence_bundle,
        coordinate_frame_id=None,
        present_families=("documentary",),
        provenance={
            "document_id": DOCUMENT_ID,
            "session_id": SESSION_ID,
            "timestamp": TIMESTAMP,
            "source_component": "minilab_end_to_end_test",
            "truth_decision": "not_performed",
        },
    )

    return (
        document,
        structures,
        relationships,
        bridge_result,
        common_state,
    )


def test_end_to_end_pipeline_runs():
    (
        document,
        structures,
        relationships,
        bridge_result,
        common_state,
    ) = run_minilab_pipeline()

    assert document.document_id == DOCUMENT_ID
    assert structures.document_id == DOCUMENT_ID
    assert relationships.document_id == DOCUMENT_ID
    assert bridge_result.document_id == DOCUMENT_ID

    assert common_state.scene_id == SCENE_ID
    assert common_state.session_id == SESSION_ID


def test_document_elements_enter_structure_stage():
    document, structures, _, _, _ = run_minilab_pipeline()

    assert document.elements
    assert structures.candidates


def test_structure_stage_detects_hypothesis():
    _, structures, _, _, _ = run_minilab_pipeline()

    types = {
        candidate.structure_type
        for candidate in structures.candidates
    }

    assert ScientificStructureType.HYPOTHESIS in types


def test_structure_stage_detects_method():
    _, structures, _, _, _ = run_minilab_pipeline()

    types = {
        candidate.structure_type
        for candidate in structures.candidates
    }

    assert ScientificStructureType.METHOD in types


def test_structure_stage_detects_variable():
    _, structures, _, _, _ = run_minilab_pipeline()

    types = {
        candidate.structure_type
        for candidate in structures.candidates
    }

    assert ScientificStructureType.VARIABLE in types


def test_structure_stage_detects_dataset():
    _, structures, _, _, _ = run_minilab_pipeline()

    types = {
        candidate.structure_type
        for candidate in structures.candidates
    }

    assert ScientificStructureType.DATASET in types


def test_structure_stage_detects_result():
    _, structures, _, _, _ = run_minilab_pipeline()

    types = {
        candidate.structure_type
        for candidate in structures.candidates
    }

    assert ScientificStructureType.RESULT in types


def test_structure_stage_detects_claim():
    _, structures, _, _, _ = run_minilab_pipeline()

    types = {
        candidate.structure_type
        for candidate in structures.candidates
    }

    assert ScientificStructureType.CLAIM in types


def test_structure_stage_detects_limitation():
    _, structures, _, _, _ = run_minilab_pipeline()

    types = {
        candidate.structure_type
        for candidate in structures.candidates
    }

    assert ScientificStructureType.LIMITATION in types


def test_structure_candidates_preserve_document_identity():
    _, structures, _, _, _ = run_minilab_pipeline()

    assert all(
        candidate.document_id == DOCUMENT_ID
        for candidate in structures.candidates
    )


def test_structure_candidates_preserve_source_elements():
    _, structures, _, _, _ = run_minilab_pipeline()

    assert all(
        candidate.source_element_ids
        for candidate in structures.candidates
    )


def test_relationship_stage_produces_candidates():
    _, _, relationships, _, _ = run_minilab_pipeline()

    assert relationships.relationships


def test_relationships_preserve_document_identity():
    _, _, relationships, _, _ = run_minilab_pipeline()

    assert all(
        relationship.document_id == DOCUMENT_ID
        for relationship in relationships.relationships
    )


def test_relationships_remain_unverified():
    _, _, relationships, _, _ = run_minilab_pipeline()

    assert all(
        relationship.confidence is None
        for relationship in relationships.relationships
    )


def test_relationships_preserve_source_provenance():
    _, _, relationships, _, _ = run_minilab_pipeline()

    assert all(
        relationship.source_element_ids
        for relationship in relationships.relationships
    )


def test_bridge_produces_evidence():
    _, _, _, bridge_result, _ = run_minilab_pipeline()

    assert bridge_result.evidence_items


def test_all_bridge_evidence_is_documentary():
    _, _, _, bridge_result, _ = run_minilab_pipeline()

    assert all(
        item.source_type == EvidenceSource.DOCUMENTARY
        for item in bridge_result.evidence_items
    )


def test_result_becomes_reported_result():
    _, _, _, bridge_result, _ = run_minilab_pipeline()

    result_items = [
        item
        for item in bridge_result.evidence_items
        if item.value.get("structure_type") == "result"
    ]

    assert result_items

    assert all(
        item.epistemic_status == "reported_result"
        for item in result_items
    )


def test_result_never_becomes_observed_fact():
    _, _, _, bridge_result, _ = run_minilab_pipeline()

    result_items = [
        item
        for item in bridge_result.evidence_items
        if item.value.get("structure_type") == "result"
    ]

    assert result_items

    assert all(
        item.epistemic_status != "observed"
        for item in result_items
    )


def test_claim_becomes_author_claim():
    _, _, _, bridge_result, _ = run_minilab_pipeline()

    claim_items = [
        item
        for item in bridge_result.evidence_items
        if item.value.get("structure_type") == "claim"
    ]

    assert claim_items

    assert all(
        item.epistemic_status == "author_claim"
        for item in claim_items
    )


def test_hypothesis_is_not_promoted_to_supported_fact():
    _, _, _, bridge_result, _ = run_minilab_pipeline()

    hypothesis_items = [
        item
        for item in bridge_result.evidence_items
        if item.value.get("structure_type") == "hypothesis"
    ]

    assert hypothesis_items

    assert all(
        item.epistemic_status == "unknown"
        for item in hypothesis_items
    )

    assert all(
        item.confidence is None
        for item in hypothesis_items
    )


def test_relationships_become_aisthesis_inference():
    _, _, _, bridge_result, _ = run_minilab_pipeline()

    relationship_items = [
        item
        for item in bridge_result.evidence_items
        if item.evidence_type.startswith(
            "scientific_relationship:"
        )
    ]

    assert relationship_items

    assert all(
        item.epistemic_status == "aisthesis_inference"
        for item in relationship_items
    )


def test_relationships_do_not_create_support_semantics():
    _, _, _, bridge_result, _ = run_minilab_pipeline()

    relationship_items = [
        item
        for item in bridge_result.evidence_items
        if item.evidence_type.startswith(
            "scientific_relationship:"
        )
    ]

    assert relationship_items

    assert all(
        item.supports == ()
        for item in relationship_items
    )

    assert all(
        item.supporting_evidence_ids == ()
        for item in relationship_items
    )


def test_relationships_do_not_create_contradiction_semantics():
    _, _, _, bridge_result, _ = run_minilab_pipeline()

    relationship_items = [
        item
        for item in bridge_result.evidence_items
        if item.evidence_type.startswith(
            "scientific_relationship:"
        )
    ]

    assert relationship_items

    assert all(
        item.contradicts == ()
        for item in relationship_items
    )

    assert all(
        item.opposing_evidence_ids == ()
        for item in relationship_items
    )


def test_no_bridge_evidence_receives_truth_confidence():
    _, _, _, bridge_result, _ = run_minilab_pipeline()

    assert all(
        item.confidence is None
        for item in bridge_result.evidence_items
    )


def test_bridge_evidence_preserves_verification_boundary():
    _, _, _, bridge_result, _ = run_minilab_pipeline()

    assert all(
        item.value.get("verification_status")
        == "not_verified"
        for item in bridge_result.evidence_items
    )


def test_common_state_receives_documentary_family():
    _, _, _, _, common_state = run_minilab_pipeline()

    assert "documentary" in common_state.present_families

    assert (
        common_state.availability["documentary"]
        == "available"
    )


def test_common_state_preserves_reported_result():
    _, _, _, _, common_state = run_minilab_pipeline()

    statuses = {
        item.epistemic_status
        for item in common_state.evidence_items
    }

    assert "reported_result" in statuses


def test_common_state_preserves_author_claim():
    _, _, _, _, common_state = run_minilab_pipeline()

    statuses = {
        item.epistemic_status
        for item in common_state.evidence_items
    }

    assert "author_claim" in statuses


def test_common_state_preserves_aisthesis_inference():
    _, _, _, _, common_state = run_minilab_pipeline()

    statuses = {
        item.epistemic_status
        for item in common_state.evidence_items
    }

    assert "aisthesis_inference" in statuses


def test_common_state_does_not_perform_truth_decision():
    _, _, _, _, common_state = run_minilab_pipeline()

    assert (
        common_state.integrity["truth_decision"]
        == "not_performed"
    )


def test_common_state_preserves_document_provenance():
    _, _, _, _, common_state = run_minilab_pipeline()

    assert (
        common_state.provenance["document_id"]
        == DOCUMENT_ID
    )


def test_common_state_preserves_source_references():
    _, _, _, bridge_result, common_state = (
        run_minilab_pipeline()
    )

    bridge_ids = {
        item.evidence_id
        for item in bridge_result.evidence_items
    }

    common_ids = set(
        common_state.source_references.keys()
    )

    assert bridge_ids == common_ids


def test_common_state_does_not_require_coordinate_frame():
    _, _, _, _, common_state = run_minilab_pipeline()

    assert common_state.coordinate_frame_id is None

    assert (
        "coordinate_frame_unspecified"
        not in common_state.uncertainty["integration"]
    )


def test_end_to_end_preserves_timestamp():
    _, _, _, bridge_result, common_state = (
        run_minilab_pipeline()
    )

    assert common_state.timestamp == TIMESTAMP

    assert all(
        item.timestamp == TIMESTAMP
        for item in bridge_result.evidence_items
    )


def test_end_to_end_preserves_session_identity():
    _, _, _, bridge_result, common_state = (
        run_minilab_pipeline()
    )

    assert common_state.session_id == SESSION_ID

    assert all(
        item.provenance.get("session_id")
        == SESSION_ID
        for item in bridge_result.evidence_items
    )


def test_end_to_end_is_deterministic():
    first = run_minilab_pipeline()
    second = run_minilab_pipeline()

    first_document = first[0]
    second_document = second[0]

    first_structures = first[1]
    second_structures = second[1]

    first_relationships = first[2]
    second_relationships = second[2]

    first_bridge = first[3]
    second_bridge = second[3]

    first_common = first[4]
    second_common = second[4]

    assert (
        first_document.to_json()
        == second_document.to_json()
    )

    assert (
        first_structures.to_json()
        == second_structures.to_json()
    )

    assert (
        first_relationships.to_json()
        == second_relationships.to_json()
    )

    assert (
        first_bridge.evidence_bundle.items
        == second_bridge.evidence_bundle.items
    )

    assert (
        first_common.to_json()
        == second_common.to_json()
    )


def test_end_to_end_preserves_epistemic_separation():
    _, _, _, _, common_state = run_minilab_pipeline()

    result_items = [
        item
        for item in common_state.evidence_items
        if item.value.get("structure_type") == "result"
    ]

    claim_items = [
        item
        for item in common_state.evidence_items
        if item.value.get("structure_type") == "claim"
    ]

    relationship_items = [
        item
        for item in common_state.evidence_items
        if item.evidence_type.startswith(
            "scientific_relationship:"
        )
    ]

    assert result_items
    assert claim_items
    assert relationship_items

    assert all(
        item.epistemic_status == "reported_result"
        for item in result_items
    )

    assert all(
        item.epistemic_status == "author_claim"
        for item in claim_items
    )

    assert all(
        item.epistemic_status == "aisthesis_inference"
        for item in relationship_items
    )


def test_end_to_end_never_forms_belief():
    _, _, _, _, common_state = run_minilab_pipeline()

    assert (
        common_state.integrity["truth_decision"]
        == "not_performed"
    )

    assert all(
        item.confidence is None
        for item in common_state.evidence_items
    )

    assert all(
        item.supporting_evidence_ids == ()
        for item in common_state.evidence_items
    )

    assert all(
        item.opposing_evidence_ids == ()
        for item in common_state.evidence_items
    )