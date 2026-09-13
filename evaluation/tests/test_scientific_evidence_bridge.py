import pytest

from fifth_layer.scientific_document import (
    DocumentElement,
    ScientificDocumentState,
    ScientificPage,
)
from fifth_layer.scientific_evidence_bridge import (
    ScientificEvidenceBridge,
    ScientificEvidenceBridgeResult,
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
from fifth_layer.world_model.evidence import (
    EvidenceBundle,
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


def make_structures(
    *candidates,
    document_id="doc-001",
):
    return ScientificStructureState(
        document_id=document_id,
        candidates=tuple(candidates),
    )


def make_relationship(
    *,
    document_id="doc-001",
    relationship_type=ScientificRelationshipType.SUPPORTS,
):
    source = ScientificGraphNodeRef(
        node_id="result-001",
        node_kind=ScientificNodeKind.STRUCTURE_CANDIDATE,
        document_id=document_id,
    )

    target = ScientificGraphNodeRef(
        node_id="hypothesis-001",
        node_kind=ScientificNodeKind.STRUCTURE_CANDIDATE,
        document_id=document_id,
    )

    return ScientificRelationshipCandidate(
        relationship_id="relationship-001",
        relationship_type=relationship_type,
        status=ScientificRelationshipStatus.HEURISTIC,
        source=source,
        target=target,
        document_id=document_id,
        source_element_ids=("element-001",),
        confidence=None,
        inference_method="test_rule",
    )


def make_relationship_graph(
    *relationships,
    document_id="doc-001",
):
    return ScientificRelationshipGraph(
        document_id=document_id,
        relationships=tuple(relationships),
    )


def build_single_structure(structure_type):
    document = make_document()

    candidate = make_candidate(
        "candidate-001",
        structure_type,
    )

    structures = make_structures(candidate)

    relationships = make_relationship_graph()

    return ScientificEvidenceBridge().build(
        document,
        structures,
        relationships,
        session_id="session-001",
        timestamp=10.0,
    )


def test_bridge_result_requires_document_id():
    bundle = EvidenceBundle(
        scene_id="scene-001",
        items=(),
    )

    with pytest.raises(ValueError):
        ScientificEvidenceBridgeResult(
            document_id="",
            scene_id="scene-001",
            evidence_bundle=bundle,
        )


def test_bridge_result_requires_scene_id():
    bundle = EvidenceBundle(
        scene_id="scene-001",
        items=(),
    )

    with pytest.raises(ValueError):
        ScientificEvidenceBridgeResult(
            document_id="doc-001",
            scene_id="",
            evidence_bundle=bundle,
        )


def test_bridge_result_requires_bundle():
    with pytest.raises(ValueError):
        ScientificEvidenceBridgeResult(
            document_id="doc-001",
            scene_id="scene-001",
            evidence_bundle="not-a-bundle",
        )


def test_bridge_result_rejects_scene_mismatch():
    bundle = EvidenceBundle(
        scene_id="other-scene",
        items=(),
    )

    with pytest.raises(ValueError):
        ScientificEvidenceBridgeResult(
            document_id="doc-001",
            scene_id="scene-001",
            evidence_bundle=bundle,
        )


def test_bridge_rejects_non_document():
    with pytest.raises(ValueError):
        ScientificEvidenceBridge().build(
            "not-document",
            make_structures(),
            make_relationship_graph(),
        )


def test_bridge_rejects_non_structure_state():
    with pytest.raises(ValueError):
        ScientificEvidenceBridge().build(
            make_document(),
            "not-structures",
            make_relationship_graph(),
        )


def test_bridge_rejects_non_relationship_graph():
    with pytest.raises(ValueError):
        ScientificEvidenceBridge().build(
            make_document(),
            make_structures(),
            "not-relationships",
        )


def test_bridge_rejects_structure_document_mismatch():
    with pytest.raises(ValueError):
        ScientificEvidenceBridge().build(
            make_document("doc-a"),
            make_structures(document_id="doc-b"),
            make_relationship_graph(document_id="doc-a"),
        )


def test_bridge_rejects_relationship_document_mismatch():
    with pytest.raises(ValueError):
        ScientificEvidenceBridge().build(
            make_document("doc-a"),
            make_structures(document_id="doc-a"),
            make_relationship_graph(document_id="doc-b"),
        )


def test_bridge_generates_default_document_scene_id():
    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(),
        make_relationship_graph(),
    )

    assert result.scene_id == "document:doc-001"


def test_bridge_accepts_explicit_scene_id():
    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(),
        make_relationship_graph(),
        scene_id="mini-lab-scene",
    )

    assert result.scene_id == "mini-lab-scene"

    assert (
        result.evidence_bundle.scene_id
        == "mini-lab-scene"
    )


def test_bridge_rejects_blank_scene_id():
    with pytest.raises(ValueError):
        ScientificEvidenceBridge().build(
            make_document(),
            make_structures(),
            make_relationship_graph(),
            scene_id="",
        )


def test_bridge_rejects_blank_session_id():
    with pytest.raises(ValueError):
        ScientificEvidenceBridge().build(
            make_document(),
            make_structures(),
            make_relationship_graph(),
            session_id="",
        )


@pytest.mark.parametrize(
    ("structure_type", "expected_status"),
    [
        (
            ScientificStructureType.RESEARCH_QUESTION,
            "unknown",
        ),
        (
            ScientificStructureType.HYPOTHESIS,
            "unknown",
        ),
        (
            ScientificStructureType.METHOD,
            "method",
        ),
        (
            ScientificStructureType.VARIABLE,
            "measurement",
        ),
        (
            ScientificStructureType.DATASET,
            "dataset_reference",
        ),
        (
            ScientificStructureType.RESULT,
            "reported_result",
        ),
        (
            ScientificStructureType.CLAIM,
            "author_claim",
        ),
        (
            ScientificStructureType.LIMITATION,
            "limitation",
        ),
        (
            ScientificStructureType.UNKNOWN,
            "unknown",
        ),
    ],
)
def test_structure_epistemic_mapping(
    structure_type,
    expected_status,
):
    result = build_single_structure(structure_type)

    assert len(result.evidence_items) == 1

    item = result.evidence_items[0]

    assert item.epistemic_status == expected_status


@pytest.mark.parametrize(
    "structure_type",
    list(ScientificStructureType),
)
def test_all_structure_evidence_is_documentary(
    structure_type,
):
    result = build_single_structure(structure_type)

    assert (
        result.evidence_items[0].source_type
        == EvidenceSource.DOCUMENTARY
    )


def test_reported_result_is_not_observed_fact():
    result = build_single_structure(
        ScientificStructureType.RESULT
    )

    item = result.evidence_items[0]

    assert item.epistemic_status == "reported_result"
    assert item.epistemic_status != "observed"


def test_claim_is_author_claim():
    result = build_single_structure(
        ScientificStructureType.CLAIM
    )

    item = result.evidence_items[0]

    assert item.epistemic_status == "author_claim"


def test_hypothesis_is_not_promoted_to_fact():
    result = build_single_structure(
        ScientificStructureType.HYPOTHESIS
    )

    item = result.evidence_items[0]

    assert item.epistemic_status == "unknown"
    assert item.confidence is None


def test_structure_confidence_is_not_promoted():
    result = build_single_structure(
        ScientificStructureType.RESULT
    )

    item = result.evidence_items[0]

    assert item.confidence is None


def test_structure_preserves_candidate_identity():
    result = build_single_structure(
        ScientificStructureType.RESULT
    )

    item = result.evidence_items[0]

    assert (
        item.value["candidate_id"]
        == "candidate-001"
    )


def test_structure_preserves_source_elements():
    result = build_single_structure(
        ScientificStructureType.RESULT
    )

    item = result.evidence_items[0]

    assert (
        tuple(item.value["source_element_ids"])
        == ("element-001",)
    )


def test_structure_marks_verification_as_not_verified():
    result = build_single_structure(
        ScientificStructureType.RESULT
    )

    item = result.evidence_items[0]

    assert (
        item.value["verification_status"]
        == "not_verified"
    )


def test_relationship_becomes_aisthesis_inference():
    relationship = make_relationship()

    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(),
        make_relationship_graph(relationship),
    )

    assert len(result.evidence_items) == 1

    item = result.evidence_items[0]

    assert (
        item.epistemic_status
        == "aisthesis_inference"
    )


def test_supports_relationship_does_not_populate_supports():
    relationship = make_relationship(
        relationship_type=(
            ScientificRelationshipType.SUPPORTS
        )
    )

    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(),
        make_relationship_graph(relationship),
    )

    item = result.evidence_items[0]

    assert item.supports == ()
    assert item.supporting_evidence_ids == ()


def test_contradicts_relationship_does_not_populate_contradicts():
    relationship = make_relationship(
        relationship_type=(
            ScientificRelationshipType.CONTRADICTS
        )
    )

    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(),
        make_relationship_graph(relationship),
    )

    item = result.evidence_items[0]

    assert item.contradicts == ()
    assert item.opposing_evidence_ids == ()


def test_relationship_confidence_is_not_promoted():
    relationship = make_relationship()

    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(),
        make_relationship_graph(relationship),
    )

    assert result.evidence_items[0].confidence is None


def test_relationship_preserves_type():
    relationship = make_relationship(
        relationship_type=(
            ScientificRelationshipType.SUPPORTS
        )
    )

    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(),
        make_relationship_graph(relationship),
    )

    item = result.evidence_items[0]

    assert (
        item.value["relationship_type"]
        == "supports"
    )


def test_relationship_preserves_status():
    relationship = make_relationship()

    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(),
        make_relationship_graph(relationship),
    )

    item = result.evidence_items[0]

    assert (
        item.value["relationship_status"]
        == "heuristic"
    )


def test_relationship_marks_verification_not_verified():
    relationship = make_relationship()

    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(),
        make_relationship_graph(relationship),
    )

    item = result.evidence_items[0]

    assert (
        item.value["verification_status"]
        == "not_verified"
    )


def test_bridge_preserves_timestamp():
    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(
            make_candidate(
                "result-001",
                ScientificStructureType.RESULT,
            )
        ),
        make_relationship_graph(),
        timestamp=42.0,
    )

    assert result.evidence_items[0].timestamp == 42.0


def test_bridge_preserves_session_in_provenance():
    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(
            make_candidate(
                "result-001",
                ScientificStructureType.RESULT,
            )
        ),
        make_relationship_graph(),
        session_id="session-xyz",
    )

    assert (
        result.evidence_items[0]
        .provenance["session_id"]
        == "session-xyz"
    )


def test_bundle_provenance_declares_no_truth_decision():
    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(),
        make_relationship_graph(),
    )

    assert (
        result.evidence_bundle
        .provenance["truth_decision"]
        == "not_performed"
    )


def test_structure_provenance_declares_no_truth_decision():
    result = build_single_structure(
        ScientificStructureType.RESULT
    )

    assert (
        result.evidence_items[0]
        .provenance["truth_decision"]
        == "not_performed"
    )


def test_relationship_provenance_declares_no_truth_decision():
    relationship = make_relationship()

    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(),
        make_relationship_graph(relationship),
    )

    assert (
        result.evidence_items[0]
        .provenance["truth_decision"]
        == "not_performed"
    )


def test_structure_evidence_has_no_support_semantics():
    result = build_single_structure(
        ScientificStructureType.RESULT
    )

    item = result.evidence_items[0]

    assert item.supports == ()
    assert item.contradicts == ()
    assert item.supporting_evidence_ids == ()
    assert item.opposing_evidence_ids == ()


def test_empty_inputs_produce_empty_bundle():
    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(),
        make_relationship_graph(),
    )

    assert result.evidence_items == ()


def test_bridge_is_deterministic():
    document = make_document()

    structures = make_structures(
        make_candidate(
            "result-001",
            ScientificStructureType.RESULT,
        ),
        make_candidate(
            "claim-001",
            ScientificStructureType.CLAIM,
        ),
    )

    relationships = make_relationship_graph()

    bridge = ScientificEvidenceBridge()

    first = bridge.build(
        document,
        structures,
        relationships,
        session_id="session-001",
        timestamp=10.0,
    )

    second = bridge.build(
        document,
        structures,
        relationships,
        session_id="session-001",
        timestamp=10.0,
    )

    assert (
        first.evidence_bundle.items
        == second.evidence_bundle.items
    )


def test_evidence_ids_are_deterministic():
    document = make_document()

    structures = make_structures(
        make_candidate(
            "result-001",
            ScientificStructureType.RESULT,
        )
    )

    relationships = make_relationship_graph()

    bridge = ScientificEvidenceBridge()

    first = bridge.build(
        document,
        structures,
        relationships,
    )

    second = bridge.build(
        document,
        structures,
        relationships,
    )

    assert (
        first.evidence_items[0].evidence_id
        == second.evidence_items[0].evidence_id
    )


def test_multiple_structures_remain_separate_evidence_items():
    structures = make_structures(
        make_candidate(
            "result-001",
            ScientificStructureType.RESULT,
        ),
        make_candidate(
            "claim-001",
            ScientificStructureType.CLAIM,
        ),
    )

    result = ScientificEvidenceBridge().build(
        make_document(),
        structures,
        make_relationship_graph(),
    )

    assert len(result.evidence_items) == 2

    statuses = {
        item.epistemic_status
        for item in result.evidence_items
    }

    assert statuses == {
        "reported_result",
        "author_claim",
    }


def test_structure_and_relationship_remain_distinct():
    structures = make_structures(
        make_candidate(
            "result-001",
            ScientificStructureType.RESULT,
        )
    )

    relationship = make_relationship()

    result = ScientificEvidenceBridge().build(
        make_document(),
        structures,
        make_relationship_graph(relationship),
    )

    assert len(result.evidence_items) == 2

    statuses = {
        item.epistemic_status
        for item in result.evidence_items
    }

    assert "reported_result" in statuses
    assert "aisthesis_inference" in statuses


def test_result_schema_version():
    result = ScientificEvidenceBridge().build(
        make_document(),
        make_structures(),
        make_relationship_graph(),
    )

    assert (
        result.schema_version
        == "scientific-evidence-bridge-v0.1"
    )