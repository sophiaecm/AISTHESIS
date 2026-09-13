"""Tests for the AISTHESIS Scientific Input Adapter v0.1.

These tests validate the application-layer adapter that connects
a PDF input to the existing frozen Mini-Lab pipeline.

The tests do not validate scientific truth or real-paper scientific
understanding accuracy.
"""

from pathlib import Path

import pytest

from fifth_layer.scientific_document import (
    DocumentElement,
    ScientificDocumentState,
    ScientificPage,
)
from fifth_layer.scientific_evidence_bridge import (
    ScientificEvidenceBridgeResult,
)
from fifth_layer.scientific_input_adapter import (
    ScientificInputAdapter,
    ScientificInputResult,
    ScientificInputSummary,
)
from fifth_layer.scientific_relationships import (
    ScientificRelationshipGraph,
)
from fifth_layer.scientific_structure import (
    ScientificStructureState,
)
from fifth_layer.world_model.common_evidence_state import (
    CommonEvidenceState,
)
from fifth_layer.world_model.evidence import (
    EvidenceBundle,
    EvidenceSource,
)


class FakeScientificDocumentParser:
    """Deterministic parser substitute for adapter unit tests."""

    def __init__(self, document):
        self.document = document
        self.received_path = None

    def parse(self, path):
        self.received_path = path
        return self.document


def make_document():
    elements = (
        DocumentElement(
            element_id="element-hypothesis",
            element_type="text_block",
            page_index=0,
            bbox=(0.0, 0.0, 500.0, 40.0),
            text_content=(
                "We hypothesize that visual occlusion "
                "will increase prediction error."
            ),
            source="adapter_test",
        ),
        DocumentElement(
            element_id="element-method",
            element_type="text_block",
            page_index=0,
            bbox=(0.0, 50.0, 500.0, 90.0),
            text_content=(
                "We used a controlled experiment to compare "
                "prediction performance."
            ),
            source="adapter_test",
        ),
        DocumentElement(
            element_id="element-variable",
            element_type="text_block",
            page_index=0,
            bbox=(0.0, 100.0, 500.0, 140.0),
            text_content=(
                "The measured prediction error was the "
                "dependent variable."
            ),
            source="adapter_test",
        ),
        DocumentElement(
            element_id="element-dataset",
            element_type="text_block",
            page_index=0,
            bbox=(0.0, 150.0, 500.0, 190.0),
            text_content=(
                "The dataset contained controlled video sequences."
            ),
            source="adapter_test",
        ),
        DocumentElement(
            element_id="element-result",
            element_type="text_block",
            page_index=0,
            bbox=(0.0, 200.0, 500.0, 240.0),
            text_content=(
                "The results showed that prediction error "
                "increased under occlusion."
            ),
            source="adapter_test",
        ),
        DocumentElement(
            element_id="element-claim",
            element_type="text_block",
            page_index=0,
            bbox=(0.0, 250.0, 500.0, 290.0),
            text_content=(
                "These findings suggest that partial observability "
                "affects predictive performance."
            ),
            source="adapter_test",
        ),
        DocumentElement(
            element_id="element-limitation",
            element_type="text_block",
            page_index=0,
            bbox=(0.0, 300.0, 500.0, 340.0),
            text_content=(
                "A limitation is the restricted sample size."
            ),
            source="adapter_test",
        ),
    )

    page = ScientificPage(
        page_index=0,
        width=600.0,
        height=800.0,
        elements=elements,
    )

    return ScientificDocumentState(
        document_id="adapter-doc-001",
        source_metadata={
            "source_reference": "synthetic_adapter_test.pdf",
        },
        pages=(page,),
    )


def make_adapter():
    document = make_document()
    parser = FakeScientificDocumentParser(document)

    adapter = ScientificInputAdapter(
        parser=parser,
    )

    return adapter, parser


def run_adapter():
    adapter, parser = make_adapter()

    result = adapter.analyze_pdf(
        "synthetic_adapter_test.pdf",
        scene_id="adapter-scene-001",
        session_id="adapter-session-001",
        timestamp=50.0,
    )

    return result, parser


def test_adapter_returns_scientific_input_result():
    result, _ = run_adapter()

    assert isinstance(
        result,
        ScientificInputResult,
    )


def test_adapter_preserves_document_state():
    result, _ = run_adapter()

    assert isinstance(
        result.document,
        ScientificDocumentState,
    )

    assert result.document_id == "adapter-doc-001"


def test_adapter_produces_structure_state():
    result, _ = run_adapter()

    assert isinstance(
        result.structures,
        ScientificStructureState,
    )

    assert result.structures.candidates


def test_adapter_produces_relationship_graph():
    result, _ = run_adapter()

    assert isinstance(
        result.relationships,
        ScientificRelationshipGraph,
    )

    assert result.relationships.relationships


def test_adapter_produces_bridge_result():
    result, _ = run_adapter()

    assert isinstance(
        result.bridge_result,
        ScientificEvidenceBridgeResult,
    )


def test_adapter_produces_common_evidence_state():
    result, _ = run_adapter()

    assert isinstance(
        result.common_evidence_state,
        CommonEvidenceState,
    )


def test_adapter_uses_common_documentary_family():
    result, _ = run_adapter()

    state = result.common_evidence_state

    assert "documentary" in state.present_families

    assert (
        state.availability["documentary"]
        == "available"
    )


def test_adapter_preserves_scene_identity():
    result, _ = run_adapter()

    assert result.scene_id == "adapter-scene-001"

    assert (
        result.bridge_result.scene_id
        == "adapter-scene-001"
    )

    assert (
        result.common_evidence_state.scene_id
        == "adapter-scene-001"
    )


def test_adapter_preserves_session_identity():
    result, _ = run_adapter()

    assert (
        result.common_evidence_state.session_id
        == "adapter-session-001"
    )

    assert all(
        item.provenance.get("session_id")
        == "adapter-session-001"
        for item in result.evidence_items
    )


def test_adapter_preserves_timestamp():
    result, _ = run_adapter()

    assert (
        result.common_evidence_state.timestamp
        == 50.0
    )

    assert all(
        item.timestamp == 50.0
        for item in result.evidence_items
    )


def test_adapter_passes_path_to_parser():
    _, parser = run_adapter()

    assert parser.received_path == Path(
        "synthetic_adapter_test.pdf"
    )


def test_adapter_rejects_empty_path():
    adapter, _ = make_adapter()

    with pytest.raises(ValueError):
        adapter.analyze_pdf("")


def test_adapter_rejects_non_pdf_path():
    adapter, _ = make_adapter()

    with pytest.raises(ValueError):
        adapter.analyze_pdf(
            "scientific_document.txt"
        )


def test_adapter_rejects_non_path_input():
    adapter, _ = make_adapter()

    with pytest.raises(ValueError):
        adapter.analyze_pdf(123)


def test_adapter_accepts_path_object():
    adapter, parser = make_adapter()

    result = adapter.analyze_pdf(
        Path("synthetic_adapter_test.pdf"),
        scene_id="adapter-scene-001",
        session_id="adapter-session-001",
        timestamp=50.0,
    )

    assert isinstance(
        result,
        ScientificInputResult,
    )

    assert parser.received_path == Path(
        "synthetic_adapter_test.pdf"
    )


def test_adapter_rejects_blank_scene_id():
    adapter, _ = make_adapter()

    with pytest.raises(ValueError):
        adapter.analyze_pdf(
            "synthetic_adapter_test.pdf",
            scene_id="",
        )


def test_adapter_rejects_blank_session_id():
    adapter, _ = make_adapter()

    with pytest.raises(ValueError):
        adapter.analyze_pdf(
            "synthetic_adapter_test.pdf",
            session_id="",
        )


def test_adapter_creates_default_scene_id():
    adapter, _ = make_adapter()

    result = adapter.analyze_pdf(
        "synthetic_adapter_test.pdf",
        session_id="adapter-session-001",
        timestamp=50.0,
    )

    assert (
        result.scene_id
        == "document:adapter-doc-001"
    )


def test_adapter_creates_default_session_id():
    adapter, _ = make_adapter()

    result = adapter.analyze_pdf(
        "synthetic_adapter_test.pdf",
        scene_id="adapter-scene-001",
        timestamp=50.0,
    )

    assert (
        result.common_evidence_state.session_id
        == "scientific:adapter-doc-001"
    )


def test_summary_is_created():
    result, _ = run_adapter()

    assert isinstance(
        result.summary,
        ScientificInputSummary,
    )


def test_summary_contains_hypothesis():
    result, _ = run_adapter()

    assert result.summary.hypotheses

    assert any(
        "hypothesize" in text.lower()
        for text in result.summary.hypotheses
    )


def test_summary_contains_method():
    result, _ = run_adapter()

    assert result.summary.methods


def test_summary_contains_variable():
    result, _ = run_adapter()

    assert result.summary.variables


def test_summary_contains_dataset():
    result, _ = run_adapter()

    assert result.summary.datasets


def test_summary_contains_result():
    result, _ = run_adapter()

    assert result.summary.results


def test_summary_contains_claim():
    result, _ = run_adapter()

    assert result.summary.claims


def test_summary_contains_limitation():
    result, _ = run_adapter()

    assert result.summary.limitations


def test_summary_contains_relationships():
    result, _ = run_adapter()

    assert result.summary.relationships


def test_summary_relationships_are_ui_safe_strings():
    result, _ = run_adapter()

    assert all(
        isinstance(value, str)
        for value in result.summary.relationships
    )

    assert all(
        "--" in value and "-->" in value
        for value in result.summary.relationships
    )


def test_summary_to_dict_is_plain_data():
    result, _ = run_adapter()

    data = result.summary.to_dict()

    assert isinstance(data, dict)
    assert isinstance(data["hypotheses"], list)
    assert isinstance(data["results"], list)
    assert isinstance(data["relationships"], list)


def test_summary_counts_structures():
    result, _ = run_adapter()

    assert result.summary.structure_count > 0


def test_summary_counts_relationships():
    result, _ = run_adapter()

    assert (
        result.summary.relationship_count
        == len(result.summary.relationships)
    )

    assert result.summary.relationship_count > 0


def test_result_is_reported_result_not_observed():
    result, _ = run_adapter()

    result_items = [
        item
        for item in result.evidence_items
        if item.value.get("structure_type")
        == "result"
    ]

    assert result_items

    assert all(
        item.epistemic_status
        == "reported_result"
        for item in result_items
    )

    assert all(
        item.epistemic_status != "observed"
        for item in result_items
    )


def test_claim_is_author_claim():
    result, _ = run_adapter()

    claim_items = [
        item
        for item in result.evidence_items
        if item.value.get("structure_type")
        == "claim"
    ]

    assert claim_items

    assert all(
        item.epistemic_status == "author_claim"
        for item in claim_items
    )


def test_hypothesis_is_not_promoted():
    result, _ = run_adapter()

    hypothesis_items = [
        item
        for item in result.evidence_items
        if item.value.get("structure_type")
        == "hypothesis"
    ]

    assert hypothesis_items

    assert all(
        item.epistemic_status == "unknown"
        for item in hypothesis_items
    )


def test_relationships_remain_aisthesis_inference():
    result, _ = run_adapter()

    relationship_items = [
        item
        for item in result.evidence_items
        if item.evidence_type.startswith(
            "scientific_relationship:"
        )
    ]

    assert relationship_items

    assert all(
        item.epistemic_status
        == "aisthesis_inference"
        for item in relationship_items
    )


def test_relationships_do_not_create_support_semantics():
    result, _ = run_adapter()

    relationship_items = [
        item
        for item in result.evidence_items
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
    result, _ = run_adapter()

    relationship_items = [
        item
        for item in result.evidence_items
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


def test_adapter_does_not_create_truth_confidence():
    result, _ = run_adapter()

    assert all(
        item.confidence is None
        for item in result.evidence_items
    )


def test_adapter_preserves_truth_decision_boundary():
    result, _ = run_adapter()

    assert (
        result.common_evidence_state
        .integrity["truth_decision"]
        == "not_performed"
    )


def test_documentary_evidence_needs_no_coordinate_frame():
    result, _ = run_adapter()

    assert (
        result.common_evidence_state
        .coordinate_frame_id
        is None
    )

    assert (
        "coordinate_frame_unspecified"
        not in result.common_evidence_state
        .uncertainty["integration"]
    )


def test_adapter_schema_version():
    result, _ = run_adapter()

    assert (
        result.schema_version
        == "scientific-input-adapter-v0.1"
    )


def test_adapter_is_deterministic():
    first, _ = run_adapter()
    second, _ = run_adapter()

    assert (
        first.structures.to_json()
        == second.structures.to_json()
    )

    assert (
        first.relationships.to_json()
        == second.relationships.to_json()
    )

    assert (
        first.bridge_result.evidence_bundle.items
        == second.bridge_result.evidence_bundle.items
    )

    assert (
        first.common_evidence_state.to_json()
        == second.common_evidence_state.to_json()
    )

    assert (
        first.summary.to_dict()
        == second.summary.to_dict()
    )


def test_result_contract_rejects_empty_source_path():
    result, _ = run_adapter()

    with pytest.raises(ValueError):
        ScientificInputResult(
            source_path="",
            document=result.document,
            structures=result.structures,
            relationships=result.relationships,
            bridge_result=result.bridge_result,
            common_evidence_state=(
                result.common_evidence_state
            ),
            summary=result.summary,
        )


def test_result_contract_rejects_wrong_document_type():
    result, _ = run_adapter()

    with pytest.raises(ValueError):
        ScientificInputResult(
            source_path="test.pdf",
            document="not-document",
            structures=result.structures,
            relationships=result.relationships,
            bridge_result=result.bridge_result,
            common_evidence_state=(
                result.common_evidence_state
            ),
            summary=result.summary,
        )


def test_result_contract_rejects_wrong_summary_type():
    result, _ = run_adapter()

    with pytest.raises(ValueError):
        ScientificInputResult(
            source_path="test.pdf",
            document=result.document,
            structures=result.structures,
            relationships=result.relationships,
            bridge_result=result.bridge_result,
            common_evidence_state=(
                result.common_evidence_state
            ),
            summary="not-summary",
        )


def test_all_final_evidence_is_documentary():
    result, _ = run_adapter()

    assert result.evidence_items

    assert all(
        item.source_type
        == EvidenceSource.DOCUMENTARY
        for item in result.evidence_items
    )


def test_adapter_preserves_end_to_end_epistemic_separation():
    result, _ = run_adapter()

    statuses = {
        item.epistemic_status
        for item in result.evidence_items
    }

    assert "reported_result" in statuses
    assert "author_claim" in statuses
    assert "aisthesis_inference" in statuses

    assert (
        result.common_evidence_state
        .integrity["truth_decision"]
        == "not_performed"
    )