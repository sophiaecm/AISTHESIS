import pytest

from fifth_layer.scientific_document import (
    DocumentElement,
    ScientificDocumentState,
    ScientificPage,
)

from fifth_layer.scientific_relationships import (
    ScientificGraphNodeRef,
    ScientificNodeKind,
    ScientificRelationshipBuilder,
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


def make_node(
    node_id="node-001",
    document_id="doc-001",
):
    return ScientificGraphNodeRef(
        node_id=node_id,
        node_kind=ScientificNodeKind.STRUCTURE_CANDIDATE,
        document_id=document_id,
    )


def make_relationship(
    relationship_id="rel-001",
    document_id="doc-001",
):
    return ScientificRelationshipCandidate(
        relationship_id=relationship_id,
        relationship_type=ScientificRelationshipType.TESTED_BY,
        status=ScientificRelationshipStatus.CANDIDATE,
        source=make_node(
            "hypothesis-001",
            document_id,
        ),
        target=make_node(
            "method-001",
            document_id,
        ),
        document_id=document_id,
        source_element_ids=(
            "element-001",
            "element-002",
        ),
        confidence=None,
        inference_method="test",
    )


def make_structure(
    candidate_id,
    structure_type,
    page_number=1,
    source_element_id=None,
    document_id="doc-001",
):
    if source_element_id is None:
        source_element_id = (
            f"element-{candidate_id}"
        )

    return ScientificStructureCandidate(
        candidate_id=candidate_id,
        structure_type=structure_type,
        status=ScientificStructureStatus.HEURISTIC,
        text=f"Example {structure_type.value}.",
        document_id=document_id,
        source_element_ids=(
            source_element_id,
        ),
        page_number=page_number,
        confidence=None,
        extraction_method="test",
    )


def make_document(
    document_id="doc-001",
):
    element = DocumentElement(
        element_id="document-element-001",
        element_type="text_block",
        page_index=0,
        bbox=(0.0, 0.0, 100.0, 20.0),
        text_content="Example document.",
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


def make_structure_state(
    *candidates,
    document_id="doc-001",
):
    return ScientificStructureState(
        document_id=document_id,
        candidates=tuple(candidates),
    )


def test_node_can_be_created():
    node = make_node()

    assert node.node_id == "node-001"

    assert (
        node.node_kind
        == ScientificNodeKind.STRUCTURE_CANDIDATE
    )


def test_node_is_immutable():
    node = make_node()

    with pytest.raises(Exception):
        node.node_id = "changed"


def test_node_requires_id():
    with pytest.raises(ValueError):
        make_node(node_id="")


def test_node_requires_document_id():
    with pytest.raises(ValueError):
        make_node(document_id="")


@pytest.mark.parametrize(
    "node_kind",
    [
        ScientificNodeKind.STRUCTURE_CANDIDATE,
        ScientificNodeKind.DOCUMENT_ELEMENT,
    ],
)
def test_all_node_kinds_are_representable(
    node_kind,
):
    node = ScientificGraphNodeRef(
        node_id="node-001",
        node_kind=node_kind,
        document_id="doc-001",
    )

    assert node.node_kind == node_kind


def test_relationship_can_be_created():
    relationship = make_relationship()

    assert relationship.relationship_id == "rel-001"

    assert (
        relationship.relationship_type
        == ScientificRelationshipType.TESTED_BY
    )


def test_relationship_is_immutable():
    relationship = make_relationship()

    with pytest.raises(Exception):
        relationship.confidence = 1.0


def test_relationship_requires_id():
    with pytest.raises(ValueError):
        make_relationship(relationship_id="")


def test_relationship_requires_document_id():
    source = make_node(
        "source",
        "doc-001",
    )

    target = make_node(
        "target",
        "doc-001",
    )

    with pytest.raises(ValueError):
        ScientificRelationshipCandidate(
            relationship_id="rel-001",
            relationship_type=(
                ScientificRelationshipType.TESTED_BY
            ),
            status=(
                ScientificRelationshipStatus.CANDIDATE
            ),
            source=source,
            target=target,
            document_id="",
            source_element_ids=("element-001",),
        )


def test_relationship_rejects_source_document_mismatch():
    with pytest.raises(ValueError):
        ScientificRelationshipCandidate(
            relationship_id="rel-001",
            relationship_type=(
                ScientificRelationshipType.TESTED_BY
            ),
            status=(
                ScientificRelationshipStatus.CANDIDATE
            ),
            source=make_node(
                "source",
                "other-doc",
            ),
            target=make_node(
                "target",
                "doc-001",
            ),
            document_id="doc-001",
            source_element_ids=("element-001",),
        )


def test_relationship_rejects_target_document_mismatch():
    with pytest.raises(ValueError):
        ScientificRelationshipCandidate(
            relationship_id="rel-001",
            relationship_type=(
                ScientificRelationshipType.TESTED_BY
            ),
            status=(
                ScientificRelationshipStatus.CANDIDATE
            ),
            source=make_node(
                "source",
                "doc-001",
            ),
            target=make_node(
                "target",
                "other-doc",
            ),
            document_id="doc-001",
            source_element_ids=("element-001",),
        )


def test_relationship_rejects_self_edge():
    node = make_node(
        "same-node",
        "doc-001",
    )

    with pytest.raises(ValueError):
        ScientificRelationshipCandidate(
            relationship_id="rel-001",
            relationship_type=(
                ScientificRelationshipType.UNKNOWN
            ),
            status=(
                ScientificRelationshipStatus.CANDIDATE
            ),
            source=node,
            target=node,
            document_id="doc-001",
            source_element_ids=("element-001",),
        )


def test_relationship_requires_provenance():
    with pytest.raises(ValueError):
        ScientificRelationshipCandidate(
            relationship_id="rel-001",
            relationship_type=(
                ScientificRelationshipType.TESTED_BY
            ),
            status=(
                ScientificRelationshipStatus.CANDIDATE
            ),
            source=make_node("source"),
            target=make_node("target"),
            document_id="doc-001",
            source_element_ids=(),
        )


def test_relationship_rejects_blank_provenance():
    with pytest.raises(ValueError):
        ScientificRelationshipCandidate(
            relationship_id="rel-001",
            relationship_type=(
                ScientificRelationshipType.TESTED_BY
            ),
            status=(
                ScientificRelationshipStatus.CANDIDATE
            ),
            source=make_node("source"),
            target=make_node("target"),
            document_id="doc-001",
            source_element_ids=("",),
        )


def test_relationship_rejects_duplicate_provenance():
    with pytest.raises(ValueError):
        ScientificRelationshipCandidate(
            relationship_id="rel-001",
            relationship_type=(
                ScientificRelationshipType.TESTED_BY
            ),
            status=(
                ScientificRelationshipStatus.CANDIDATE
            ),
            source=make_node("source"),
            target=make_node("target"),
            document_id="doc-001",
            source_element_ids=(
                "element-001",
                "element-001",
            ),
        )


@pytest.mark.parametrize(
    "confidence",
    [-1.0, -0.01, 1.01, 2.0],
)
def test_relationship_rejects_invalid_confidence(
    confidence,
):
    with pytest.raises(ValueError):
        ScientificRelationshipCandidate(
            relationship_id="rel-001",
            relationship_type=(
                ScientificRelationshipType.TESTED_BY
            ),
            status=(
                ScientificRelationshipStatus.HEURISTIC
            ),
            source=make_node("source"),
            target=make_node("target"),
            document_id="doc-001",
            source_element_ids=("element-001",),
            confidence=confidence,
        )


@pytest.mark.parametrize(
    "relationship_type",
    list(ScientificRelationshipType),
)
def test_all_relationship_types_are_representable(
    relationship_type,
):
    relationship = ScientificRelationshipCandidate(
        relationship_id=(
            f"rel-{relationship_type.value}"
        ),
        relationship_type=relationship_type,
        status=ScientificRelationshipStatus.CANDIDATE,
        source=make_node("source"),
        target=make_node("target"),
        document_id="doc-001",
        source_element_ids=("element-001",),
    )

    assert (
        relationship.relationship_type
        == relationship_type
    )


@pytest.mark.parametrize(
    "status",
    list(ScientificRelationshipStatus),
)
def test_all_relationship_statuses_are_representable(
    status,
):
    relationship = ScientificRelationshipCandidate(
        relationship_id=f"rel-{status.value}",
        relationship_type=(
            ScientificRelationshipType.UNKNOWN
        ),
        status=status,
        source=make_node("source"),
        target=make_node("target"),
        document_id="doc-001",
        source_element_ids=("element-001",),
    )

    assert relationship.status == status


def test_graph_can_be_created():
    relationship = make_relationship()

    graph = ScientificRelationshipGraph(
        document_id="doc-001",
        relationships=(relationship,),
    )

    assert graph.relationships == (
        relationship,
    )

    assert (
        graph.schema_version
        == "scientific-relationship-graph-v0.1"
    )


def test_graph_is_immutable():
    graph = ScientificRelationshipGraph(
        document_id="doc-001",
        relationships=(),
    )

    with pytest.raises(Exception):
        graph.document_id = "changed"


def test_graph_rejects_duplicate_relationship_ids():
    first = make_relationship(
        relationship_id="same-id"
    )

    second = make_relationship(
        relationship_id="same-id"
    )

    with pytest.raises(ValueError):
        ScientificRelationshipGraph(
            document_id="doc-001",
            relationships=(first, second),
        )


def test_graph_rejects_foreign_relationship():
    relationship = make_relationship(
        document_id="other-doc"
    )

    with pytest.raises(ValueError):
        ScientificRelationshipGraph(
            document_id="doc-001",
            relationships=(relationship,),
        )


def test_graph_allows_empty_relationships():
    graph = ScientificRelationshipGraph(
        document_id="doc-001",
        relationships=(),
    )

    assert graph.relationships == ()


def test_relationship_serialization_is_plain():
    result = make_relationship().to_dict()

    assert result["relationship_type"] == "tested_by"
    assert result["status"] == "candidate"

    assert (
        result["source"]["node_kind"]
        == "structure_candidate"
    )


def test_graph_serialization_is_deterministic():
    graph = ScientificRelationshipGraph(
        document_id="doc-001",
        relationships=(make_relationship(),),
    )

    assert graph.to_json() == graph.to_json()


def test_graph_json_contains_schema_version():
    graph = ScientificRelationshipGraph(
        document_id="doc-001",
        relationships=(),
    )

    assert (
        '"schema_version":'
        '"scientific-relationship-graph-v0.1"'
        in graph.to_json()
    )


def test_builder_rejects_non_document():
    structures = make_structure_state()

    with pytest.raises(ValueError):
        ScientificRelationshipBuilder().build(
            "not-document",
            structures,
        )


def test_builder_rejects_non_structure_state():
    document = make_document()

    with pytest.raises(ValueError):
        ScientificRelationshipBuilder().build(
            document,
            "not-structure-state",
        )


def test_builder_rejects_document_mismatch():
    document = make_document(
        document_id="doc-a"
    )

    structures = make_structure_state(
        document_id="doc-b"
    )

    with pytest.raises(ValueError):
        ScientificRelationshipBuilder().build(
            document,
            structures,
        )


def test_builder_hypothesis_tested_by_method():
    hypothesis = make_structure(
        "hypothesis-001",
        ScientificStructureType.HYPOTHESIS,
    )

    method = make_structure(
        "method-001",
        ScientificStructureType.METHOD,
    )

    graph = ScientificRelationshipBuilder().build(
        make_document(),
        make_structure_state(
            hypothesis,
            method,
        ),
    )

    assert len(graph.relationships) == 1

    relationship = graph.relationships[0]

    assert (
        relationship.relationship_type
        == ScientificRelationshipType.TESTED_BY
    )

    assert (
        relationship.source.node_id
        == "hypothesis-001"
    )

    assert (
        relationship.target.node_id
        == "method-001"
    )


def test_builder_method_uses_dataset():
    method = make_structure(
        "method-001",
        ScientificStructureType.METHOD,
    )

    dataset = make_structure(
        "dataset-001",
        ScientificStructureType.DATASET,
    )

    graph = ScientificRelationshipBuilder().build(
        make_document(),
        make_structure_state(
            method,
            dataset,
        ),
    )

    assert len(graph.relationships) == 1

    assert (
        graph.relationships[0].relationship_type
        == ScientificRelationshipType.USES
    )


def test_builder_method_measures_variable():
    method = make_structure(
        "method-001",
        ScientificStructureType.METHOD,
    )

    variable = make_structure(
        "variable-001",
        ScientificStructureType.VARIABLE,
    )

    graph = ScientificRelationshipBuilder().build(
        make_document(),
        make_structure_state(
            method,
            variable,
        ),
    )

    assert (
        graph.relationships[0].relationship_type
        == ScientificRelationshipType.MEASURES
    )


def test_builder_result_about_variable():
    result = make_structure(
        "result-001",
        ScientificStructureType.RESULT,
    )

    variable = make_structure(
        "variable-001",
        ScientificStructureType.VARIABLE,
    )

    graph = ScientificRelationshipBuilder().build(
        make_document(),
        make_structure_state(
            result,
            variable,
        ),
    )

    assert (
        graph.relationships[0].relationship_type
        == ScientificRelationshipType.ABOUT
    )


def test_builder_result_supports_hypothesis_is_heuristic():
    result = make_structure(
        "result-001",
        ScientificStructureType.RESULT,
    )

    hypothesis = make_structure(
        "hypothesis-001",
        ScientificStructureType.HYPOTHESIS,
    )

    graph = ScientificRelationshipBuilder().build(
        make_document(),
        make_structure_state(
            result,
            hypothesis,
        ),
    )

    relationship = graph.relationships[0]

    assert (
        relationship.relationship_type
        == ScientificRelationshipType.SUPPORTS
    )

    assert (
        relationship.status
        == ScientificRelationshipStatus.HEURISTIC
    )

    assert relationship.confidence is None


def test_builder_claim_supported_by_result_is_heuristic():
    claim = make_structure(
        "claim-001",
        ScientificStructureType.CLAIM,
    )

    result = make_structure(
        "result-001",
        ScientificStructureType.RESULT,
    )

    graph = ScientificRelationshipBuilder().build(
        make_document(),
        make_structure_state(
            claim,
            result,
        ),
    )

    relationship = graph.relationships[0]

    assert (
        relationship.relationship_type
        == ScientificRelationshipType.SUPPORTED_BY
    )

    assert (
        relationship.status
        == ScientificRelationshipStatus.HEURISTIC
    )


def test_builder_does_not_link_different_pages():
    hypothesis = make_structure(
        "hypothesis-001",
        ScientificStructureType.HYPOTHESIS,
        page_number=1,
    )

    method = make_structure(
        "method-001",
        ScientificStructureType.METHOD,
        page_number=2,
    )

    graph = ScientificRelationshipBuilder().build(
        make_document(),
        make_structure_state(
            hypothesis,
            method,
        ),
    )

    assert graph.relationships == ()


def test_builder_does_not_link_unknown_page():
    hypothesis = make_structure(
        "hypothesis-001",
        ScientificStructureType.HYPOTHESIS,
        page_number=None,
    )

    method = make_structure(
        "method-001",
        ScientificStructureType.METHOD,
        page_number=1,
    )

    graph = ScientificRelationshipBuilder().build(
        make_document(),
        make_structure_state(
            hypothesis,
            method,
        ),
    )

    assert graph.relationships == ()


def test_builder_does_not_invent_unlisted_relationship():
    limitation = make_structure(
        "limitation-001",
        ScientificStructureType.LIMITATION,
    )

    dataset = make_structure(
        "dataset-001",
        ScientificStructureType.DATASET,
    )

    graph = ScientificRelationshipBuilder().build(
        make_document(),
        make_structure_state(
            limitation,
            dataset,
        ),
    )

    assert graph.relationships == ()


def test_builder_preserves_combined_provenance():
    hypothesis = make_structure(
        "hypothesis-001",
        ScientificStructureType.HYPOTHESIS,
        source_element_id="element-h",
    )

    method = make_structure(
        "method-001",
        ScientificStructureType.METHOD,
        source_element_id="element-m",
    )

    graph = ScientificRelationshipBuilder().build(
        make_document(),
        make_structure_state(
            hypothesis,
            method,
        ),
    )

    assert (
        graph.relationships[0].source_element_ids
        == (
            "element-h",
            "element-m",
        )
    )


def test_builder_is_deterministic():
    hypothesis = make_structure(
        "hypothesis-001",
        ScientificStructureType.HYPOTHESIS,
    )

    method = make_structure(
        "method-001",
        ScientificStructureType.METHOD,
    )

    structures = make_structure_state(
        hypothesis,
        method,
    )

    document = make_document()

    builder = ScientificRelationshipBuilder()

    first = builder.build(
        document,
        structures,
    )

    second = builder.build(
        document,
        structures,
    )

    assert first == second

    assert (
        first.relationships[0].relationship_id
        == second.relationships[0].relationship_id
    )


def test_builder_does_not_assign_truth_confidence():
    hypothesis = make_structure(
        "hypothesis-001",
        ScientificStructureType.HYPOTHESIS,
    )

    method = make_structure(
        "method-001",
        ScientificStructureType.METHOD,
    )

    graph = ScientificRelationshipBuilder().build(
        make_document(),
        make_structure_state(
            hypothesis,
            method,
        ),
    )

    assert graph.relationships[0].confidence is None


def test_builder_records_inference_method():
    hypothesis = make_structure(
        "hypothesis-001",
        ScientificStructureType.HYPOTHESIS,
    )

    method = make_structure(
        "method-001",
        ScientificStructureType.METHOD,
    )

    graph = ScientificRelationshipBuilder().build(
        make_document(),
        make_structure_state(
            hypothesis,
            method,
        ),
    )

    assert (
        graph.relationships[0].inference_method
        == "same_page_structure_rule_v0.1"
    )


def test_builder_empty_structures_produce_empty_graph():
    graph = ScientificRelationshipBuilder().build(
        make_document(),
        make_structure_state(),
    )

    assert graph.relationships == ()


def test_builder_preserves_document_identity():
    document = make_document(
        document_id="doc-special"
    )

    hypothesis = make_structure(
        "hypothesis-001",
        ScientificStructureType.HYPOTHESIS,
        document_id="doc-special",
    )

    method = make_structure(
        "method-001",
        ScientificStructureType.METHOD,
        document_id="doc-special",
    )

    structures = make_structure_state(
        hypothesis,
        method,
        document_id="doc-special",
    )

    graph = ScientificRelationshipBuilder().build(
        document,
        structures,
    )

    assert graph.document_id == "doc-special"

    assert all(
        relationship.document_id == "doc-special"
        for relationship in graph.relationships
    )


def test_supports_edge_is_not_promoted_to_verified_status():
    result = make_structure(
        "result-001",
        ScientificStructureType.RESULT,
    )

    hypothesis = make_structure(
        "hypothesis-001",
        ScientificStructureType.HYPOTHESIS,
    )

    graph = ScientificRelationshipBuilder().build(
        make_document(),
        make_structure_state(
            result,
            hypothesis,
        ),
    )

    relationship = graph.relationships[0]

    assert (
        relationship.status
        != ScientificRelationshipStatus.CANDIDATE
    )

    assert (
        relationship.status
        == ScientificRelationshipStatus.HEURISTIC
    )

    assert relationship.confidence is None