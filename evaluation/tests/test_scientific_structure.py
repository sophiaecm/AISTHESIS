import pytest

from fifth_layer.scientific_document import (
    DocumentElement,
    ScientificDocumentState,
    ScientificPage,
)

from fifth_layer.scientific_structure import (
    ScientificStructureBuilder,
    ScientificStructureCandidate,
    ScientificStructureState,
    ScientificStructureStatus,
    ScientificStructureType,
)


def make_candidate(
    candidate_id: str = "cand-001",
    document_id: str = "doc-001",
    text: str = "We hypothesize that X improves Y.",
    structure_type: ScientificStructureType = (
        ScientificStructureType.HYPOTHESIS
    ),
    status: ScientificStructureStatus = (
        ScientificStructureStatus.CANDIDATE
    ),
):
    return ScientificStructureCandidate(
        candidate_id=candidate_id,
        structure_type=structure_type,
        status=status,
        text=text,
        document_id=document_id,
        source_element_ids=("element-001",),
        page_number=1,
        confidence=0.7,
        extraction_method="test",
    )


def make_document_with_text(
    text: str,
    *,
    element_type: str = "text_block",
    element_id: str = "element-001",
    document_id: str = "document:test",
):
    element = DocumentElement(
        element_id=element_id,
        element_type=element_type,
        page_index=0,
        bbox=(0.0, 0.0, 100.0, 20.0),
        text_content=text,
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


# ================================================================
# CONTRACT TESTS
# ================================================================


def test_candidate_can_be_created():
    candidate = make_candidate()

    assert candidate.candidate_id == "cand-001"
    assert (
        candidate.structure_type
        == ScientificStructureType.HYPOTHESIS
    )
    assert (
        candidate.status
        == ScientificStructureStatus.CANDIDATE
    )
    assert candidate.document_id == "doc-001"


def test_candidate_is_immutable():
    candidate = make_candidate()

    with pytest.raises(Exception):
        candidate.text = "changed"


def test_candidate_requires_id():
    with pytest.raises(ValueError):
        make_candidate(candidate_id="")


def test_candidate_requires_document_id():
    with pytest.raises(ValueError):
        make_candidate(document_id="")


def test_candidate_requires_text():
    with pytest.raises(ValueError):
        make_candidate(text="")


def test_candidate_requires_source_reference():
    with pytest.raises(ValueError):
        ScientificStructureCandidate(
            candidate_id="cand-001",
            structure_type=ScientificStructureType.CLAIM,
            status=ScientificStructureStatus.CANDIDATE,
            text="A claim.",
            document_id="doc-001",
            source_element_ids=(),
        )


def test_candidate_rejects_blank_source_reference():
    with pytest.raises(ValueError):
        ScientificStructureCandidate(
            candidate_id="cand-001",
            structure_type=ScientificStructureType.CLAIM,
            status=ScientificStructureStatus.CANDIDATE,
            text="A claim.",
            document_id="doc-001",
            source_element_ids=("",),
        )


def test_candidate_rejects_duplicate_source_references():
    with pytest.raises(ValueError):
        ScientificStructureCandidate(
            candidate_id="cand-001",
            structure_type=ScientificStructureType.CLAIM,
            status=ScientificStructureStatus.CANDIDATE,
            text="A claim.",
            document_id="doc-001",
            source_element_ids=(
                "element-001",
                "element-001",
            ),
        )


@pytest.mark.parametrize(
    "page_number",
    [0, -1, -100],
)
def test_candidate_rejects_invalid_page_numbers(page_number):
    with pytest.raises(ValueError):
        ScientificStructureCandidate(
            candidate_id="cand-001",
            structure_type=ScientificStructureType.RESULT,
            status=ScientificStructureStatus.CANDIDATE,
            text="Result candidate.",
            document_id="doc-001",
            source_element_ids=("element-001",),
            page_number=page_number,
        )


@pytest.mark.parametrize(
    "confidence",
    [-0.01, -1.0, 1.01, 5.0],
)
def test_candidate_rejects_invalid_confidence(confidence):
    with pytest.raises(ValueError):
        ScientificStructureCandidate(
            candidate_id="cand-001",
            structure_type=ScientificStructureType.CLAIM,
            status=ScientificStructureStatus.HEURISTIC,
            text="Claim candidate.",
            document_id="doc-001",
            source_element_ids=("element-001",),
            confidence=confidence,
        )


def test_candidate_allows_unknown_confidence():
    candidate = ScientificStructureCandidate(
        candidate_id="cand-001",
        structure_type=ScientificStructureType.UNKNOWN,
        status=ScientificStructureStatus.UNKNOWN,
        text="Unknown scientific structure.",
        document_id="doc-001",
        source_element_ids=("element-001",),
        confidence=None,
    )

    assert candidate.confidence is None


def test_state_can_be_created():
    candidate = make_candidate()

    state = ScientificStructureState(
        document_id="doc-001",
        candidates=(candidate,),
    )

    assert state.document_id == "doc-001"
    assert state.candidates == (candidate,)
    assert (
        state.schema_version
        == "scientific-structure-v0.1"
    )


def test_state_is_immutable():
    state = ScientificStructureState(
        document_id="doc-001",
        candidates=(make_candidate(),),
    )

    with pytest.raises(Exception):
        state.document_id = "changed"


def test_state_rejects_duplicate_candidate_ids():
    a = make_candidate(candidate_id="same-id")
    b = make_candidate(candidate_id="same-id")

    with pytest.raises(ValueError):
        ScientificStructureState(
            document_id="doc-001",
            candidates=(a, b),
        )


def test_state_rejects_candidate_from_different_document():
    candidate = make_candidate(
        document_id="other-doc",
    )

    with pytest.raises(ValueError):
        ScientificStructureState(
            document_id="doc-001",
            candidates=(candidate,),
        )


def test_state_allows_empty_candidate_collection():
    state = ScientificStructureState(
        document_id="doc-001",
        candidates=(),
    )

    assert state.candidates == ()


@pytest.mark.parametrize(
    "structure_type",
    [
        ScientificStructureType.RESEARCH_QUESTION,
        ScientificStructureType.HYPOTHESIS,
        ScientificStructureType.METHOD,
        ScientificStructureType.VARIABLE,
        ScientificStructureType.DATASET,
        ScientificStructureType.RESULT,
        ScientificStructureType.CLAIM,
        ScientificStructureType.LIMITATION,
        ScientificStructureType.UNKNOWN,
    ],
)
def test_all_structure_types_are_representable(
    structure_type,
):
    candidate = ScientificStructureCandidate(
        candidate_id=(
            f"candidate-{structure_type.value}"
        ),
        structure_type=structure_type,
        status=ScientificStructureStatus.CANDIDATE,
        text=f"Example {structure_type.value}",
        document_id="doc-001",
        source_element_ids=("element-001",),
    )

    assert candidate.structure_type == structure_type


@pytest.mark.parametrize(
    "status",
    [
        ScientificStructureStatus.CANDIDATE,
        ScientificStructureStatus.EXPLICIT,
        ScientificStructureStatus.HEURISTIC,
        ScientificStructureStatus.UNKNOWN,
    ],
)
def test_all_statuses_are_representable(status):
    candidate = ScientificStructureCandidate(
        candidate_id=f"candidate-{status.value}",
        structure_type=ScientificStructureType.UNKNOWN,
        status=status,
        text="Example structure.",
        document_id="doc-001",
        source_element_ids=("element-001",),
    )

    assert candidate.status == status


def test_candidate_does_not_require_confidence():
    candidate = ScientificStructureCandidate(
        candidate_id="cand-no-confidence",
        structure_type=ScientificStructureType.METHOD,
        status=ScientificStructureStatus.EXPLICIT,
        text="Participants completed the task.",
        document_id="doc-001",
        source_element_ids=("element-001",),
    )

    assert candidate.confidence is None


def test_extraction_method_is_metadata_not_truth_status():
    candidate = ScientificStructureCandidate(
        candidate_id="cand-heuristic",
        structure_type=ScientificStructureType.LIMITATION,
        status=ScientificStructureStatus.HEURISTIC,
        text="A possible limitation.",
        document_id="doc-001",
        source_element_ids=("element-001",),
        extraction_method=(
            "section_heading_heuristic"
        ),
    )

    assert (
        candidate.status
        == ScientificStructureStatus.HEURISTIC
    )

    assert (
        candidate.extraction_method
        == "section_heading_heuristic"
    )


def test_candidate_preserves_multiple_source_elements():
    candidate = ScientificStructureCandidate(
        candidate_id="cand-multi-source",
        structure_type=ScientificStructureType.RESULT,
        status=ScientificStructureStatus.CANDIDATE,
        text="Combined result text.",
        document_id="doc-001",
        source_element_ids=(
            "element-001",
            "element-002",
        ),
    )

    assert candidate.source_element_ids == (
        "element-001",
        "element-002",
    )


# ================================================================
# BUILDER TESTS
# ================================================================


def test_builder_detects_hypothesis_candidate():
    document = make_document_with_text(
        "We hypothesize that sleep improves memory."
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert len(state.candidates) == 1

    candidate = state.candidates[0]

    assert (
        candidate.structure_type
        == ScientificStructureType.HYPOTHESIS
    )

    assert (
        candidate.status
        == ScientificStructureStatus.HEURISTIC
    )

    assert (
        candidate.text
        == "We hypothesize that sleep improves memory."
    )

    assert candidate.source_element_ids == (
        "element-001",
    )

    assert candidate.page_number == 1
    assert candidate.confidence is None


def test_builder_detects_research_question_candidate():
    document = make_document_with_text(
        "Our research question is whether "
        "sleep improves memory."
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert len(state.candidates) == 1

    assert (
        state.candidates[0].structure_type
        == ScientificStructureType.RESEARCH_QUESTION
    )


def test_builder_detects_method_candidate():
    document = make_document_with_text(
        "We recruited 120 participants for the study."
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert len(state.candidates) == 1

    assert (
        state.candidates[0].structure_type
        == ScientificStructureType.METHOD
    )


def test_builder_detects_variable_candidate():
    document = make_document_with_text(
        "The dependent variable was response time."
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert len(state.candidates) == 1

    assert (
        state.candidates[0].structure_type
        == ScientificStructureType.VARIABLE
    )


def test_builder_detects_dataset_candidate():
    document = make_document_with_text(
        "We evaluated the model on the "
        "NeuroExample dataset."
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert len(state.candidates) == 1

    assert (
        state.candidates[0].structure_type
        == ScientificStructureType.DATASET
    )


def test_builder_detects_result_candidate():
    document = make_document_with_text(
        "We found that accuracy increased after training."
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert len(state.candidates) == 1

    assert (
        state.candidates[0].structure_type
        == ScientificStructureType.RESULT
    )


def test_builder_detects_claim_candidate():
    document = make_document_with_text(
        "We conclude that the intervention "
        "improves performance."
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert len(state.candidates) == 1

    assert (
        state.candidates[0].structure_type
        == ScientificStructureType.CLAIM
    )


def test_builder_detects_limitation_candidate():
    document = make_document_with_text(
        "A limitation of this study is "
        "the small sample size."
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert len(state.candidates) == 1

    assert (
        state.candidates[0].structure_type
        == ScientificStructureType.LIMITATION
    )


def test_builder_does_not_create_candidate_for_unmatched_text():
    document = make_document_with_text(
        "Memory is an important topic in neuroscience."
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert state.candidates == ()


def test_builder_ignores_caption_elements():
    document = make_document_with_text(
        "Results show that performance improved.",
        element_type="caption",
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert state.candidates == ()


def test_builder_ignores_image_regions():
    document = make_document_with_text(
        "We found that performance improved.",
        element_type="image_region",
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert state.candidates == ()


def test_builder_preserves_source_element_id():
    document = make_document_with_text(
        "We hypothesize that X predicts Y.",
        element_id="source-element-42",
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert (
        state.candidates[0].source_element_ids
        == ("source-element-42",)
    )


def test_builder_is_deterministic():
    document = make_document_with_text(
        "We hypothesize that X predicts Y."
    )

    builder = ScientificStructureBuilder()

    first = builder.build(document)
    second = builder.build(document)

    assert first == second

    assert (
        first.candidates[0].candidate_id
        == second.candidates[0].candidate_id
    )


def test_builder_candidate_id_changes_with_source_element():
    first_document = make_document_with_text(
        "We hypothesize that X predicts Y.",
        element_id="element-001",
    )

    second_document = make_document_with_text(
        "We hypothesize that X predicts Y.",
        element_id="element-002",
    )

    builder = ScientificStructureBuilder()

    first = builder.build(first_document)
    second = builder.build(second_document)

    assert (
        first.candidates[0].candidate_id
        != second.candidates[0].candidate_id
    )


def test_builder_rejects_non_document_state():
    with pytest.raises(ValueError):
        ScientificStructureBuilder().build(
            "not-a-document"
        )


def test_builder_can_extract_multiple_candidates_from_one_element():
    document = make_document_with_text(
        "We hypothesize that sleep improves memory. "
        "We found that performance increased."
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert len(state.candidates) == 2

    assert {
        candidate.structure_type
        for candidate in state.candidates
    } == {
        ScientificStructureType.HYPOTHESIS,
        ScientificStructureType.RESULT,
    }


def test_builder_keeps_candidates_as_heuristic():
    document = make_document_with_text(
        "We conclude that the treatment is effective."
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert (
        state.candidates[0].status
        == ScientificStructureStatus.HEURISTIC
    )


def test_builder_does_not_assign_truth_confidence():
    document = make_document_with_text(
        "We found that performance improved."
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert state.candidates[0].confidence is None


def test_builder_extraction_method_records_rule():
    document = make_document_with_text(
        "We hypothesize that X predicts Y."
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert (
        state.candidates[0]
        .extraction_method
        .startswith("lexical_pattern_v0.1:")
    )


def test_builder_uses_human_readable_page_number():
    element = DocumentElement(
        element_id="element-page-two",
        element_type="text_block",
        page_index=1,
        bbox=(0.0, 0.0, 100.0, 20.0),
        text_content=(
            "We found that performance improved."
        ),
        source="test",
    )

    page0 = ScientificPage(
        page_index=0,
        width=100.0,
        height=100.0,
        elements=(),
    )

    page1 = ScientificPage(
        page_index=1,
        width=100.0,
        height=100.0,
        elements=(element,),
    )

    document = ScientificDocumentState(
        document_id="document:test",
        source_metadata={
            "source_reference": "synthetic.pdf",
        },
        pages=(page0, page1),
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert state.candidates[0].page_number == 2


def test_builder_does_not_promote_candidate_to_explicit():
    document = make_document_with_text(
        "Our hypothesis is that X predicts Y."
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert (
        state.candidates[0].status
        != ScientificStructureStatus.EXPLICIT
    )


def test_builder_preserves_document_identity():
    document = make_document_with_text(
        "We found that performance improved.",
        document_id="document:special",
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert state.document_id == "document:special"

    assert (
        state.candidates[0].document_id
        == "document:special"
    )


# ================================================================
# SERIALIZATION TESTS
# ================================================================


def test_candidate_serialization_is_plain_and_stable():
    candidate = make_candidate()

    result = candidate.to_dict()

    assert result["candidate_id"] == "cand-001"
    assert result["structure_type"] == "hypothesis"
    assert result["status"] == "candidate"

    assert result["source_element_ids"] == [
        "element-001"
    ]


def test_state_serialization_is_deterministic():
    state = ScientificStructureState(
        document_id="doc-001",
        candidates=(make_candidate(),),
    )

    first = state.to_json()
    second = state.to_json()

    assert first == second


def test_state_json_contains_schema_version():
    state = ScientificStructureState(
        document_id="doc-001",
        candidates=(make_candidate(),),
    )

    assert (
        '"schema_version":"scientific-structure-v0.1"'
        in state.to_json()
    )


def test_state_json_does_not_serialize_enum_repr():
    state = ScientificStructureState(
        document_id="doc-001",
        candidates=(make_candidate(),),
    )

    result = state.to_json()

    assert "ScientificStructureType." not in result
    assert "ScientificStructureStatus." not in result


# ================================================================
# DUPLICATE SUPPRESSION TESTS
# ================================================================


def test_builder_suppresses_duplicate_sentence_on_same_page():
    first = DocumentElement(
        element_id="element-a",
        element_type="text_block",
        page_index=0,
        bbox=(0.0, 0.0, 100.0, 20.0),
        text_content=(
            "We found that performance improved."
        ),
        source="test",
    )

    second = DocumentElement(
        element_id="element-b",
        element_type="text_block",
        page_index=0,
        bbox=(0.0, 30.0, 100.0, 50.0),
        text_content=(
            "We found that performance improved."
        ),
        source="test",
    )

    page = ScientificPage(
        page_index=0,
        width=100.0,
        height=100.0,
        elements=(first, second),
    )

    document = ScientificDocumentState(
        document_id="document:test",
        source_metadata={
            "source_reference": "synthetic.pdf",
        },
        pages=(page,),
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert len(state.candidates) == 1

    assert (
        state.candidates[0].source_element_ids
        == (
            "element-a",
            "element-b",
        )
    )


def test_duplicate_suppression_preserves_heuristic_status():
    first = DocumentElement(
        element_id="element-a",
        element_type="text_block",
        page_index=0,
        bbox=(0.0, 0.0, 100.0, 20.0),
        text_content=(
            "We conclude that X improves Y."
        ),
        source="test",
    )

    second = DocumentElement(
        element_id="element-b",
        element_type="text_block",
        page_index=0,
        bbox=(0.0, 30.0, 100.0, 50.0),
        text_content=(
            "We conclude that X improves Y."
        ),
        source="test",
    )

    page = ScientificPage(
        page_index=0,
        width=100.0,
        height=100.0,
        elements=(first, second),
    )

    document = ScientificDocumentState(
        document_id="document:test",
        source_metadata={
            "source_reference": "synthetic.pdf",
        },
        pages=(page,),
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert (
        state.candidates[0].status
        == ScientificStructureStatus.HEURISTIC
    )


def test_same_sentence_on_different_pages_is_not_suppressed():
    element0 = DocumentElement(
        element_id="page0-element",
        element_type="text_block",
        page_index=0,
        bbox=(0.0, 0.0, 100.0, 20.0),
        text_content=(
            "We found that performance improved."
        ),
        source="test",
    )

    element1 = DocumentElement(
        element_id="page1-element",
        element_type="text_block",
        page_index=1,
        bbox=(0.0, 0.0, 100.0, 20.0),
        text_content=(
            "We found that performance improved."
        ),
        source="test",
    )

    page0 = ScientificPage(
        page_index=0,
        width=100.0,
        height=100.0,
        elements=(element0,),
    )

    page1 = ScientificPage(
        page_index=1,
        width=100.0,
        height=100.0,
        elements=(element1,),
    )

    document = ScientificDocumentState(
        document_id="document:test",
        source_metadata={
            "source_reference": "synthetic.pdf",
        },
        pages=(page0, page1),
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert len(state.candidates) == 2


def test_duplicate_merge_is_deterministic():
    first = DocumentElement(
        element_id="element-b",
        element_type="text_block",
        page_index=0,
        bbox=(0.0, 0.0, 100.0, 20.0),
        text_content=(
            "We found that performance improved."
        ),
        source="test",
    )

    second = DocumentElement(
        element_id="element-a",
        element_type="text_block",
        page_index=0,
        bbox=(0.0, 30.0, 100.0, 50.0),
        text_content=(
            "We found that performance improved."
        ),
        source="test",
    )

    page = ScientificPage(
        page_index=0,
        width=100.0,
        height=100.0,
        elements=(first, second),
    )

    document = ScientificDocumentState(
        document_id="document:test",
        source_metadata={
            "source_reference": "synthetic.pdf",
        },
        pages=(page,),
    )

    builder = ScientificStructureBuilder()

    first_state = builder.build(document)
    second_state = builder.build(document)

    assert first_state == second_state

    assert (
        first_state.candidates[0].candidate_id
        == second_state.candidates[0].candidate_id
    )

    assert (
        first_state.candidates[0].source_element_ids
        == (
            "element-a",
            "element-b",
        )
    )


def test_duplicate_merge_does_not_assign_confidence():
    first = DocumentElement(
        element_id="element-a",
        element_type="text_block",
        page_index=0,
        bbox=(0.0, 0.0, 100.0, 20.0),
        text_content=(
            "We found that performance improved."
        ),
        source="test",
    )

    second = DocumentElement(
        element_id="element-b",
        element_type="text_block",
        page_index=0,
        bbox=(0.0, 30.0, 100.0, 50.0),
        text_content=(
            "We found that performance improved."
        ),
        source="test",
    )

    page = ScientificPage(
        page_index=0,
        width=100.0,
        height=100.0,
        elements=(first, second),
    )

    document = ScientificDocumentState(
        document_id="document:test",
        source_metadata={
            "source_reference": "synthetic.pdf",
        },
        pages=(page,),
    )

    state = ScientificStructureBuilder().build(
        document
    )

    assert state.candidates[0].confidence is None