"""Application-layer scientific input adapter for AISTHESIS Mini-Lab.

This module connects a local scientific PDF input to the already
implemented and frozen Mini-Lab pipeline:

PDF path
    -> ScientificDocumentParser
    -> ScientificDocumentState
    -> ScientificStructureBuilder
    -> ScientificStructureState
    -> ScientificRelationshipBuilder
    -> ScientificRelationshipGraph
    -> ScientificEvidenceBridge
    -> EvidenceBundle
    -> CommonEvidenceStateBuilder
    -> CommonEvidenceState
    -> ScientificInputResult

This is an application adapter.

It does not introduce a second "AISTHESIS Science" architecture and
does not alter the frozen Mini-Lab scientific core.

Critical epistemic boundaries:

    extracted text != verified scientific fact
    structure candidate != verified scientific structure
    reported result != reproduced result
    author claim != established fact
    relationship candidate != verified scientific relationship
    AISTHESIS inference != author statement
    CommonEvidenceState != truth state
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from fifth_layer.scientific_document import (
    ScientificDocumentParser,
    ScientificDocumentState,
)
from fifth_layer.scientific_structure import (
    ScientificStructureBuilder,
    ScientificStructureCandidate,
    ScientificStructureState,
    ScientificStructureType,
)
from fifth_layer.scientific_relationships import (
    ScientificRelationshipBuilder,
    ScientificRelationshipCandidate,
    ScientificRelationshipGraph,
)
from fifth_layer.scientific_evidence_bridge import (
    ScientificEvidenceBridge,
    ScientificEvidenceBridgeResult,
)
from fifth_layer.world_model.common_evidence_state import (
    CommonEvidenceState,
)
from fifth_layer.world_model.common_evidence_state_builder import (
    CommonEvidenceStateBuilder,
)


@dataclass(frozen=True)
class ScientificInputSummary:
    """UI-safe summary of one Mini-Lab scientific-document run.

    This object exposes extracted/candidate content for presentation.

    It does not convert candidates into verified scientific facts.
    """

    research_questions: Tuple[str, ...]
    hypotheses: Tuple[str, ...]
    methods: Tuple[str, ...]
    variables: Tuple[str, ...]
    datasets: Tuple[str, ...]
    results: Tuple[str, ...]
    claims: Tuple[str, ...]
    limitations: Tuple[str, ...]
    relationships: Tuple[str, ...]

    @property
    def structure_count(self) -> int:
        return (
            len(self.research_questions)
            + len(self.hypotheses)
            + len(self.methods)
            + len(self.variables)
            + len(self.datasets)
            + len(self.results)
            + len(self.claims)
            + len(self.limitations)
        )

    @property
    def relationship_count(self) -> int:
        return len(self.relationships)

    def to_dict(self) -> dict:
        return {
            "research_questions": list(self.research_questions),
            "hypotheses": list(self.hypotheses),
            "methods": list(self.methods),
            "variables": list(self.variables),
            "datasets": list(self.datasets),
            "results": list(self.results),
            "claims": list(self.claims),
            "limitations": list(self.limitations),
            "relationships": list(self.relationships),
            "structure_count": self.structure_count,
            "relationship_count": self.relationship_count,
        }


@dataclass(frozen=True)
class ScientificInputResult:
    """Complete result of one application-layer scientific PDF run."""

    source_path: str
    document: ScientificDocumentState
    structures: ScientificStructureState
    relationships: ScientificRelationshipGraph
    bridge_result: ScientificEvidenceBridgeResult
    common_evidence_state: CommonEvidenceState
    summary: ScientificInputSummary
    schema_version: str = "scientific-input-adapter-v0.1"

    def __post_init__(self) -> None:
        if not isinstance(self.source_path, str):
            raise ValueError("source_path must be a string")

        if not self.source_path.strip():
            raise ValueError("source_path must not be empty")

        if not isinstance(self.document, ScientificDocumentState):
            raise ValueError(
                "document must be ScientificDocumentState"
            )

        if not isinstance(
            self.structures,
            ScientificStructureState,
        ):
            raise ValueError(
                "structures must be ScientificStructureState"
            )

        if not isinstance(
            self.relationships,
            ScientificRelationshipGraph,
        ):
            raise ValueError(
                "relationships must be ScientificRelationshipGraph"
            )

        if not isinstance(
            self.bridge_result,
            ScientificEvidenceBridgeResult,
        ):
            raise ValueError(
                "bridge_result must be "
                "ScientificEvidenceBridgeResult"
            )

        if not isinstance(
            self.common_evidence_state,
            CommonEvidenceState,
        ):
            raise ValueError(
                "common_evidence_state must be "
                "CommonEvidenceState"
            )

        if not isinstance(self.summary, ScientificInputSummary):
            raise ValueError(
                "summary must be ScientificInputSummary"
            )

        if not isinstance(self.schema_version, str):
            raise ValueError("schema_version must be a string")

        if not self.schema_version.strip():
            raise ValueError(
                "schema_version must not be empty"
            )

        document_id = self.document.document_id

        if self.structures.document_id != document_id:
            raise ValueError(
                "structure state document mismatch"
            )

        if self.relationships.document_id != document_id:
            raise ValueError(
                "relationship graph document mismatch"
            )

        if self.bridge_result.document_id != document_id:
            raise ValueError(
                "bridge result document mismatch"
            )

        if (
            self.common_evidence_state.scene_id
            != self.bridge_result.scene_id
        ):
            raise ValueError(
                "CommonEvidenceState scene mismatch"
            )

    @property
    def document_id(self) -> str:
        return self.document.document_id

    @property
    def scene_id(self) -> str:
        return self.common_evidence_state.scene_id

    @property
    def evidence_items(self):
        return self.common_evidence_state.evidence_items


class ScientificInputAdapter:
    """Run the existing Mini-Lab pipeline for one local PDF.

    This class belongs to the application/input boundary.

    It coordinates existing frozen components but does not add new
    scientific interpretation rules.
    """

    SOURCE_COMPONENT = "scientific_input_adapter_v0.1"

    def __init__(
        self,
        *,
        parser: Optional[ScientificDocumentParser] = None,
        structure_builder: Optional[
            ScientificStructureBuilder
        ] = None,
        relationship_builder: Optional[
            ScientificRelationshipBuilder
        ] = None,
        evidence_bridge: Optional[
            ScientificEvidenceBridge
        ] = None,
        common_evidence_builder: Optional[
            CommonEvidenceStateBuilder
        ] = None,
    ) -> None:
        self._parser = (
            parser
            if parser is not None
            else ScientificDocumentParser()
        )

        self._structure_builder = (
            structure_builder
            if structure_builder is not None
            else ScientificStructureBuilder()
        )

        self._relationship_builder = (
            relationship_builder
            if relationship_builder is not None
            else ScientificRelationshipBuilder()
        )

        self._evidence_bridge = (
            evidence_bridge
            if evidence_bridge is not None
            else ScientificEvidenceBridge()
        )

        self._common_evidence_builder = (
            common_evidence_builder
            if common_evidence_builder is not None
            else CommonEvidenceStateBuilder()
        )

    def analyze_pdf(
        self,
        path,
        *,
        scene_id: Optional[str] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> ScientificInputResult:
        """Analyze one local PDF through the existing Mini-Lab.

        No scientific truth decision is performed.
        """

        resolved_path = self._validate_pdf_path(path)

        document = self._parser.parse(resolved_path)

        structures = self._structure_builder.build(
            document
        )

        relationships = self._relationship_builder.build(
            document,
            structures,
        )

        resolved_scene_id = (
            scene_id
            if scene_id is not None
            else f"document:{document.document_id}"
        )

        if not isinstance(resolved_scene_id, str):
            raise ValueError("scene_id must be a string")

        if not resolved_scene_id.strip():
            raise ValueError(
                "scene_id must not be empty"
            )

        resolved_session_id = (
            session_id
            if session_id is not None
            else f"scientific:{document.document_id}"
        )

        if not isinstance(resolved_session_id, str):
            raise ValueError(
                "session_id must be a string"
            )

        if not resolved_session_id.strip():
            raise ValueError(
                "session_id must not be empty"
            )

        bridge_result = self._evidence_bridge.build(
            document,
            structures,
            relationships,
            scene_id=resolved_scene_id,
            session_id=resolved_session_id,
            timestamp=timestamp,
        )

        common_evidence_state = (
            self._common_evidence_builder.build(
                scene_id=resolved_scene_id,
                session_id=resolved_session_id,
                timestamp=timestamp,
                coordinate_frame_id=None,
                evidence=bridge_result.evidence_bundle,
                present_families=("documentary",),
            )
        )

        summary = self._build_summary(
            structures,
            relationships,
        )

        return ScientificInputResult(
            source_path=str(resolved_path),
            document=document,
            structures=structures,
            relationships=relationships,
            bridge_result=bridge_result,
            common_evidence_state=common_evidence_state,
            summary=summary,
        )

    @staticmethod
    def _validate_pdf_path(path) -> Path:
        if isinstance(path, Path):
            resolved = path
        elif isinstance(path, str):
            if not path.strip():
                raise ValueError(
                    "PDF path must not be empty"
                )
            resolved = Path(path)
        else:
            raise ValueError(
                "PDF path must be a string or Path"
            )

        if resolved.suffix.lower() != ".pdf":
            raise ValueError(
                "ScientificInputAdapter currently supports "
                "PDF files only"
            )

        return resolved

    @staticmethod
    def _build_summary(
        structures: ScientificStructureState,
        relationships: ScientificRelationshipGraph,
    ) -> ScientificInputSummary:
        return ScientificInputSummary(
            research_questions=(
                ScientificInputAdapter._texts_for_type(
                    structures,
                    ScientificStructureType.RESEARCH_QUESTION,
                )
            ),
            hypotheses=(
                ScientificInputAdapter._texts_for_type(
                    structures,
                    ScientificStructureType.HYPOTHESIS,
                )
            ),
            methods=(
                ScientificInputAdapter._texts_for_type(
                    structures,
                    ScientificStructureType.METHOD,
                )
            ),
            variables=(
                ScientificInputAdapter._texts_for_type(
                    structures,
                    ScientificStructureType.VARIABLE,
                )
            ),
            datasets=(
                ScientificInputAdapter._texts_for_type(
                    structures,
                    ScientificStructureType.DATASET,
                )
            ),
            results=(
                ScientificInputAdapter._texts_for_type(
                    structures,
                    ScientificStructureType.RESULT,
                )
            ),
            claims=(
                ScientificInputAdapter._texts_for_type(
                    structures,
                    ScientificStructureType.CLAIM,
                )
            ),
            limitations=(
                ScientificInputAdapter._texts_for_type(
                    structures,
                    ScientificStructureType.LIMITATION,
                )
            ),
            relationships=(
                ScientificInputAdapter._relationship_summaries(
                    relationships
                )
            ),
        )

    @staticmethod
    def _texts_for_type(
        structures: ScientificStructureState,
        structure_type: ScientificStructureType,
    ) -> Tuple[str, ...]:
        candidates = [
            candidate
            for candidate in structures.candidates
            if candidate.structure_type == structure_type
        ]

        candidates.sort(
            key=ScientificInputAdapter._candidate_sort_key
        )

        return tuple(
            candidate.text
            for candidate in candidates
        )

    @staticmethod
    def _candidate_sort_key(
        candidate: ScientificStructureCandidate,
    ):
        page_number = (
            candidate.page_number
            if candidate.page_number is not None
            else 10**12
        )

        return (
            page_number,
            candidate.candidate_id,
        )

    @staticmethod
    def _relationship_summaries(
        relationships: ScientificRelationshipGraph,
    ) -> Tuple[str, ...]:
        ordered = sorted(
            relationships.relationships,
            key=ScientificInputAdapter._relationship_sort_key,
        )

        return tuple(
            ScientificInputAdapter._format_relationship(
                relationship
            )
            for relationship in ordered
        )

    @staticmethod
    def _relationship_sort_key(
        relationship: ScientificRelationshipCandidate,
    ):
        return (
            relationship.relationship_type.value,
            relationship.source.node_id,
            relationship.target.node_id,
            relationship.relationship_id,
        )

    @staticmethod
    def _format_relationship(
        relationship: ScientificRelationshipCandidate,
    ) -> str:
        return (
            f"{relationship.source.node_id} "
            f"--{relationship.relationship_type.value}--> "
            f"{relationship.target.node_id}"
        )