"""Scientific evidence bridge for AISTHESIS Mini-Lab.

This module converts scientific-document analysis artifacts into
documentary EvidenceItem objects that can enter the existing
CommonEvidenceState architecture.

Critical epistemic boundaries:

    content reported in a paper != observed physical fact
    reported result != reproduced result
    author claim != established fact
    relationship candidate != verified scientific relationship
    AISTHESIS inference != author statement
    evidence aggregation != belief formation

The bridge transports provenance and epistemic status.
It does not decide truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from fifth_layer.scientific_document import ScientificDocumentState
from fifth_layer.scientific_relationships import (
    ScientificRelationshipGraph,
    ScientificRelationshipStatus,
)
from fifth_layer.scientific_structure import (
    ScientificStructureState,
    ScientificStructureStatus,
    ScientificStructureType,
)
from fifth_layer.world_model.evidence import (
    EvidenceBundle,
    EvidenceItem,
    EvidenceSource,
    stable_id,
)


@dataclass(frozen=True)
class ScientificEvidenceBridgeResult:
    """Immutable result of scientific-document evidence conversion."""

    document_id: str
    scene_id: str
    evidence_bundle: EvidenceBundle
    schema_version: str = "scientific-evidence-bridge-v0.1"

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")

        if not self.scene_id.strip():
            raise ValueError("scene_id must not be empty")

        if not isinstance(self.evidence_bundle, EvidenceBundle):
            raise ValueError(
                "evidence_bundle must be an EvidenceBundle"
            )

        if self.evidence_bundle.scene_id != self.scene_id:
            raise ValueError(
                "evidence bundle scene_id must match result scene_id"
            )

        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty")

    @property
    def evidence_items(self) -> Tuple[EvidenceItem, ...]:
        return self.evidence_bundle.items


class ScientificEvidenceBridge:
    """Convert Mini-Lab artifacts into documentary evidence.

    v0.1 is deliberately conservative.

    The bridge does not:
    - verify claims,
    - verify results,
    - calculate scientific truth confidence,
    - promote heuristic relationships,
    - create belief,
    - alter CommonEvidenceState,
    - alter Reasoner Orchestration,
    - alter Dynamic Reasoning Connectome.
    """

    SOURCE_COMPONENT = "scientific_evidence_bridge_v0.1"

    def build(
        self,
        document: ScientificDocumentState,
        structures: ScientificStructureState,
        relationships: ScientificRelationshipGraph,
        *,
        scene_id: Optional[str] = None,
        session_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> ScientificEvidenceBridgeResult:
        """Create documentary evidence from Mini-Lab states."""

        if not isinstance(document, ScientificDocumentState):
            raise ValueError(
                "ScientificEvidenceBridge expects "
                "ScientificDocumentState"
            )

        if not isinstance(structures, ScientificStructureState):
            raise ValueError(
                "ScientificEvidenceBridge expects "
                "ScientificStructureState"
            )

        if not isinstance(
            relationships,
            ScientificRelationshipGraph,
        ):
            raise ValueError(
                "ScientificEvidenceBridge expects "
                "ScientificRelationshipGraph"
            )

        if document.document_id != structures.document_id:
            raise ValueError(
                "document and structure state must refer "
                "to the same document"
            )

        if document.document_id != relationships.document_id:
            raise ValueError(
                "document and relationship graph must refer "
                "to the same document"
            )

        resolved_scene_id = (
            scene_id
            if scene_id is not None
            else f"document:{document.document_id}"
        )

        if not isinstance(resolved_scene_id, str):
            raise ValueError("scene_id must be a string")

        if not resolved_scene_id.strip():
            raise ValueError("scene_id must not be empty")

        if session_id is not None:
            if not isinstance(session_id, str):
                raise ValueError("session_id must be a string")

            if not session_id.strip():
                raise ValueError(
                    "session_id must not be empty"
                )

        items = []

        for candidate in structures.candidates:
            item = self._structure_to_evidence(
                document_id=document.document_id,
                scene_id=resolved_scene_id,
                session_id=session_id,
                timestamp=timestamp,
                candidate=candidate,
            )

            if item is not None:
                items.append(item)

        for relationship in relationships.relationships:
            items.append(
                self._relationship_to_evidence(
                    document_id=document.document_id,
                    scene_id=resolved_scene_id,
                    session_id=session_id,
                    timestamp=timestamp,
                    relationship=relationship,
                )
            )

        bundle = EvidenceBundle(
            scene_id=resolved_scene_id,
            items=tuple(items),
            provenance={
                "document_id": document.document_id,
                "source_component": self.SOURCE_COMPONENT,
                "document_schema_version": getattr(
                    document,
                    "schema_version",
                    None,
                ),
                "structure_schema_version": structures.schema_version,
                "relationship_schema_version": (
                    relationships.schema_version
                ),
                "bridge_schema_version": (
                    "scientific-evidence-bridge-v0.1"
                ),
                "session_id": session_id,
                "timestamp": timestamp,
                "truth_decision": "not_performed",
            },
        )

        return ScientificEvidenceBridgeResult(
            document_id=document.document_id,
            scene_id=resolved_scene_id,
            evidence_bundle=bundle,
        )

    def _structure_to_evidence(
        self,
        *,
        document_id: str,
        scene_id: str,
        session_id: Optional[str],
        timestamp: Optional[float],
        candidate,
    ) -> Optional[EvidenceItem]:
        """Convert one structure candidate into documentary evidence."""

        epistemic_status = self._structure_epistemic_status(
            candidate.structure_type
        )

        if epistemic_status is None:
            return None

        evidence_type = (
            f"scientific_structure:"
            f"{candidate.structure_type.value}"
        )

        evidence_id = stable_id(
            "scientific-documentary-evidence",
            document_id,
            "structure",
            candidate.candidate_id,
            candidate.structure_type.value,
        )

        value = {
            "document_id": document_id,
            "candidate_id": candidate.candidate_id,
            "structure_type": candidate.structure_type.value,
            "structure_status": candidate.status.value,
            "text": candidate.text,
            "page_number": candidate.page_number,
            "source_element_ids": list(
                candidate.source_element_ids
            ),
            "extraction_method": candidate.extraction_method,
            "candidate_confidence": candidate.confidence,
            "verification_status": "not_verified",
        }

        provenance = {
            "document_id": document_id,
            "candidate_id": candidate.candidate_id,
            "source_element_ids": list(
                candidate.source_element_ids
            ),
            "page_number": candidate.page_number,
            "structure_status": candidate.status.value,
            "extraction_method": candidate.extraction_method,
            "session_id": session_id,
            "source_epistemic_layer": "document_structure",
            "truth_decision": "not_performed",
        }

        return EvidenceItem(
            evidence_id=evidence_id,
            scene_id=scene_id,
            source_type=EvidenceSource.DOCUMENTARY,
            source_component=self.SOURCE_COMPONENT,
            evidence_type=evidence_type,
            value=value,
            timestamp=timestamp,
            confidence=None,
            supports=(),
            contradicts=(),
            provenance=provenance,
            modality=None,
            epistemic_status=epistemic_status,
            supporting_evidence_ids=(),
            opposing_evidence_ids=(),
        )

    def _relationship_to_evidence(
        self,
        *,
        document_id: str,
        scene_id: str,
        session_id: Optional[str],
        timestamp: Optional[float],
        relationship,
    ) -> EvidenceItem:
        """Convert one graph edge into AISTHESIS-inferred evidence.

        A graph relationship is generated by AISTHESIS rather than
        directly asserted as verified documentary fact.

        Even SUPPORTS and CONTRADICTS graph vocabulary therefore
        remain aisthesis_inference at the evidence boundary.
        """

        evidence_id = stable_id(
            "scientific-documentary-evidence",
            document_id,
            "relationship",
            relationship.relationship_id,
            relationship.relationship_type.value,
            relationship.source.node_id,
            relationship.target.node_id,
        )

        value = {
            "document_id": document_id,
            "relationship_id": relationship.relationship_id,
            "relationship_type": (
                relationship.relationship_type.value
            ),
            "relationship_status": relationship.status.value,
            "source_node": relationship.source.to_dict(),
            "target_node": relationship.target.to_dict(),
            "source_element_ids": list(
                relationship.source_element_ids
            ),
            "inference_method": relationship.inference_method,
            "candidate_confidence": relationship.confidence,
            "verification_status": "not_verified",
        }

        provenance = {
            "document_id": document_id,
            "relationship_id": relationship.relationship_id,
            "source_element_ids": list(
                relationship.source_element_ids
            ),
            "relationship_status": relationship.status.value,
            "inference_method": relationship.inference_method,
            "session_id": session_id,
            "source_epistemic_layer": (
                "aisthesis_relationship_inference"
            ),
            "truth_decision": "not_performed",
        }

        return EvidenceItem(
            evidence_id=evidence_id,
            scene_id=scene_id,
            source_type=EvidenceSource.DOCUMENTARY,
            source_component=self.SOURCE_COMPONENT,
            evidence_type=(
                "scientific_relationship:"
                f"{relationship.relationship_type.value}"
            ),
            value=value,
            timestamp=timestamp,
            confidence=None,
            supports=(),
            contradicts=(),
            provenance=provenance,
            modality=None,
            epistemic_status="aisthesis_inference",
            supporting_evidence_ids=(),
            opposing_evidence_ids=(),
        )

    @staticmethod
    def _structure_epistemic_status(
        structure_type: ScientificStructureType,
    ) -> Optional[str]:
        """Map structure type to existing documentary status.

        The mapping describes what kind of content was extracted.

        It does not verify the content.
        """

        mapping = {
            ScientificStructureType.RESEARCH_QUESTION: "unknown",
            ScientificStructureType.HYPOTHESIS: "unknown",
            ScientificStructureType.METHOD: "method",
            ScientificStructureType.VARIABLE: "measurement",
            ScientificStructureType.DATASET: "dataset_reference",
            ScientificStructureType.RESULT: "reported_result",
            ScientificStructureType.CLAIM: "author_claim",
            ScientificStructureType.LIMITATION: "limitation",
            ScientificStructureType.UNKNOWN: "unknown",
        }

        return mapping.get(structure_type)