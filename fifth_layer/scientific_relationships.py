"""Scientific relationship graph contracts for AISTHESIS Mini-Lab.

This module represents candidate relationships between scientific
structures and document elements.

Important epistemic boundaries:

    relationship candidate != verified scientific relationship
    graph edge != scientific fact
    supports edge != demonstrated support
    contradicts edge != demonstrated contradiction
    structural proximity != semantic relationship
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from fifth_layer.scientific_document import ScientificDocumentState
from fifth_layer.scientific_structure import (
    ScientificStructureState,
    ScientificStructureType,
)
from fifth_layer.world_model.evidence import stable_id


class ScientificNodeKind(str, Enum):
    """Kinds of nodes that may participate in the graph."""

    STRUCTURE_CANDIDATE = "structure_candidate"
    DOCUMENT_ELEMENT = "document_element"


class ScientificRelationshipType(str, Enum):
    """Candidate scientific relationship types."""

    TESTED_BY = "tested_by"
    USES = "uses"
    MEASURES = "measures"
    ABOUT = "about"
    SUPPORTED_BY = "supported_by"
    DESCRIBES = "describes"
    MODELS = "models"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    UNKNOWN = "unknown"


class ScientificRelationshipStatus(str, Enum):
    """Epistemic status of a relationship candidate."""

    CANDIDATE = "candidate"
    HEURISTIC = "heuristic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ScientificGraphNodeRef:
    """Reference to an existing scientific structure or document element.

    The graph does not duplicate the source object. It preserves a
    reference to an already-existing identity.
    """

    node_id: str
    node_kind: ScientificNodeKind
    document_id: str

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("node_id must not be empty")

        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_kind": self.node_kind.value,
            "document_id": self.document_id,
        }


@dataclass(frozen=True)
class ScientificRelationshipCandidate:
    """A possible relationship between two scientific graph nodes.

    The existence of this object does not establish that the
    relationship is scientifically correct.
    """

    relationship_id: str
    relationship_type: ScientificRelationshipType
    status: ScientificRelationshipStatus

    source: ScientificGraphNodeRef
    target: ScientificGraphNodeRef

    document_id: str

    source_element_ids: Tuple[str, ...]

    confidence: Optional[float] = None
    inference_method: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.relationship_id.strip():
            raise ValueError(
                "relationship_id must not be empty"
            )

        if not self.document_id.strip():
            raise ValueError(
                "document_id must not be empty"
            )

        if self.source.document_id != self.document_id:
            raise ValueError(
                "source document_id does not match "
                "relationship document_id"
            )

        if self.target.document_id != self.document_id:
            raise ValueError(
                "target document_id does not match "
                "relationship document_id"
            )

        if (
            self.source.node_id == self.target.node_id
            and self.source.node_kind == self.target.node_kind
        ):
            raise ValueError(
                "self relationships are not allowed"
            )

        if not self.source_element_ids:
            raise ValueError(
                "relationship candidate must preserve "
                "source provenance"
            )

        if any(
            not isinstance(source_id, str)
            or not source_id.strip()
            for source_id in self.source_element_ids
        ):
            raise ValueError(
                "source_element_ids must contain "
                "non-empty strings"
            )

        if len(self.source_element_ids) != len(
            set(self.source_element_ids)
        ):
            raise ValueError(
                "source_element_ids must not contain duplicates"
            )

        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    "confidence must be between 0.0 and 1.0"
                )

    def to_dict(self) -> dict:
        return {
            "relationship_id": self.relationship_id,
            "relationship_type": (
                self.relationship_type.value
            ),
            "status": self.status.value,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "document_id": self.document_id,
            "source_element_ids": list(
                self.source_element_ids
            ),
            "confidence": self.confidence,
            "inference_method": self.inference_method,
        }


@dataclass(frozen=True)
class ScientificRelationshipGraph:
    """Immutable candidate relationship graph."""

    document_id: str
    relationships: Tuple[
        ScientificRelationshipCandidate, ...
    ]

    schema_version: str = "scientific-relationship-graph-v0.1"

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError(
                "document_id must not be empty"
            )

        if not self.schema_version.strip():
            raise ValueError(
                "schema_version must not be empty"
            )

        relationship_ids = [
            relationship.relationship_id
            for relationship in self.relationships
        ]

        if len(relationship_ids) != len(
            set(relationship_ids)
        ):
            raise ValueError(
                "duplicate relationship_id detected"
            )

        for relationship in self.relationships:
            if (
                relationship.document_id
                != self.document_id
            ):
                raise ValueError(
                    "relationship document_id does not "
                    "match graph document_id"
                )

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "schema_version": self.schema_version,
            "relationships": [
                relationship.to_dict()
                for relationship in self.relationships
            ],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class ScientificRelationshipBuilder:
    """Conservative structural relationship candidate builder.

    v0.1 deliberately uses narrow, inspectable structural rules.

    It does not perform semantic scientific verification.
    """

    _STRUCTURE_RULES = (
        (
            ScientificStructureType.HYPOTHESIS,
            ScientificStructureType.METHOD,
            ScientificRelationshipType.TESTED_BY,
            "same_page_structure_rule_v0.1",
        ),
        (
            ScientificStructureType.METHOD,
            ScientificStructureType.DATASET,
            ScientificRelationshipType.USES,
            "same_page_structure_rule_v0.1",
        ),
        (
            ScientificStructureType.METHOD,
            ScientificStructureType.VARIABLE,
            ScientificRelationshipType.MEASURES,
            "same_page_structure_rule_v0.1",
        ),
        (
            ScientificStructureType.RESULT,
            ScientificStructureType.VARIABLE,
            ScientificRelationshipType.ABOUT,
            "same_page_structure_rule_v0.1",
        ),
        (
            ScientificStructureType.RESULT,
            ScientificStructureType.HYPOTHESIS,
            ScientificRelationshipType.SUPPORTS,
            "same_page_structure_rule_v0.1",
        ),
        (
            ScientificStructureType.CLAIM,
            ScientificStructureType.RESULT,
            ScientificRelationshipType.SUPPORTED_BY,
            "same_page_structure_rule_v0.1",
        ),
    )

    def build(
        self,
        document: ScientificDocumentState,
        structures: ScientificStructureState,
    ) -> ScientificRelationshipGraph:
        """Build conservative candidate relationships.

        v0.1 requires document and structure state to refer to the
        same document.

        Relationships are created only for explicitly allowed
        structure-type pairs occurring on the same page.

        Same-page proximity is a routing heuristic, not proof of a
        semantic scientific relationship.
        """

        if not isinstance(
            document,
            ScientificDocumentState,
        ):
            raise ValueError(
                "ScientificRelationshipBuilder expects "
                "ScientificDocumentState"
            )

        if not isinstance(
            structures,
            ScientificStructureState,
        ):
            raise ValueError(
                "ScientificRelationshipBuilder expects "
                "ScientificStructureState"
            )

        if document.document_id != structures.document_id:
            raise ValueError(
                "document and structure state must refer "
                "to the same document"
            )

        relationships = []

        for (
            source_type,
            target_type,
            relationship_type,
            rule_name,
        ) in self._STRUCTURE_RULES:
            sources = [
                candidate
                for candidate in structures.candidates
                if candidate.structure_type == source_type
            ]

            targets = [
                candidate
                for candidate in structures.candidates
                if candidate.structure_type == target_type
            ]

            for source_candidate in sources:
                for target_candidate in targets:
                    if (
                        source_candidate.page_number is None
                        or target_candidate.page_number is None
                    ):
                        continue

                    if (
                        source_candidate.page_number
                        != target_candidate.page_number
                    ):
                        continue

                    source_ref = ScientificGraphNodeRef(
                        node_id=source_candidate.candidate_id,
                        node_kind=(
                            ScientificNodeKind.STRUCTURE_CANDIDATE
                        ),
                        document_id=document.document_id,
                    )

                    target_ref = ScientificGraphNodeRef(
                        node_id=target_candidate.candidate_id,
                        node_kind=(
                            ScientificNodeKind.STRUCTURE_CANDIDATE
                        ),
                        document_id=document.document_id,
                    )

                    provenance = tuple(
                        sorted(
                            set(
                                source_candidate.source_element_ids
                            )
                            | set(
                                target_candidate.source_element_ids
                            )
                        )
                    )

                    relationship_id = stable_id(
                        "scientific-relationship-candidate",
                        document.document_id,
                        relationship_type.value,
                        source_ref.node_kind.value,
                        source_ref.node_id,
                        target_ref.node_kind.value,
                        target_ref.node_id,
                    )

                    relationships.append(
                        ScientificRelationshipCandidate(
                            relationship_id=relationship_id,
                            relationship_type=relationship_type,
                            status=(
                                ScientificRelationshipStatus.HEURISTIC
                            ),
                            source=source_ref,
                            target=target_ref,
                            document_id=document.document_id,
                            source_element_ids=provenance,
                            confidence=None,
                            inference_method=rule_name,
                        )
                    )

        relationships = self._deduplicate(
            relationships
        )

        relationships.sort(
            key=lambda relationship: (
                relationship.relationship_type.value,
                relationship.source.node_id,
                relationship.target.node_id,
                relationship.relationship_id,
            )
        )

        return ScientificRelationshipGraph(
            document_id=document.document_id,
            relationships=tuple(relationships),
        )

    @staticmethod
    def _deduplicate(
        relationships: list[
            ScientificRelationshipCandidate
        ],
    ) -> list[ScientificRelationshipCandidate]:
        """Deduplicate exact graph edges deterministically."""

        unique = {}

        for relationship in relationships:
            key = (
                relationship.relationship_type,
                relationship.source.node_kind,
                relationship.source.node_id,
                relationship.target.node_kind,
                relationship.target.node_id,
            )

            if key not in unique:
                unique[key] = relationship

        return list(unique.values())