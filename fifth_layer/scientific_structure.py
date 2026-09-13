"""Scientific structure contracts and conservative extraction for AISTHESIS Mini-Lab.

This module represents candidate scientific structures extracted from a
ScientificDocumentState.

Important epistemic boundaries:

    candidate structure != verified scientific fact
    extracted hypothesis != supported hypothesis
    extracted result != reproduced result
    extracted claim != true claim
    extraction confidence != truth probability
"""

from __future__ import annotations

import json
import re

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

from fifth_layer.scientific_document import ScientificDocumentState
from fifth_layer.world_model.evidence import stable_id


class ScientificStructureType(str, Enum):
    """Kinds of scientific structures that may be represented."""

    RESEARCH_QUESTION = "research_question"
    HYPOTHESIS = "hypothesis"
    METHOD = "method"
    VARIABLE = "variable"
    DATASET = "dataset"
    RESULT = "result"
    CLAIM = "claim"
    LIMITATION = "limitation"
    UNKNOWN = "unknown"


class ScientificStructureStatus(str, Enum):
    """Epistemic status of a scientific-structure candidate."""

    CANDIDATE = "candidate"
    EXPLICIT = "explicit"
    HEURISTIC = "heuristic"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ScientificStructureCandidate:
    """A possible scientific structure found in a document.

    This object preserves what was extracted and where it came from.
    It does not assert that the candidate is scientifically correct.
    """

    candidate_id: str
    structure_type: ScientificStructureType
    status: ScientificStructureStatus
    text: str

    document_id: str
    source_element_ids: Tuple[str, ...]

    page_number: Optional[int] = None
    confidence: Optional[float] = None
    extraction_method: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty")

        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")

        if not self.text.strip():
            raise ValueError("text must not be empty")

        if not self.source_element_ids:
            raise ValueError(
                "A scientific structure candidate must preserve "
                "at least one source element reference"
            )

        if any(
            not isinstance(source_id, str) or not source_id.strip()
            for source_id in self.source_element_ids
        ):
            raise ValueError(
                "source_element_ids must contain non-empty strings"
            )

        if len(self.source_element_ids) != len(
            set(self.source_element_ids)
        ):
            raise ValueError(
                "source_element_ids must not contain duplicates"
            )

        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be >= 1")

        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    "confidence must be between 0.0 and 1.0"
                )

    def to_dict(self) -> dict:
        """Return a plain serializable representation."""

        return {
            "candidate_id": self.candidate_id,
            "structure_type": self.structure_type.value,
            "status": self.status.value,
            "text": self.text,
            "document_id": self.document_id,
            "source_element_ids": list(self.source_element_ids),
            "page_number": self.page_number,
            "confidence": self.confidence,
            "extraction_method": self.extraction_method,
        }


@dataclass(frozen=True)
class ScientificStructureState:
    """Immutable collection of scientific-structure candidates."""

    document_id: str
    candidates: Tuple[ScientificStructureCandidate, ...]

    schema_version: str = "scientific-structure-v0.1"

    def __post_init__(self) -> None:
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")

        if not self.schema_version.strip():
            raise ValueError("schema_version must not be empty")

        candidate_ids = [
            candidate.candidate_id
            for candidate in self.candidates
        ]

        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("duplicate candidate_id detected")

        for candidate in self.candidates:
            if candidate.document_id != self.document_id:
                raise ValueError(
                    "candidate document_id does not match "
                    "state document_id"
                )

    def to_dict(self) -> dict:
        """Return a deterministic plain representation."""

        return {
            "document_id": self.document_id,
            "schema_version": self.schema_version,
            "candidates": [
                candidate.to_dict()
                for candidate in self.candidates
            ],
        }

    def to_json(self) -> str:
        """Serialize deterministically without enum representations."""

        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


class ScientificStructureBuilder:
    """Conservative lexical scientific-structure candidate extractor.

    The builder intentionally performs bounded lexical extraction.

    It does not:
    - verify scientific truth
    - determine whether a hypothesis is supported
    - reproduce experimental results
    - evaluate claims
    - construct scientific relationships
    - interpret figures, tables, or equations
    """

    _RULES = (
        (
            ScientificStructureType.RESEARCH_QUESTION,
            "research_question_explicit_v0.1",
            (
                r"\bresearch question\b",
                r"\bwe (?:ask|investigate) whether\b",
                r"\bthis study (?:asks|investigates) whether\b",
            ),
        ),
        (
            ScientificStructureType.HYPOTHESIS,
            "hypothesis_explicit_v0.1",
            (
                r"\bwe hypothesi[sz]e\b",
                r"\bour hypothesis\b",
                r"\bwe predict(?:ed)? that\b",
            ),
        ),
        (
            ScientificStructureType.VARIABLE,
            "variable_explicit_v0.1",
            (
                r"\bindependent variable\b",
                r"\bdependent variable\b",
                r"\boutcome variable\b",
                r"\bpredictor variable\b",
                r"\bcovariate\b",
                r"\boutcome measure\b",
            ),
        ),
        (
            ScientificStructureType.DATASET,
            "dataset_explicit_v0.1",
            (
                r"\bdataset\b",
                r"\bdata set\b",
                r"\bdatabase\b",
                r"\bcorpus\b",
            ),
        ),
        (
            ScientificStructureType.METHOD,
            "method_explicit_v0.1",
            (
                r"\bwe used\b",
                r"\bwe recruited\b",
                r"\bparticipants\b",
                r"\bwe measured\b",
                r"\bwe analy[sz]ed\b",
                r"\bwe conducted\b",
            ),
        ),
        (
            ScientificStructureType.RESULT,
            "result_explicit_v0.1",
            (
                r"\bresults? (?:show|showed|indicate|indicated)\b",
                r"\bwe found\b",
                r"\bwe observed\b",
                r"\bwas significantly\b",
                r"\bwere significantly\b",
            ),
        ),
        (
            ScientificStructureType.CLAIM,
            "claim_explicit_v0.1",
            (
                r"\bwe conclude\b",
                r"\bour findings suggest\b",
                r"\bthese findings suggest\b",
                r"\bdemonstrate(?:s|d)? that\b",
            ),
        ),
        (
            ScientificStructureType.LIMITATION,
            "limitation_explicit_v0.1",
            (
                r"\blimitations?\b",
                r"\ba limitation\b",
                r"\bshould be interpreted with caution\b",
            ),
        ),
    )

    _ALLOWED_ELEMENT_TYPES = frozenset(
        {
            "text_block",
            "title",
            "section_heading",
        }
    )

    def build(
        self,
        document: ScientificDocumentState,
    ) -> ScientificStructureState:
        """Build conservative candidates from a scientific document."""

        if not isinstance(document, ScientificDocumentState):
            raise ValueError(
                "ScientificStructureBuilder expects "
                "ScientificDocumentState"
            )

        candidates: list[ScientificStructureCandidate] = []

        for element in document.elements:
            if element.element_type not in self._ALLOWED_ELEMENT_TYPES:
                continue

            if not element.text_content:
                continue

            sentences = self._sentences(element.text_content)

            for sentence_index, sentence in enumerate(sentences):
                match = self._classify(sentence)

                if match is None:
                    continue

                structure_type, rule_name = match

                candidate_id = stable_id(
                    "scientific-structure-candidate",
                    document.document_id,
                    element.element_id,
                    sentence_index,
                    structure_type.value,
                    sentence,
                )

                candidates.append(
                    ScientificStructureCandidate(
                        candidate_id=candidate_id,
                        structure_type=structure_type,
                        status=ScientificStructureStatus.HEURISTIC,
                        text=sentence,
                        document_id=document.document_id,
                        source_element_ids=(element.element_id,),
                        page_number=element.page_index + 1,
                        confidence=None,
                        extraction_method=(
                            f"lexical_pattern_v0.1:{rule_name}"
                        ),
                    )
                )

        candidates = self._suppress_duplicates(candidates)

        candidates.sort(
            key=lambda candidate: (
                candidate.page_number
                if candidate.page_number is not None
                else 0,
                candidate.structure_type.value,
                candidate.text.casefold(),
                candidate.candidate_id,
            )
        )

        return ScientificStructureState(
            document_id=document.document_id,
            candidates=tuple(candidates),
        )

    @staticmethod
    def _sentences(text: str) -> tuple[str, ...]:
        """Split text conservatively into candidate sentences."""

        parts = re.split(
            r"(?<=[.!?])\s+|\n+",
            text.strip(),
        )

        return tuple(
            part.strip()
            for part in parts
            if part.strip()
        )

    @classmethod
    def _classify(
        cls,
        sentence: str,
    ) -> tuple[ScientificStructureType, str] | None:
        """Return the first matching lexical rule."""

        for structure_type, rule_name, patterns in cls._RULES:
            for pattern in patterns:
                if re.search(
                    pattern,
                    sentence,
                    flags=re.IGNORECASE,
                ):
                    return structure_type, rule_name

        return None

    @staticmethod
    def _suppress_duplicates(
        candidates: list[ScientificStructureCandidate],
    ) -> list[ScientificStructureCandidate]:
        """Suppress exact same-page duplicate candidates.

        Duplicate identity is intentionally narrow:

        - same scientific structure type
        - same normalized text
        - same human-readable page

        When duplicates originate from multiple document elements,
        provenance is merged instead of discarded.

        The operation does not infer that repeated statements on
        different pages refer to the same scientific event.
        """

        grouped: dict[
            tuple[ScientificStructureType, str, Optional[int]],
            ScientificStructureCandidate,
        ] = {}

        for candidate in candidates:
            normalized_text = " ".join(
                candidate.text.split()
            ).casefold()

            key = (
                candidate.structure_type,
                normalized_text,
                candidate.page_number,
            )

            if key not in grouped:
                grouped[key] = candidate
                continue

            existing = grouped[key]

            merged_sources = tuple(
                sorted(
                    set(existing.source_element_ids)
                    | set(candidate.source_element_ids)
                )
            )

            merged_id = stable_id(
                "scientific-structure-candidate",
                candidate.document_id,
                candidate.structure_type.value,
                candidate.page_number,
                normalized_text,
                merged_sources,
            )

            grouped[key] = ScientificStructureCandidate(
                candidate_id=merged_id,
                structure_type=existing.structure_type,
                status=existing.status,
                text=existing.text,
                document_id=existing.document_id,
                source_element_ids=merged_sources,
                page_number=existing.page_number,
                confidence=None,
                extraction_method=existing.extraction_method,
            )

        return list(grouped.values())