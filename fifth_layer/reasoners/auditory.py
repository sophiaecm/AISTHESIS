"""Auditory sensory-consequence reasoning for AISTHESIS.

This reasoner does not claim that sound was actually heard.

It predicts plausible auditory consequences from visual,
semantic, motion, and interaction evidence.

Observed evidence
    ->
Expected auditory consequence
    ->
Probability + uncertainty
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.latent_state import LatentState
from fifth_layer.future_state import FutureState
from fifth_layer.reasoners.base import BaseReasoner


class AuditoryReasoner(BaseReasoner):
    """Infer expected auditory consequences from non-auditory evidence."""

    HUMAN_CLASSES: Set[str] = {
        "person",
        "pedestrian",
        "man",
        "woman",
        "child",
        "boy",
        "girl",
        "human",
    }

    VEHICLE_CLASSES: Set[str] = {
        "car",
        "truck",
        "bus",
        "motorcycle",
        "motorbike",
        "train",
    }

    ANIMAL_CLASSES: Set[str] = {
        "dog",
        "cat",
        "horse",
        "cow",
        "sheep",
        "bird",
    }

    # ---------------------------------------------------------
    # Expected consequences
    # ---------------------------------------------------------

    def infer_expected_consequences(
        self,
        world_state: WorldState,
    ) -> ExpectedConsequences:

        data = world_state.data

        detections = data.get(
            "detections",
            [],
        ) or []

        motion_evidence = data.get(
            "motion_evidence",
            [],
        ) or []

        scene_description = str(
            data.get(
                "scene_description",
                "",
            )
        ).lower()

        candidates: List[Dict[str, Any]] = []

        # -----------------------------------------------------
        # Human movement
        # -----------------------------------------------------

        human_present = any(
            self._class_name(detection)
            in self.HUMAN_CLASSES
            for detection in detections
        )

        moving_human = self._find_moving_class(
            motion_evidence,
            self.HUMAN_CLASSES,
        )

        if moving_human is not None:

            motion_state = str(
                moving_human.get(
                    "motion_state",
                    "",
                )
            ).lower()

            speed = self._safe_float(
                moving_human.get(
                    "speed_pixels_per_second",
                    0.0,
                )
            )

            footsteps_probability = 0.62

            if speed >= 80.0:
                footsteps_probability += 0.08

            if speed >= 180.0:
                footsteps_probability += 0.05

            candidates.append(
                self._candidate(
                    consequence="footsteps_possible",
                    probability=footsteps_probability,
                    evidence=[
                        "visible_human",
                        f"motion:{motion_state}",
                    ],
                    source_object=self._motion_class_name(
                        moving_human
                    ),
                )
            )

            candidates.append(
                self._candidate(
                    consequence=(
                        "clothing_rustle_possible"
                    ),
                    probability=0.46,
                    evidence=[
                        "visible_human",
                        "human_motion",
                    ],
                    source_object=self._motion_class_name(
                        moving_human
                    ),
                )
            )

        # -----------------------------------------------------
        # Speech-like sound
        # -----------------------------------------------------
        #
        # Visible person alone is NOT enough.
        # We require an additional semantic cue.

        speech_terms = (
            "speaking",
            "talking",
            "conversation",
            "conversing",
            "mouth open",
            "speaks",
        )

        if (
            human_present
            and self._contains_any(
                scene_description,
                speech_terms,
            )
        ):
            candidates.append(
                self._candidate(
                    consequence=(
                        "speech_like_sound_possible"
                    ),
                    probability=0.66,
                    evidence=[
                        "visible_human",
                        "semantic_speech_cue",
                    ],
                    source_object="person",
                )
            )

        # -----------------------------------------------------
        # Contact / impact
        # -----------------------------------------------------

        contact_terms = (
            "hits",
            "hitting",
            "strikes",
            "striking",
            "collides",
            "collision",
            "falls",
            "falling",
            "drops",
            "dropping",
            "lands",
            "impact",
        )

        if self._contains_any(
            scene_description,
            contact_terms,
        ):
            candidates.append(
                self._candidate(
                    consequence=(
                        "contact_or_impact_sound_possible"
                    ),
                    probability=0.64,
                    evidence=[
                        "semantic_contact_or_impact_cue",
                    ],
                    source_object=None,
                )
            )

        # -----------------------------------------------------
        # Vehicle movement
        # -----------------------------------------------------

        moving_vehicle = self._find_moving_class(
            motion_evidence,
            self.VEHICLE_CLASSES,
        )

        if moving_vehicle is not None:

            vehicle_name = self._motion_class_name(
                moving_vehicle
            )

            candidates.append(
                self._candidate(
                    consequence=(
                        "vehicle_motion_sound_possible"
                    ),
                    probability=0.70,
                    evidence=[
                        f"visible_vehicle:{vehicle_name}",
                        "vehicle_motion",
                    ],
                    source_object=vehicle_name,
                )
            )

        # -----------------------------------------------------
        # Animal vocalization
        # -----------------------------------------------------

        visible_animal = next(
            (
                self._class_name(detection)
                for detection in detections
                if self._class_name(detection)
                in self.ANIMAL_CLASSES
            ),
            None,
        )

        animal_sound_terms = (
            "barking",
            "barks",
            "meowing",
            "meows",
            "chirping",
            "chirps",
            "vocalizing",
        )

        if (
            visible_animal is not None
            and self._contains_any(
                scene_description,
                animal_sound_terms,
            )
        ):
            candidates.append(
                self._candidate(
                    consequence=(
                        "animal_vocalization_possible"
                    ),
                    probability=0.68,
                    evidence=[
                        f"visible_animal:{visible_animal}",
                        "semantic_vocalization_cue",
                    ],
                    source_object=visible_animal,
                )
            )

        # -----------------------------------------------------
        # Final expected-consequence representation
        # -----------------------------------------------------

        candidates = self._deduplicate_candidates(
            candidates
        )

        if candidates:

            strongest = max(
                candidates,
                key=lambda item: item[
                    "probability"
                ],
            )

            strongest_probability = float(
                strongest["probability"]
            )

            uncertainty = round(
                1.0 - strongest_probability,
                3,
            )

            auditory_state = (
                "expected_auditory_"
                "consequences_present"
            )

        else:

            strongest = None
            uncertainty = 1.0

            auditory_state = (
                "insufficient_auditory_evidence"
            )

        predictions = {
            "auditory_state": auditory_state,
            "auditory_consequences": candidates,
            "strongest_auditory_consequence":
                strongest,
            "auditory_uncertainty":
                uncertainty,
            "auditory_modality_observed":
                False,
            "reasoning_mode":
                "cross_modal_expected_consequence",
        }

        return ExpectedConsequences(
            predictions=predictions
        )

    # ---------------------------------------------------------
    # Latent state
    # ---------------------------------------------------------

    def infer_latent_state(
        self,
        world_state: WorldState,
        expected_consequences: ExpectedConsequences,
    ) -> LatentState:

        predictions = (
            expected_consequences.predictions
        )

        candidates = predictions.get(
            "auditory_consequences",
            [],
        )

        features: Dict[str, Any] = {
            "auditory_modality_observed": False,
            "reasoning_mode":
                "cross_modal_expected_consequence",
        }

        if not candidates:

            features[
                "auditory_latent_state"
            ] = "insufficient_evidence"

            features[
                "auditory_event_possible"
            ] = False

            features[
                "auditory_uncertainty"
            ] = 1.0

            return LatentState(
                features=features
            )

        strongest = max(
            candidates,
            key=lambda item: item[
                "probability"
            ],
        )

        probability = self._safe_float(
            strongest.get(
                "probability",
                0.0,
            )
        )

        features[
            "auditory_latent_state"
        ] = "unobserved_sound_event_possible"

        features[
            "auditory_event_possible"
        ] = True

        features[
            "most_likely_auditory_consequence"
        ] = strongest.get(
            "consequence"
        )

        features[
            "auditory_probability"
        ] = probability

        features[
            "auditory_uncertainty"
        ] = round(
            1.0 - probability,
            3,
        )

        features[
            "auditory_evidence"
        ] = strongest.get(
            "evidence",
            [],
        )

        features[
            "auditory_source_object"
        ] = strongest.get(
            "source_object"
        )

        return LatentState(
            features=features
        )

    # ---------------------------------------------------------
    # Future state
    # ---------------------------------------------------------

    def infer_future_state(
        self,
        latent_state: LatentState,
    ) -> FutureState:

        features = latent_state.features

        future_data = dict(features)

        consequence = features.get(
            "most_likely_auditory_consequence"
        )

        probability = self._safe_float(
            features.get(
                "auditory_probability",
                0.0,
            )
        )

        if not consequence:

            future_data[
                "auditory_future_event"
            ] = "indeterminate"

            future_data[
                "auditory_event_probability"
            ] = 0.0

        else:

            future_data[
                "auditory_future_event"
            ] = consequence

            future_data[
                "auditory_event_probability"
            ] = probability

            future_data[
                "prediction_type"
            ] = (
                "expected_sensory_consequence"
            )

        future_data[
            "auditory_modality_observed"
        ] = False

        return FutureState(
            horizon=1.0,
            data=future_data,
        )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    @staticmethod
    def _safe_float(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:
            return float(value)

        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _contains_any(
        text: str,
        terms: Iterable[str],
    ) -> bool:

        return any(
            term in text
            for term in terms
        )

    @staticmethod
    def _class_name(
        detection: Dict[str, Any],
    ) -> str:

        return str(
            detection.get(
                "class_name",
                detection.get(
                    "class",
                    "",
                ),
            )
        ).strip().lower()

    @staticmethod
    def _motion_class_name(
        evidence: Dict[str, Any],
    ) -> str:

        return str(
            evidence.get(
                "class_name",
                evidence.get(
                    "class",
                    "object",
                ),
            )
        ).strip().lower()

    def _find_moving_class(
        self,
        motion_evidence:
            List[Dict[str, Any]],
        allowed_classes: Set[str],
    ) -> Optional[Dict[str, Any]]:

        for evidence in motion_evidence:

            class_name = (
                self._motion_class_name(
                    evidence
                )
            )

            motion_state = str(
                evidence.get(
                    "motion_state",
                    "",
                )
            ).lower()

            if (
                class_name
                in allowed_classes
                and motion_state.startswith(
                    "moving"
                )
            ):
                return evidence

        return None

    @staticmethod
    def _candidate(
        consequence: str,
        probability: float,
        evidence: List[str],
        source_object: Optional[str],
    ) -> Dict[str, Any]:

        probability = round(
            max(
                0.0,
                min(
                    1.0,
                    probability,
                ),
            ),
            3,
        )

        return {
            "consequence":
                consequence,
            "probability":
                probability,
            "uncertainty":
                round(
                    1.0
                    - probability,
                    3,
                ),
            "evidence":
                evidence,
            "source_object":
                source_object,
            "observed":
                False,
        }

    @staticmethod
    def _deduplicate_candidates(
        candidates:
            List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        best: Dict[
            str,
            Dict[str, Any],
        ] = {}

        for candidate in candidates:

            name = candidate[
                "consequence"
            ]

            existing = best.get(
                name
            )

            if (
                existing is None
                or candidate[
                    "probability"
                ]
                > existing[
                    "probability"
                ]
            ):
                best[name] = (
                    candidate
                )

        return sorted(
            best.values(),
            key=lambda item:
                item["probability"],
            reverse=True,
        )