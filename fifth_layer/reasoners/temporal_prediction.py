"""Temporal prediction reasoning for AISTHESIS."""

from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.latent_state import LatentState
from fifth_layer.future_state import FutureState
from fifth_layer.reasoners.base import BaseReasoner


class TemporalPredictionReasoner(BaseReasoner):
    """
    Predict short-term object motion from temporal perception evidence.

    This reasoner does not assume hidden actors or interactions.
    It only extrapolates observed motion using a simple
    constant-velocity assumption.
    """

    def infer_expected_consequences(
        self,
        world_state: WorldState,
    ) -> ExpectedConsequences:
        data = world_state.data

        motion_evidence = data.get(
            "motion_evidence",
            [],
        )

        predictions = {
            "temporal_evidence_present": bool(
                motion_evidence
            ),
            "moving_object_count": 0,
        }

        if not motion_evidence:
            return ExpectedConsequences(
                predictions=predictions
            )

        moving_objects = [
            item
            for item in motion_evidence
            if item.get("motion_state")
            != "stationary"
        ]

        predictions[
            "moving_object_count"
        ] = len(moving_objects)

        if not moving_objects:
            predictions[
                "temporal_state"
            ] = "tracked_objects_stationary"

            return ExpectedConsequences(
                predictions=predictions
            )

        strongest = max(
            moving_objects,
            key=lambda item: item.get(
                "normalized_motion",
                0.0,
            ),
        )

        class_name = strongest.get(
            "class_name",
            "object",
        )

        motion_state = strongest.get(
            "motion_state",
            "moving",
        )

        current_center = strongest.get(
            "current_center"
        )

        velocity_x = strongest.get(
            "velocity_x"
        )

        velocity_y = strongest.get(
            "velocity_y"
        )

        predictions[
            "strongest_moving_object"
        ] = class_name

        predictions[
            "motion_state"
        ] = motion_state

        predictions[
            "current_center"
        ] = current_center

        predictions[
            "velocity_x"
        ] = velocity_x

        predictions[
            "velocity_y"
        ] = velocity_y

        predictions[
            "prediction_basis"
        ] = "constant_velocity_assumption"

        # One-second short-horizon extrapolation.
        if (
            current_center is not None
            and velocity_x is not None
            and velocity_y is not None
        ):
            predicted_center = [
                round(
                    current_center[0]
                    + velocity_x,
                    2,
                ),
                round(
                    current_center[1]
                    + velocity_y,
                    2,
                ),
            ]

            predictions[
                "predicted_center_1s"
            ] = predicted_center

        predictions[
            "expected_motion_continuation"
        ] = (
            f"{class_name}_may_continue_"
            f"{motion_state}"
        )

        return ExpectedConsequences(
            predictions=predictions
        )

    def infer_latent_state(
        self,
        world_state: WorldState,
        expected_consequences: ExpectedConsequences,
    ) -> LatentState:
        features = dict(
            world_state.data
        )

        features.update(
            expected_consequences.predictions
        )

        moving_count = features.get(
            "moving_object_count",
            0,
        )

        if moving_count > 0:
            features[
                "latent_temporal_state"
            ] = "motion_continuation_possible"

        elif features.get(
            "temporal_evidence_present"
        ):
            features[
                "latent_temporal_state"
            ] = "stationary_state_observed"

        else:
            features[
                "latent_temporal_state"
            ] = "insufficient_temporal_evidence"

        return LatentState(
            features=features
        )

    def infer_future_state(
        self,
        latent_state: LatentState,
    ) -> FutureState:
        future_data = dict(
            latent_state.features
        )

        temporal_state = (
            latent_state.features.get(
                "latent_temporal_state"
            )
        )

        if (
            temporal_state
            == "motion_continuation_possible"
        ):
            class_name = (
                latent_state.features.get(
                    "strongest_moving_object",
                    "object",
                )
            )

            motion_state = (
                latent_state.features.get(
                    "motion_state",
                    "moving",
                )
            )

            future_data[
                "predicted_event"
            ] = (
                f"{class_name}_may_continue_"
                f"{motion_state}"
            )

        elif (
            temporal_state
            == "stationary_state_observed"
        ):
            future_data[
                "predicted_event"
            ] = (
                "tracked_objects_likely_remain_stationary"
            )

        else:
            future_data[
                "predicted_event"
            ] = "indeterminate"

        return FutureState(
            horizon=1.0,
            data=future_data,
        )