"""Scene-aware local reasoner for Fifth Layer Engine v0.14."""

from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.future_state import FutureState
from fifth_layer.latent_state import LatentState
from fifth_layer.reasoners.base import BaseReasoner
from fifth_layer.world_state import WorldState


class SceneReasoner(BaseReasoner):
    """Reason over structured scene information without an LLM."""

    def infer_expected_consequences(
        self,
        world_state: WorldState,
    ) -> ExpectedConsequences:
        scene_relations = world_state.data.get(
            "scene_relations",
            [],
        )

        predictions = {}

        for relation in scene_relations:
            relation_type = relation.get(
                "relation"
            )

            if relation_type == "overlapping":
                predictions[
                    "spatial_interaction_possible"
                ] = True

            elif relation_type == "near":
                predictions[
                    "close_proximity_detected"
                ] = True

            elif relation_type == "far":
                predictions[
                    "spatial_separation_detected"
                ] = True

            elif relation_type == "separate":
                predictions[
                    "objects_currently_separate"
                ] = True

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

        if features.get(
            "spatial_interaction_possible"
        ):
            features[
                "latent_scene_state"
            ] = "possible_object_interaction"

        elif features.get(
            "close_proximity_detected"
        ):
            features[
                "latent_scene_state"
            ] = "objects_in_close_proximity"

        elif features.get(
            "spatial_separation_detected"
        ):
            features[
                "latent_scene_state"
            ] = "objects_spatially_separated"

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

        latent_scene_state = (
            latent_state.features.get(
                "latent_scene_state"
            )
        )

        if (
            latent_scene_state
            == "possible_object_interaction"
        ):
            future_data[
                "predicted_scene_event"
            ] = "object_interaction_may_occur"

        elif (
            latent_scene_state
            == "objects_in_close_proximity"
        ):
            future_data[
                "predicted_scene_event"
            ] = "objects_may_interact"

        elif (
            latent_scene_state
            == "objects_spatially_separated"
        ):
            future_data[
                "predicted_scene_event"
            ] = "no_immediate_spatial_interaction"

        return FutureState(
            horizon=1.0,
            data=future_data,
        )