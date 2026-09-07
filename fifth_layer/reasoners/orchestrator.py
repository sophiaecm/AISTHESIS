from fifth_layer.reasoners.semantic_conflict import SemanticConflictReasoner
from fifth_layer.reasoners.active_perception import ActivePerceptionReasoner
from fifth_layer.reasoners.occlusion import OcclusionReasoner
from fifth_layer.reasoners.scene import SceneReasoner
from fifth_layer.reasoners.sensor_fusion import SensorFusionReasoner
from fifth_layer.reasoners.temporal_prediction import TemporalPredictionReasoner
from fifth_layer.world_state import WorldState


class AisthesisOrchestrator:
    """
    Central reasoning coordinator for AISTHESIS.

    Coordinates perception and reasoning modules and produces
    expected consequences, latent states, future predictions,
    uncertainty, and risk signals.
    """

    def __init__(self):
        self.semantic_conflict_reasoner = (
            SemanticConflictReasoner()
        )

        self.active_perception_reasoner = (
            ActivePerceptionReasoner()
        )

        self.occlusion_reasoner = (
            OcclusionReasoner()
        )

        self.scene_reasoner = (
            SceneReasoner()
        )

        self.temporal_prediction_reasoner = (
            TemporalPredictionReasoner()
        )

        self.sensor_fusion_reasoner = (
            SensorFusionReasoner()
        )

    def analyze(
        self,
        world_state: WorldState,
    ) -> dict:

        # --------------------------------------------------
        # 1. Semantic Conflict
        # --------------------------------------------------

        semantic_result = (
            self.semantic_conflict_reasoner.analyze(
                world_state
            )
        )

        # --------------------------------------------------
        # 2. Active Perception
        # --------------------------------------------------

        active_query = (
            self.active_perception_reasoner.build_query(
                semantic_conflict_result=semantic_result,
                scene_description=world_state.data.get(
                    "scene_description",
                    "",
                ),
            )
        )

        # --------------------------------------------------
        # 3. Occlusion Reasoning
        # --------------------------------------------------

        occlusion_expected = (
            self.occlusion_reasoner
            .infer_expected_consequences(
                world_state
            )
        )

        occlusion_latent = (
            self.occlusion_reasoner
            .infer_latent_state(
                world_state,
                occlusion_expected,
            )
        )

        occlusion_future = (
            self.occlusion_reasoner
            .infer_future_state(
                occlusion_latent
            )
        )

        # --------------------------------------------------
        # 4. Scene Reasoning
        # --------------------------------------------------

        scene_expected = (
            self.scene_reasoner
            .infer_expected_consequences(
                world_state
            )
        )

        scene_latent = (
            self.scene_reasoner
            .infer_latent_state(
                world_state,
                scene_expected,
            )
        )

        scene_future = (
            self.scene_reasoner
            .infer_future_state(
                scene_latent
            )
        )

        # --------------------------------------------------
        # 5. Temporal Prediction
        # --------------------------------------------------

        temporal_expected = (
            self.temporal_prediction_reasoner
            .infer_expected_consequences(
                world_state
            )
        )

        temporal_latent = (
            self.temporal_prediction_reasoner
            .infer_latent_state(
                world_state,
                temporal_expected,
            )
        )

        temporal_future = (
            self.temporal_prediction_reasoner
            .infer_future_state(
                temporal_latent
            )
        )

        # --------------------------------------------------
        # 6. Sensor Fusion
        # --------------------------------------------------

        fusion_expected = (
            self.sensor_fusion_reasoner
            .infer_expected_consequences(
                world_state
            )
        )

        fusion_latent = (
            self.sensor_fusion_reasoner
            .infer_latent_state(
                world_state,
                fusion_expected,
            )
        )

        fusion_future = (
            self.sensor_fusion_reasoner
            .infer_future_state(
                fusion_latent
            )
        )

        # --------------------------------------------------
        # 7. Unified AISTHESIS Result
        # --------------------------------------------------

        return {
            "semantic_conflict": semantic_result,

            "active_perception": active_query,

            "occlusion": {
                "expected": (
                    occlusion_expected.predictions
                ),
                "latent": (
                    occlusion_latent.features
                ),
                "future": (
                    occlusion_future.data
                ),
            },

            "scene": {
                "expected": (
                    scene_expected.predictions
                ),
                "latent": (
                    scene_latent.features
                ),
                "future": (
                    scene_future.data
                ),
            },

            "temporal": {
                "expected": (
                    temporal_expected.predictions
                ),
                "latent": (
                    temporal_latent.features
                ),
                "future": (
                    temporal_future.data
                ),
            },

            "sensor_fusion": {
                "expected": (
                    fusion_expected.predictions
                ),
                "latent": (
                    fusion_latent.features
                ),
                "future": (
                    fusion_future.data
                ),
            },
        }