from fifth_layer.perception.track_prediction import prediction_context
"""Temporal and occlusion-aware trajectory reasoning for AISTHESIS."""

from fifth_layer.reasoners.occlusion_context import occlusion_context


from math import hypot

from fifth_layer.world_state import WorldState
from fifth_layer.expected_consequences import ExpectedConsequences
from fifth_layer.latent_state import LatentState
from fifth_layer.future_state import FutureState
from fifth_layer.reasoners.base import BaseReasoner

TEMPORAL_RAW_CONFIDENCE = 0.50
VISIBILITY_RAW_CONFIDENCE = 0.65


class TemporalPredictionReasoner(BaseReasoner):
    """
    Predict short-horizon object trajectories from observed motion.

    The reasoner uses a conservative constant-velocity model and
    optionally checks whether the projected trajectory approaches:

    - another visible object's 2D bounding box,
    - the image boundary,
    - an already observed overlap / occlusion region.

    Important:
    Occlusion prediction here concerns visibility of an observed object.
    It does NOT imply the existence of an unseen actor.
    """

    TRAJECTORY_HORIZONS = (
        0.5,
        1.0,
        2.0,
    )

    # Small expansion around another object's box.
    # This lets us detect "approaching" rather than only
    # exact future center intersection.
    OCCLUSION_MARGIN_RATIO = 0.04

    # If projected center comes within this normalized
    # distance of the frame boundary, consider it close.
    FRAME_EDGE_MARGIN_RATIO = 0.06

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------

    @staticmethod
    def _get_box_xyxy(
        detection,
    ):
        """
        Return a detection box as [x1, y1, x2, y2].

        Supports both:
        - box_xyxy
        - legacy box = [x, y, w, h]
        """

        box_xyxy = detection.get(
            "box_xyxy"
        )

        if (
            isinstance(
                box_xyxy,
                (list, tuple),
            )
            and len(box_xyxy) == 4
        ):
            x1, y1, x2, y2 = (
                map(
                    float,
                    box_xyxy,
                )
            )

            return [
                x1,
                y1,
                x2,
                y2,
            ]

        legacy_box = detection.get(
            "box"
        )

        if (
            isinstance(
                legacy_box,
                (list, tuple),
            )
            and len(legacy_box) == 4
        ):
            x, y, w, h = map(
                float,
                legacy_box,
            )

            return [
                x,
                y,
                x + w,
                y + h,
            ]

        return None

    @staticmethod
    def _center_from_box(
        box,
    ):
        if box is None:
            return None

        x1, y1, x2, y2 = box

        return [
            (
                x1 + x2
            )
            / 2.0,
            (
                y1 + y2
            )
            / 2.0,
        ]

    @staticmethod
    def _point_inside_box(
        point,
        box,
    ):
        if (
            point is None
            or box is None
        ):
            return False

        x, y = point
        x1, y1, x2, y2 = box

        return (
            x1 <= x <= x2
            and y1 <= y <= y2
        )

    @staticmethod
    def _boxes_iou(
        first,
        second,
    ):
        if (
            first is None
            or second is None
        ):
            return 0.0

        ax1, ay1, ax2, ay2 = first
        bx1, by1, bx2, by2 = second

        inter_x1 = max(
            ax1,
            bx1,
        )

        inter_y1 = max(
            ay1,
            by1,
        )

        inter_x2 = min(
            ax2,
            bx2,
        )

        inter_y2 = min(
            ay2,
            by2,
        )

        inter_width = max(
            0.0,
            inter_x2 - inter_x1,
        )

        inter_height = max(
            0.0,
            inter_y2 - inter_y1,
        )

        intersection = (
            inter_width
            * inter_height
        )

        first_area = max(
            0.0,
            ax2 - ax1,
        ) * max(
            0.0,
            ay2 - ay1,
        )

        second_area = max(
            0.0,
            bx2 - bx1,
        ) * max(
            0.0,
            by2 - by1,
        )

        union = (
            first_area
            + second_area
            - intersection
        )

        if union <= 0.0:
            return 0.0

        return (
            intersection
            / union
        )

    @staticmethod
    def _find_matching_detection(
        detections,
        class_name,
        current_center,
    ):
        """
        Match the temporal object back to the closest
        current-frame detection of the same class.
        """

        candidates = []

        for detection in detections:

            if (
                detection.get(
                    "class_name"
                )
                != class_name
            ):
                continue

            box = (
                TemporalPredictionReasoner
                ._get_box_xyxy(
                    detection
                )
            )

            center = (
                TemporalPredictionReasoner
                ._center_from_box(
                    box
                )
            )

            if center is None:
                continue

            if current_center is None:
                distance = 0.0

            else:
                distance = hypot(
                    center[0]
                    - current_center[0],
                    center[1]
                    - current_center[1],
                )

            candidates.append(
                (
                    distance,
                    detection,
                    box,
                )
            )

        if not candidates:
            return (
                None,
                None,
            )

        candidates.sort(
            key=lambda item: item[0]
        )

        return (
            candidates[0][1],
            candidates[0][2],
        )

    def _expand_box(
        self,
        box,
        image_width,
        image_height,
    ):
        if box is None:
            return None

        margin_x = (
            image_width
            * self.OCCLUSION_MARGIN_RATIO
        )

        margin_y = (
            image_height
            * self.OCCLUSION_MARGIN_RATIO
        )

        x1, y1, x2, y2 = box

        return [
            max(
                0.0,
                x1 - margin_x,
            ),
            max(
                0.0,
                y1 - margin_y,
            ),
            min(
                float(image_width),
                x2 + margin_x,
            ),
            min(
                float(image_height),
                y2 + margin_y,
            ),
        ]

    def _check_frame_boundary(
        self,
        point,
        image_width,
        image_height,
    ):
        if point is None:
            return {
                "outside_frame": False,
                "near_frame_edge": False,
            }

        x, y = point

        outside = (
            x < 0
            or x > image_width
            or y < 0
            or y > image_height
        )

        edge_margin_x = (
            image_width
            * self.FRAME_EDGE_MARGIN_RATIO
        )

        edge_margin_y = (
            image_height
            * self.FRAME_EDGE_MARGIN_RATIO
        )

        near_edge = (
            x <= edge_margin_x
            or x >= (
                image_width
                - edge_margin_x
            )
            or y <= edge_margin_y
            or y >= (
                image_height
                - edge_margin_y
            )
        )

        return {
            "outside_frame": outside,
            "near_frame_edge": near_edge,
        }

    def _find_trajectory_occluder(
        self,
        target_detection,
        target_box,
        detections,
        trajectory,
        image_width,
        image_height,
    ):
        """
        Check whether the future center of the tracked object
        enters or approaches another visible object's box.

        This is a 2D visibility heuristic, not a 3D proof
        of physical occlusion.
        """

        if target_detection is None:
            return None

        # Trajectory-based occlusion prediction is meaningful only
        # for a target with real observed motion. A stationary object
        # may overlap another box in the current frame, but that alone
        # must not create a future "continue behind" prediction.
        moving_target = any(
            point.get("center") is not None
            for point in trajectory
        )

        if not moving_target:
            return None

        target_id = target_detection.get(
            "object_id"
        )

        best_candidate = None

        for detection in detections:

            if (
                detection
                is target_detection
            ):
                continue

            if (
                target_id is not None
                and detection.get(
                    "object_id"
                )
                == target_id
            ):
                continue

            other_box = (
                self._get_box_xyxy(
                    detection
                )
            )

            if other_box is None:
                continue

            expanded_box = (
                self._expand_box(
                    other_box,
                    image_width,
                    image_height,
                )
            )

            current_iou = (
                self._boxes_iou(
                    target_box,
                    other_box,
                )
            )

            earliest_horizon = None
            predicted_intersection = False

            for point in trajectory:

                predicted_center = (
                    point.get(
                        "center"
                    )
                )

                if self._point_inside_box(
                    predicted_center,
                    expanded_box,
                ):
                    earliest_horizon = (
                        point.get(
                            "horizon_seconds"
                        )
                    )

                    predicted_intersection = True
                    break

            if (
                current_iou <= 0.0
                and not predicted_intersection
            ):
                continue

            candidate = {
                "object_id": detection.get(
                    "object_id"
                ),
                "class_name": detection.get(
                    "class_name",
                    "object",
                ),
                "confidence": detection.get(
                    "confidence"
                ),
                "current_iou": round(
                    current_iou,
                    4,
                ),
                "predicted_path_intersection": (
                    predicted_intersection
                ),
                "earliest_intersection_seconds": (
                    earliest_horizon
                ),
            }

            if best_candidate is None:
                best_candidate = candidate
                continue

            candidate_time = (
                earliest_horizon
                if earliest_horizon
                is not None
                else 999.0
            )

            best_time = (
                best_candidate.get(
                    "earliest_intersection_seconds"
                )
            )

            if best_time is None:
                best_time = 999.0

            if candidate_time < best_time:
                best_candidate = candidate

            elif (
                candidate_time
                == best_time
                and current_iou
                > best_candidate.get(
                    "current_iou",
                    0.0,
                )
            ):
                best_candidate = candidate

        return best_candidate

    # -----------------------------------------------------
    # Expected consequences
    # -----------------------------------------------------

    def infer_expected_consequences(
        self,
        world_state: WorldState,
    ) -> ExpectedConsequences:

        data = world_state.data

        motion_evidence = data.get(
            "motion_evidence",
            [],
        )

        detections = data.get(
            "detections",
            [],
        )

        image_width = float(
            data.get(
                "image_width",
                0.0,
            )
            or 0.0
        )

        image_height = float(
            data.get(
                "image_height",
                0.0,
            )
            or 0.0
        )

        predictions = {
            "temporal_evidence_present": bool(
                motion_evidence
            ),
            "moving_object_count": 0,
            "trajectory_available": False,
            "occlusion_aware": True,
        }

        if "predicted_tracks" in world_state.data:
            predictions["predicted_track_context"] = prediction_context(world_state.data, world_state.timestamp)

        if not motion_evidence:

            return ExpectedConsequences(
                predictions=predictions
            )

        moving_objects = [
            item
            for item in motion_evidence
            if item.get(
                "motion_state"
            )
            != "stationary"
        ]

        predictions[
            "moving_object_count"
        ] = len(
            moving_objects
        )

        if not moving_objects:

            predictions[
                "temporal_state"
            ] = (
                "tracked_objects_stationary"
            )

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
        predictions["track_id"] = strongest.get("track_id")

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
        ] = (
            "constant_velocity_short_horizon"
        )

        predictions[
            "expected_motion_continuation"
        ] = (
            f"{class_name}_may_continue_"
            f"{motion_state}"
        )

        # -------------------------------------------------
        # Build trajectory
        # -------------------------------------------------

        trajectory = []

        if (
            current_center is not None
            and velocity_x is not None
            and velocity_y is not None
        ):

            for horizon in (
                self.TRAJECTORY_HORIZONS
            ):

                predicted_center = [
                    round(
                        current_center[0]
                        + velocity_x
                        * horizon,
                        2,
                    ),
                    round(
                        current_center[1]
                        + velocity_y
                        * horizon,
                        2,
                    ),
                ]

                boundary_state = (
                    self._check_frame_boundary(
                        predicted_center,
                        image_width,
                        image_height,
                    )
                )

                trajectory.append(
                    {
                        "horizon_seconds": horizon,
                        "center": predicted_center,
                        "outside_frame": (
                            boundary_state[
                                "outside_frame"
                            ]
                        ),
                        "near_frame_edge": (
                            boundary_state[
                                "near_frame_edge"
                            ]
                        ),
                    }
                )

            predictions[
                "trajectory"
            ] = trajectory

            predictions[
                "trajectory_available"
            ] = True

            # Keep backward compatibility.
            one_second_point = next(
                (
                    point
                    for point in trajectory
                    if point[
                        "horizon_seconds"
                    ]
                    == 1.0
                ),
                None,
            )

            if one_second_point:

                predictions[
                    "predicted_center_1s"
                ] = one_second_point[
                    "center"
                ]

        # -------------------------------------------------
        # Boundary prediction
        # -------------------------------------------------

        outside_points = [
            point
            for point in trajectory
            if point.get(
                "outside_frame"
            )
        ]

        near_edge_points = [
            point
            for point in trajectory
            if point.get(
                "near_frame_edge"
            )
        ]

        if outside_points:

            predictions[
                "visibility_prediction"
            ] = (
                "likely_to_leave_view"
            )

            predictions[
                "estimated_exit_horizon_seconds"
            ] = outside_points[0].get(
                "horizon_seconds"
            )

        elif near_edge_points:

            predictions[
                "visibility_prediction"
            ] = (
                "approaching_frame_boundary"
            )

            predictions[
                "estimated_edge_horizon_seconds"
            ] = near_edge_points[0].get(
                "horizon_seconds"
            )

        else:

            predictions[
                "visibility_prediction"
            ] = (
                "likely_to_remain_in_view"
            )

        # -------------------------------------------------
        # Occlusion-aware trajectory
        # -------------------------------------------------

        (
            target_detection,
            target_box,
        ) = (
            self._find_matching_detection(
                detections,
                class_name,
                current_center,
            )
        )

        context = occlusion_context(data)
        if context is not None and target_detection is not None:
            hypotheses, latent_objects = context
            target_id = target_detection.get("object_id")
            matches = [item for item in hypotheses
                       if target_id is not None and item.get("object_id") == target_id
                       and item.get("class_name") == class_name]
            predictions["current_occlusion_probability"] = max(
                (item["occlusion_probability"] for item in matches), default=0.0)
            predictions["current_occlusion_possible"] = any(
                target_id is not None and item.get("object_id") == target_id
                and item.get("class_name") == class_name
                for item in latent_objects)
            # Current visibility context supports the trajectory result; it does
            # not establish a future intersection or an additional hidden actor.

        occluder = (
            self._find_trajectory_occluder(
                target_detection,
                target_box,
                detections,
                trajectory,
                image_width,
                image_height,
            )
        )

        if occluder is not None:

            predictions[
                "potential_occluder"
            ] = occluder

            current_iou = float(
                occluder.get(
                    "current_iou",
                    0.0,
                )
            )

            future_intersection = bool(
                occluder.get(
                    "predicted_path_intersection"
                )
            )

            if (
                current_iou > 0.0
                and future_intersection
            ):

                predictions[
                    "occlusion_prediction"
                ] = (
                    "occlusion_interaction_possible"
                )

            elif future_intersection:

                predictions[
                    "occlusion_prediction"
                ] = (
                    "approaching_occlusion"
                )

            elif current_iou > 0.0:

                predictions[
                    "occlusion_prediction"
                ] = (
                    "current_overlap_observed"
                )

            else:

                predictions[
                    "occlusion_prediction"
                ] = (
                    "no_predicted_occlusion"
                )

        else:

            predictions[
                "occlusion_prediction"
            ] = (
                "no_predicted_occlusion"
            )

        return ExpectedConsequences(
            predictions=predictions
        )

    # -----------------------------------------------------
    # Latent state
    # -----------------------------------------------------

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

        occlusion_prediction = (
            features.get(
                "occlusion_prediction"
            )
        )

        visibility_prediction = (
            features.get(
                "visibility_prediction"
            )
        )

        if (
            moving_count > 0
            and occlusion_prediction
            == "approaching_occlusion"
        ):

            features[
                "latent_temporal_state"
            ] = (
                "trajectory_toward_occlusion"
            )

        elif (
            moving_count > 0
            and occlusion_prediction
            == "occlusion_interaction_possible"
        ):

            features[
                "latent_temporal_state"
            ] = (
                "visibility_loss_possible"
            )

        elif (
            moving_count > 0
            and visibility_prediction
            == "likely_to_leave_view"
        ):

            features[
                "latent_temporal_state"
            ] = (
                "trajectory_toward_view_exit"
            )

        elif moving_count > 0:

            features[
                "latent_temporal_state"
            ] = (
                "motion_continuation_possible"
            )

        elif features.get(
            "temporal_evidence_present"
        ):

            features[
                "latent_temporal_state"
            ] = (
                "stationary_state_observed"
            )

        else:

            features[
                "latent_temporal_state"
            ] = (
                "insufficient_temporal_evidence"
            )

        return LatentState(
            features=features
        )

    # -----------------------------------------------------
    # Future state
    # -----------------------------------------------------

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
        # Preserve the live UI's existing raw confidence policy in the reasoner.
        future_data['raw_confidence'] = (VISIBILITY_RAW_CONFIDENCE if temporal_state in {
            'trajectory_toward_occlusion', 'visibility_loss_possible',
            'trajectory_toward_view_exit'} else TEMPORAL_RAW_CONFIDENCE)
        future_data['prediction_type'] = 'trajectory_position'

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

        potential_occluder = (
            latent_state.features.get(
                "potential_occluder"
            )
            or {}
        )

        occluder_class = (
            potential_occluder.get(
                "class_name",
                "visible_object",
            )
        )

        if (
            temporal_state
            == "trajectory_toward_occlusion"
        ):

            future_data[
                "predicted_event"
            ] = (
                f"{class_name}_may_become_"
                f"occluded_by_{occluder_class}"
            )

            future_data[
                "future_visibility_state"
            ] = (
                "partial_visibility_loss_possible"
            )

        elif (
            temporal_state
            == "visibility_loss_possible"
        ):

            future_data[
                "predicted_event"
            ] = (
                f"{class_name}_may_continue_"
                f"behind_{occluder_class}"
            )

            future_data[
                "future_visibility_state"
            ] = (
                "temporary_visibility_loss_possible"
            )

        elif (
            temporal_state
            == "trajectory_toward_view_exit"
        ):

            future_data[
                "predicted_event"
            ] = (
                f"{class_name}_may_leave_view"
            )

            future_data[
                "future_visibility_state"
            ] = (
                "out_of_frame_possible"
            )

        elif (
            temporal_state
            == "motion_continuation_possible"
        ):

            future_data[
                "predicted_event"
            ] = (
                f"{class_name}_may_continue_"
                f"{motion_state}"
            )

            future_data[
                "future_visibility_state"
            ] = (
                latent_state.features.get(
                    "visibility_prediction",
                    "unknown",
                )
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

            future_data[
                "future_visibility_state"
            ] = (
                "likely_to_remain_in_view"
            )

        else:

            future_data[
                "predicted_event"
            ] = "indeterminate"

            future_data[
                "future_visibility_state"
            ] = "indeterminate"

        return FutureState(
            horizon=2.0,
            data=future_data,
        )
