"""
Gesture recognition tuned for pinch, scroll, and pose detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, TYPE_CHECKING, Tuple

from vision.hand_tracker import correct_fingertip

if TYPE_CHECKING:
    from vision.hand_tracker import HandData


class GestureType(str, Enum):
    NONE = "NONE"
    LEFT_CLICK = "LEFT_CLICK_GESTURE"
    RIGHT_CLICK = "RIGHT_CLICK_GESTURE"
    SCROLL = "SCROLL"


@dataclass
class GestureResult:
    gesture: GestureType = GestureType.NONE
    handedness: Optional[str] = None
    details: Dict[str, object] = field(default_factory=dict)


def distance_3d(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def distance_xy(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Euclidean distance using x/y components only."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.hypot(dx, dy)


def finger_angle_score(
    dip: Tuple[float, float, float],
    pip: Tuple[float, float, float],
    mcp: Tuple[float, float, float],
) -> float:
    """
    Return 0-1 score of finger straightness based on the angle at the PIP joint.
    """
    bax = dip[0] - pip[0]
    bay = dip[1] - pip[1]
    bcx = mcp[0] - pip[0]
    bcy = mcp[1] - pip[1]
    mag1 = math.hypot(bax, bay)
    mag2 = math.hypot(bcx, bcy)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 1.0
    dot = bax * bcx + bay * bcy
    cosang = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    angle = math.degrees(math.acos(cosang))
    return max(0.0, min(angle / 180.0, 1.0))


POSE_FIST = "FIST"
POSE_PALM = "OPEN_PALM"


@dataclass
class _HandPinchState:
    left_active: bool = False
    left_confidence: int = 0
    right_active: bool = False
    right_confidence: int = 0


class _PinchRecognizer:
    """Tracks pinch state across frames for multiple hands."""

    def __init__(self) -> None:
        self._states: Dict[str, _HandPinchState] = {}

    def reset(self) -> None:
        self._states.clear()

    def process(
        self,
        hands: List["HandData"],
        index_inside_flags: Optional[List[Optional[bool]]] = None,
        middle_inside_flags: Optional[List[Optional[bool]]] = None,
    ) -> Optional[GestureResult]:
        new_states: Dict[str, _HandPinchState] = {}
        candidates: List[Dict[str, object]] = []
        mode_toggle: Optional[Dict[str, object]] = None
        best_debug: Optional[Dict[str, object]] = None
        best_debug_score: float = float("inf")

        for idx, hand in enumerate(hands):
            if len(hand.landmarks) <= 20:
                continue

            key = f"{hand.handedness or 'Unknown'}_{idx}"
            state = self._states.get(key, _HandPinchState())
            new_states[key] = state

            pose = self._detect_pose(hand)
            if pose == POSE_PALM:
                mode_toggle = {
                    "mode_toggle": "ENABLE",
                    "hand_index": idx,
                    "handedness": hand.handedness or "Unknown",
                }
            elif pose == POSE_FIST and mode_toggle is None:
                mode_toggle = {
                    "mode_toggle": "DISABLE",
                    "hand_index": idx,
                    "handedness": hand.handedness or "Unknown",
                }

            inside_index = (
                index_inside_flags[idx]
                if index_inside_flags and idx < len(index_inside_flags)
                else True
            )
            inside_middle = (
                middle_inside_flags[idx]
                if middle_inside_flags and idx < len(middle_inside_flags)
                else True
            )

            wrist = hand.landmarks[0]
            index_mcp = hand.landmarks[5]
            hand_scale = max(distance_xy(wrist, index_mcp), 1e-6)
            hand_depth = abs(wrist[2])
            depth_factor = 1.0 + hand_depth * 0.5
            # Increased thresholds for more tolerance with imperfect mapping
            base_thr_tip = hand_scale * 0.45  # Increased from 0.35
            base_thr_cluster = hand_scale * 0.50  # Increased from 0.40
            thr_tip = base_thr_tip * depth_factor
            thr_cluster = base_thr_cluster * depth_factor
            spread_limit = hand_scale * 0.7

            thumb_cluster = [
                correct_fingertip(hand.landmarks[2], hand.landmarks[3], 1.25),
                correct_fingertip(hand.landmarks[3], hand.landmarks[4], 1.35),
            ]

            index_tip = correct_fingertip(hand.landmarks[7], hand.landmarks[8], 1.55)
            index_cluster = [
                correct_fingertip(hand.landmarks[6], hand.landmarks[7], 1.35),
                index_tip,
            ]

            middle_tip = correct_fingertip(hand.landmarks[11], hand.landmarks[12], 1.55)
            middle_cluster = [
                correct_fingertip(hand.landmarks[10], hand.landmarks[11], 1.35),
                middle_tip,
            ]

            def _min_cluster_distance(
                cluster_a: List[Tuple[float, float, float]],
                cluster_b: List[Tuple[float, float, float]],
            ) -> float:
                """Use 3D distance to handle hand rotation and imperfect mapping."""
                return min(
                    distance_3d(a, b) for a in cluster_a for b in cluster_b
                )

            index_min_dist = _min_cluster_distance(index_cluster, thumb_cluster)
            middle_min_dist = _min_cluster_distance(middle_cluster, thumb_cluster)
            
            # Fallback for cluster distances: also check raw landmark clusters
            if len(hand.landmarks) > 8:
                raw_index_cluster = [hand.landmarks[7], hand.landmarks[8]]
                raw_index_cluster_dist = _min_cluster_distance(raw_index_cluster, thumb_cluster)
                index_min_dist = min(index_min_dist, raw_index_cluster_dist)
            if len(hand.landmarks) > 12:
                raw_middle_cluster = [hand.landmarks[11], hand.landmarks[12]]
                raw_middle_cluster_dist = _min_cluster_distance(raw_middle_cluster, thumb_cluster)
                middle_min_dist = min(middle_min_dist, raw_middle_cluster_dist)

            thumb_tip = thumb_cluster[1]
            # Use 3D distance for absolute distance regardless of orientation
            index_tip_dist = distance_3d(index_tip, thumb_tip)
            middle_tip_dist = distance_3d(middle_tip, thumb_tip)
            
            # Fallback: if corrected fingertip mapping seems off (hand facing camera),
            # also check raw landmark distance and use the minimum
            if len(hand.landmarks) > 8 and len(hand.landmarks) > 4:
                raw_index_dist = distance_3d(hand.landmarks[8], hand.landmarks[4])
                index_tip_dist = min(index_tip_dist, raw_index_dist)
            if len(hand.landmarks) > 12 and len(hand.landmarks) > 4:
                raw_middle_dist = distance_3d(hand.landmarks[12], hand.landmarks[4])
                middle_tip_dist = min(middle_tip_dist, raw_middle_dist)

            index_angle = finger_angle_score(
                index_cluster[0], hand.landmarks[6], hand.landmarks[5]
            )
            middle_angle = finger_angle_score(
                middle_cluster[0], hand.landmarks[10], hand.landmarks[9]
            )

            # Use 3D distance for spread to handle hand rotation
            spread = distance_3d(index_tip, hand.landmarks[16]) if len(hand.landmarks) > 16 else 0.0

            # More tolerant extended check for rotated hands
            index_extended = index_angle > 0.20  # Lowered from 0.30
            middle_extended = middle_angle > 0.20  # Lowered from 0.30

            left_extra_details: Dict[str, object] = {
                "tip_dist": index_tip_dist,
                "cluster_dist": index_min_dist,
                "thr_tip": thr_tip,
                "thr_cluster": thr_cluster,
                "angle": index_angle,
                "spread": spread,
            }

            left_candidate = self._update_branch(
                state=state,
                active_attr="left_active",
                confidence_attr="left_confidence",
                inside=bool(inside_index),
                extended=index_extended,
                tip_dist=index_tip_dist,
                cluster_dist=index_min_dist,
                thr_tip=thr_tip,
                thr_cluster=thr_cluster,
                spread=spread,
                spread_limit=spread_limit,
                angle_score=index_angle,
                tip_y=index_tip[1],
                gesture=GestureType.LEFT_CLICK,
                hand_landmarks=hand.landmarks,
                extra_details=left_extra_details,
                hand_index=idx,
                finger_type="index",
            )

            right_extra_details: Dict[str, object] = {
                "tip_dist": middle_tip_dist,
                "cluster_dist": middle_min_dist,
                "thr_tip": thr_tip,
                "thr_cluster": thr_cluster,
                "angle": middle_angle,
                "spread": spread,
            }

            right_candidate = self._update_branch(
                state=state,
                active_attr="right_active",
                confidence_attr="right_confidence",
                inside=bool(inside_middle),
                extended=middle_extended,
                tip_dist=middle_tip_dist,
                cluster_dist=middle_min_dist,
                thr_tip=thr_tip,
                thr_cluster=thr_cluster,
                spread=spread,
                spread_limit=spread_limit,
                angle_score=middle_angle,
                tip_y=middle_tip[1],
                gesture=GestureType.RIGHT_CLICK,
                hand_landmarks=hand.landmarks,
                extra_details=right_extra_details,
                hand_index=idx,
                finger_type="middle",
            )

            for candidate in (left_candidate, right_candidate):
                if candidate is None:
                    continue
                details_for_debug = dict(candidate.get("details", {}))
                details_for_debug["handedness"] = hand.handedness or "Unknown"
                score = float(details_for_debug.get("tip_dist", 0.0)) / max(
                    float(details_for_debug.get("thr_tip", 1e-6)), 1e-6
                )
                if score < best_debug_score:
                    best_debug_score = score
                    best_debug = details_for_debug

                if candidate.get("active"):
                    candidate["handedness"] = hand.handedness or "Unknown"
                    candidates.append(candidate)

        self._states = new_states

        result: Optional[GestureResult] = None

        if candidates:
            chosen = min(candidates, key=lambda c: c["priority"])  # type: ignore[index]
            chosen_details = dict(chosen.get("details", {}))  # type: ignore[arg-type]
            if "hand_index" not in chosen_details and "hand_index" in chosen:
                chosen_details["hand_index"] = chosen["hand_index"]  # type: ignore[index]
            result = GestureResult(
                gesture=chosen["gesture"],  # type: ignore[arg-type]
                handedness=chosen["handedness"],  # type: ignore[arg-type]
                details=chosen_details,
            )

        # If no gesture candidates were found, but we observed hands,
        # return the best debug metrics so the UI can still surface data.
        if result is None and best_debug is not None:
            result = GestureResult(
                gesture=GestureType.NONE,
                handedness=best_debug.get("handedness"),  # type: ignore[arg-type]
                details=best_debug,
            )

        if mode_toggle:
            toggle_details = {
                "mode_toggle": mode_toggle["mode_toggle"],  # type: ignore[index]
                "mode_toggle_hand": mode_toggle["handedness"],  # type: ignore[index]
                "mode_toggle_hand_index": mode_toggle["hand_index"],  # type: ignore[index]
            }
            if result:
                result.details.update(toggle_details)
            else:
                result = GestureResult(
                    gesture=GestureType.NONE,
                    handedness=mode_toggle["handedness"],  # type: ignore[index]
                    details=toggle_details,
                )

        return result

    def _update_branch(
        self,
        *,
        state: _HandPinchState,
        active_attr: str,
        confidence_attr: str,
        inside: bool,
        extended: bool,
        tip_dist: float,
        cluster_dist: float,
        thr_tip: float,
        thr_cluster: float,
        spread: float,
        spread_limit: float,
        angle_score: float,
        tip_y: float,
        gesture: GestureType,
        hand_landmarks: List[Tuple[float, float, float]],
        extra_details: Dict[str, object],
        hand_index: int,
        finger_type: str,
    ) -> Optional[Dict[str, object]]:
        active = getattr(state, active_attr)
        confidence = getattr(state, confidence_attr)

        thr_tip = max(thr_tip, 1e-6)
        thr_cluster = max(thr_cluster, 1e-6)

        # ----- New pinch scoring -----
        tip_ratio = tip_dist / thr_tip
        cluster_ratio = cluster_dist / thr_cluster
        tip_strength = max(0.0, 1.0 - tip_ratio)
        cluster_strength = max(0.0, 1.0 - cluster_ratio)
        # combined score: mostly tip distance, cluster for stability
        combined = 0.8 * tip_strength + 0.2 * cluster_strength

        details = dict(extra_details)
        details.update(
            {
                "combined": combined,
                "tip_ratio": tip_ratio,
                "cluster_ratio": cluster_ratio,
                "tip_strength": tip_strength,
                "cluster_strength": cluster_strength,
                "angle_score": angle_score,
                "spread": spread,
                "tip_dist": tip_dist,
                "cluster_dist": cluster_dist,
                "thr_tip": thr_tip,
                "thr_cluster": thr_cluster,
                "hand_index": hand_index,
            }
        )

        # Suppression: only when tips are far apart or inverted hand
        invalid = False
        if tip_dist > thr_tip * 2.0:
            invalid = True
        # Keep a basic sanity check for inverted hand pose
        if gesture == GestureType.LEFT_CLICK and len(hand_landmarks) > 4 and len(hand_landmarks) > 8:
            thumb_tip_y = hand_landmarks[4][1]
            if tip_y < thumb_tip_y - 0.04:
                invalid = True

        # ----- Pinch classification -----
        # Lowered thresholds for more accessible pinch detection
        STRONG_PINCH = combined > 0.45
        WEAK_PINCH = combined > 0.30

        if invalid:
            STRONG_PINCH = False
            WEAK_PINCH = False

        # Raise confidence
        if STRONG_PINCH:
            confidence = min(confidence + 1, 6)
        else:
            confidence = max(confidence - 1, -6)

        # Update 'active' based on confidence thresholds
        if confidence >= 2:
            active = True
        if confidence <= -3:
            active = False

        # Store debug values
        details.update(
            {
                "pinch_confidence": confidence,
                "strong": STRONG_PINCH,
                "weak": WEAK_PINCH,
                "active": active,
            }
        )

        setattr(state, active_attr, active)
        setattr(state, confidence_attr, confidence)

        # Decide gesture (always return debug; active determines candidacy)
        return {
            "gesture": gesture,
            "priority": tip_dist / max(thr_tip, 1e-6),
            "details": details,
            "hand_index": hand_index,
            "active": active,
            "finger_type": finger_type,
        }

    @staticmethod
    def _detect_pose(hand: "HandData") -> Optional[str]:
        finger_pairs = (
            (8, 6),   # index
            (12, 10),  # middle
            (16, 14),  # ring
            (20, 18),  # pinky
        )
        extended_flags = [hand.landmarks[tip][1] < hand.landmarks[pip][1] for tip, pip in finger_pairs]
        curled_flags = [hand.landmarks[tip][1] > hand.landmarks[pip][1] for tip, pip in finger_pairs]

        thumb_tip = hand.landmarks[4]
        thumb_ip = hand.landmarks[3]
        handedness = (hand.handedness or "").lower()

        if handedness == "right":
            thumb_folded = thumb_tip[0] < thumb_ip[0]
        elif handedness == "left":
            thumb_folded = thumb_tip[0] > thumb_ip[0]
        else:
            thumb_folded = thumb_tip[1] > thumb_ip[1]

        if all(extended_flags):
            return POSE_PALM
        if all(curled_flags) and thumb_folded:
            return POSE_FIST
        return None


_PINCH_RECOGNIZER = _PinchRecognizer()


def recognize_gestures(
    hands: Optional[List["HandData"]],
    index_inside_flags: Optional[List[Optional[bool]]] = None,
    middle_inside_flags: Optional[List[Optional[bool]]] = None,
) -> GestureResult:
    """
    Inspect detected hands and return the current gesture.
    """
    if not hands:
        _PINCH_RECOGNIZER.reset()
        return GestureResult()

    result = _PINCH_RECOGNIZER.process(
        hands, index_inside_flags=index_inside_flags, middle_inside_flags=middle_inside_flags
    )
    return result if result else GestureResult()