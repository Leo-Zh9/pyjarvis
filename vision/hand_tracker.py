"""
MediaPipe Hands wrapper providing a clean interface for downstream modules.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import cv2

logger = logging.getLogger(__name__)

try:
    mp: Optional[Any] = importlib.import_module("mediapipe")
except ModuleNotFoundError:
    mp = None
    logger.warning(
        "MediaPipe is not installed. Hand tracking will be disabled until you run "
        "`pip install mediapipe`."
    )


@dataclass
class HandData:
    """Minimal structure describing a detected hand."""

    landmarks: List[tuple[float, float, float]]
    handedness: str


class HandTracker:
    """Encapsulates MediaPipe Hands configuration and processing."""

    def __init__(
        self,
        max_num_hands: int = 2,
        detection_confidence: float = 0.2,
        tracking_confidence: float = 0.2,
    ) -> None:
        self._hands = None
        self._mp_hands = None

        if mp is None:
            self._available = False
            return

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            max_num_hands=max_num_hands,
            model_complexity=0,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._available = True

    def process_frame(self, frame) -> List[HandData]:
        """
        Run hand detection on the provided frame.

        Returns a list of ``HandData`` objects—one per detected hand.
        """
        if frame is None or not self._available or self._hands is None:
            return []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print("Processing frame…")
        results = self._hands.process(rgb_frame)
        print("multi_hand_landmarks:", results.multi_hand_landmarks)
        print("multi_handedness:", results.multi_handedness)

        hand_data: List[HandData] = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for landmark_list, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                landmarks = [
                    (lm.x, lm.y, lm.z) for lm in landmark_list.landmark
                ]
                label = handedness.classification[0].label
                hand_data.append(HandData(landmarks=landmarks, handedness=label))

        return hand_data

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands:
            self._hands.close()


def correct_fingertip(
    base: tuple[float, float, float],
    tip: tuple[float, float, float],
    factor: float = 1.35,
) -> tuple[float, float, float]:
    """
    Extend a fingertip along the base->tip direction to compensate for jitter.
    """
    dx = tip[0] - base[0]
    dy = tip[1] - base[1]
    dz = tip[2] - base[2]
    return (
        base[0] + dx * factor,
        base[1] + dy * factor,
        base[2] + dz * factor,
    )

