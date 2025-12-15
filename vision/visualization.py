"""
Drawing helpers for the air touchscreen live preview.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import cv2
import importlib

from vision.hand_tracker import HandData, correct_fingertip

Color = Tuple[int, int, int]
Point = Tuple[int, int]

try:
    mp = importlib.import_module("mediapipe")
    landmark_pb2 = importlib.import_module(
        "mediapipe.framework.formats.landmark_pb2"
    )
    _mp_drawing = mp.solutions.drawing_utils
    _mp_styles = mp.solutions.drawing_styles
    _mp_hands = mp.solutions.hands
except ModuleNotFoundError:
    mp = None
    landmark_pb2 = None
    _mp_drawing = None
    _mp_styles = None
    _mp_hands = None


def draw_interaction_box(
    frame,
    top_left: Point,
    bottom_right: Point,
    color: Color = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw a rectangular region that indicates the interaction zone."""
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)


def draw_skeleton(frame, hand: HandData) -> None:
    """Render MediaPipe's default 21-point hand skeleton."""
    if (
        landmark_pb2 is None
        or _mp_drawing is None
        or _mp_styles is None
        or _mp_hands is None
    ):
        return

    landmark_list = landmark_pb2.NormalizedLandmarkList()
    for x, y, z in hand.landmarks:
        landmark = landmark_list.landmark.add()
        landmark.x = x
        landmark.y = y
        landmark.z = z

    _mp_drawing.draw_landmarks(
        frame,
        landmark_list,
        _mp_hands.HAND_CONNECTIONS,
        _mp_styles.get_default_hand_landmarks_style(),
        _mp_styles.get_default_hand_connections_style(),
    )


def draw_hand_landmarks(
    frame,
    hands: Sequence[HandData],
    inside_flags: Optional[Sequence[Optional[bool]]] = None,
) -> None:
    """
    Render fingertip indicators for each detected hand.

    The index fingertip is colored green when inside the interaction box,
    red when outside, and cyan when status is unknown.
    """
    if not hands:
        return

    frame_height, frame_width = frame.shape[:2]
    default_index_color: Color = (255, 255, 0)  # Cyan (BGR)
    thumb_color: Color = (0, 255, 255)  # Yellow
    middle_color: Color = (255, 0, 255)  # Magenta
    inside_color: Color = (0, 255, 0)  # Green
    outside_color: Color = (0, 0, 255)  # Red

    for idx, hand in enumerate(hands):
        pixel_points = [
            _normalized_to_pixel(point, frame_width, frame_height)
            for point in hand.landmarks
        ]

        # Draw subtle markers for every landmark to visualize conversion.
        for point in pixel_points:
            cv2.circle(frame, point, 2, (200, 200, 200), -1)

        index_color = default_index_color
        if inside_flags and idx < len(inside_flags):
            inside_state = inside_flags[idx]
            if inside_state is True:
                index_color = inside_color
            elif inside_state is False:
                index_color = outside_color

        _draw_fingertip(frame, pixel_points, 8, index_color)
        _draw_fingertip(frame, pixel_points, 4, thumb_color)
        _draw_fingertip(frame, pixel_points, 12, middle_color)

        if len(hand.landmarks) > 8:
            corrected_tip = correct_fingertip(
                hand.landmarks[7], hand.landmarks[8], 1.35
            )
            cx = int(max(0.0, min(corrected_tip[0], 0.9999)) * frame_width)
            cy = int(max(0.0, min(corrected_tip[1], 0.9999)) * frame_height)
            cv2.circle(frame, (cx, cy), 4, (0, 200, 0), 2)


def _draw_fingertip(
    frame, pixel_points: Sequence[Point], landmark_index: int, color: Color
) -> None:
    if landmark_index >= len(pixel_points):
        return
    cv2.circle(frame, pixel_points[landmark_index], 8, color, -1)


def _normalized_to_pixel(
    landmark: Tuple[float, float, float], width: int, height: int
) -> Point:
    x = max(0.0, min(landmark[0], 0.9999))
    y = max(0.0, min(landmark[1], 0.9999))
    return int(x * width), int(y * height)


def overlay_status_text(
    frame,
    text: Union[str, Iterable[str]],
    origin: Point = (20, 30),
    color: Color = (255, 255, 255),
) -> None:
    """Write one or more status lines onto the frame."""
    if isinstance(text, str):
        lines = [text]
    else:
        lines = list(text)

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 25

    for i, line in enumerate(lines):
        position = (origin[0], origin[1] + i * line_height)
        cv2.putText(frame, line, position, font, 0.7, color, 2, cv2.LINE_AA)

