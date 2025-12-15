"""
Webcam capture helper used by the main application loop.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2


class Camera:
    """Simple wrapper around ``cv2.VideoCapture``."""

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise RuntimeError("Failed to open the default camera.")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_FPS, fps)

    def get_frame(self) -> Optional[Any]:
        """Read a single frame from the webcam; return ``None`` on failure."""
        if not self._capture.isOpened():
            return None

        success, frame = self._capture.read()
        if not success:
            return None
        return frame

    def release(self) -> None:
        """Release the underlying capture device."""
        if self._capture is not None:
            self._capture.release()

