"""
Abstraction layer for OS mouse control.

For now the methods only log the requested actions. We'll integrate a
real backend (e.g., ``pyautogui``) in a later iteration.
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

try:
    import pyautogui

    pyautogui.FAILSAFE = False
except ImportError:  # pragma: no cover - fallback for environments without UI access
    pyautogui = None


SLOW_THRESHOLD = 12
FAST_THRESHOLD = 40


@dataclass
class MouseController:
    """Mouse controller that smooths cursors and triggers real OS actions."""

    def __post_init__(self) -> None:
        # Keep a longer history to support both light and heavy smoothing.
        self._history: Deque[Tuple[int, int]] = deque(maxlen=8)
        self._last_left_click_time = 0.0
        self._last_right_click_time = 0.0
        self._dragging = False
        self._button_down: Optional[str] = None
        self._prev_point: Optional[Tuple[int, int]] = None
        self._last_output: Optional[Tuple[float, float]] = None
        self._velocity: Tuple[float, float] = (0.0, 0.0)
        self._last_time: float = time.time()

    def smooth(self, x: int, y: int) -> Tuple[int, int]:
        """
        Return a smoothed cursor coordinate using exponential moving average
        with velocity-based prediction for smoother bridging between frames.
        """
        current_time = time.time()
        dt = max(0.001, current_time - self._last_time)  # Prevent division by zero
        self._last_time = current_time
        
        self._history.append((x, y))
        
        # Calculate velocity from recent movement
        if self._prev_point is not None:
            vx = (x - self._prev_point[0]) / dt
            vy = (y - self._prev_point[1]) / dt
            # Smooth velocity using exponential moving average
            alpha_vel = 0.3  # Velocity smoothing factor
            self._velocity = (
                self._velocity[0] * (1 - alpha_vel) + vx * alpha_vel,
                self._velocity[1] * (1 - alpha_vel) + vy * alpha_vel
            )
        else:
            self._velocity = (0.0, 0.0)
        
        # Calculate speed for adaptive smoothing
        speed = math.hypot(self._velocity[0], self._velocity[1])
        
        # Use exponential moving average with adaptive alpha based on speed
        if self._last_output is None:
            # First frame - no smoothing
            self._last_output = (float(x), float(y))
            self._prev_point = (x, y)
            return x, y
        
        # Adaptive smoothing: more smoothing for slow movements (jitter reduction),
        # less smoothing for fast movements (responsiveness)
        if speed < SLOW_THRESHOLD:
            # Slow movement: heavy smoothing to reduce jitter
            alpha = 0.15
        elif speed < FAST_THRESHOLD:
            # Medium movement: moderate smoothing
            alpha = 0.25
        else:
            # Fast movement: light smoothing for responsiveness
            alpha = 0.4
        
        # Exponential moving average
        smoothed_x = self._last_output[0] + alpha * (x - self._last_output[0])
        smoothed_y = self._last_output[1] + alpha * (y - self._last_output[1])
        
        # Velocity-based prediction: predict where cursor should be based on velocity
        # This bridges the gap between frame calculations
        if speed > 5.0:  # Only predict if there's meaningful movement
            # Predict next position based on current velocity
            predicted_x = smoothed_x + self._velocity[0] * dt * 0.5
            predicted_y = smoothed_y + self._velocity[1] * dt * 0.5
            
            # Blend predicted position with smoothed position
            # More prediction for fast movements, less for slow
            prediction_weight = min(0.3, speed / 100.0)
            smoothed_x = smoothed_x * (1 - prediction_weight) + predicted_x * prediction_weight
            smoothed_y = smoothed_y * (1 - prediction_weight) + predicted_y * prediction_weight
        
        self._prev_point = (x, y)
        self._last_output = (smoothed_x, smoothed_y)
        return int(smoothed_x), int(smoothed_y)

    def move_cursor(self, x: int, y: int) -> None:
        if pyautogui is None:
            print(f"[MouseController] Move cursor to ({x}, {y})")
            return
        pyautogui.moveTo(x, y, duration=0)

    def can_click(self, last_time: float, cooldown: float = 0.20) -> bool:
        """Debounce helper to avoid click spam."""
        return (time.time() - last_time) > cooldown

    def left_click(self) -> None:
        if not self.can_click(self._last_left_click_time) or pyautogui is None:
            return
        pyautogui.click(button="left")
        self._last_left_click_time = time.time()

    def right_click(self) -> None:
        if not self.can_click(self._last_right_click_time) or pyautogui is None:
            return
        pyautogui.click(button="right")
        self._last_right_click_time = time.time()

    def mouse_down(self, button: str = "left") -> None:
        if self._button_down == button:
            return
        if pyautogui is None:
            print(f"[MouseController] Mouse down ({button})")
        else:
            pyautogui.mouseDown(button=button)
        self._button_down = button

    def mouse_up(self, button: str = "left") -> None:
        if self._button_down is None:
            return
        if pyautogui is None:
            print(f"[MouseController] Mouse up ({button})")
        else:
            pyautogui.mouseUp(button=button)
        if self._button_down == button:
            self._button_down = None

    def current_button(self) -> Optional[str]:
        return self._button_down

    def start_drag(self) -> None:
        if self._dragging:
            return
        self.mouse_down()
        self._dragging = True

    def end_drag(self) -> None:
        if not self._dragging:
            return
        self._dragging = False
        button = self._button_down or "left"
        self.mouse_up(button=button)

    def is_dragging(self) -> bool:
        return self._dragging

