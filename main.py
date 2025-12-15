"""
Entry point for the air touchscreen prototype.

This first iteration wires together the camera feed, MediaPipe-based
hand tracking, gesture recognition stubs, and simple visualization
helpers. It shows a live preview window that can be exited with the
'q' key.
"""

from __future__ import annotations

import ctypes
from typing import List, Optional, Tuple

import cv2

from control.mouse_controller import MouseController
from gestures.recognizer import GestureResult, GestureType, recognize_gestures
from vision.camera import Camera
from vision.hand_tracker import HandData, HandTracker, correct_fingertip
from vision.visualization import (
    draw_hand_landmarks,
    draw_interaction_box,
    draw_skeleton,
    overlay_status_text,
)

Point = Tuple[int, int]
PINCH_CONFIDENCE_CLICK_THRESHOLD = 0.5
# Scale factor for the interaction zone relative to the camera frame.
# Values < 1.0 shrink the zone so fingers stay in view near frame edges.
INTERACTION_SCALE = 0.608  # further 5% shrink from 0.64


def inside_box(x: int, y: int, top_left: Point, bottom_right: Point) -> bool:
    return top_left[0] <= x <= bottom_right[0] and top_left[1] <= y <= bottom_right[1]


def inside_box_margin(
    x: int,
    y: int,
    top_left: Point,
    bottom_right: Point,
    margin: int = 25,
) -> bool:
    return (
        (top_left[0] - margin) <= x <= (bottom_right[0] + margin)
        and (top_left[1] - margin) <= y <= (bottom_right[1] + margin)
    )


def normalized_to_pixel(
    landmark: Tuple[float, float, float], width: int, height: int
) -> Point:
    x = max(0.0, min(landmark[0], 0.9999))
    y = max(0.0, min(landmark[1], 0.9999))
    return int(x * width), int(y * height)


def _select_cursor_hand(
    positions: List[Optional[Point]],
    top_left: Point,
    bottom_right: Point,
    margin: int = 25,
) -> Optional[int]:
    best_idx: Optional[int] = None
    best_depth = -9999
    fallback_idx: Optional[int] = None

    for idx, position in enumerate(positions):
        if position is None:
            continue
        if fallback_idx is None:
            fallback_idx = idx
        x, y = position
        if inside_box_margin(x, y, top_left, bottom_right, margin):
            depth = bottom_right[1] - y
            if depth > best_depth:
                best_depth = depth
                best_idx = idx

    if best_idx is not None:
        return best_idx

    return fallback_idx


def map_interaction_zone_to_screen(
    normalized_pos: Tuple[float, float, float],
    frame_width: int,
    frame_height: int,
    interaction_top_left: Point,
    interaction_bottom_right: Point,
    screen_width: int,
    screen_height: int,
) -> Tuple[int, int]:
    """
    Map normalized coordinates from the interaction zone to full screen coordinates.
    This allows users to reach all edges of the screen even if the webcam frame is smaller.
    """
    # Convert normalized coordinates (0-1) to frame pixel coordinates
    frame_x = normalized_pos[0] * frame_width
    frame_y = normalized_pos[1] * frame_height
    
    # Get interaction zone dimensions
    zone_left, zone_top = interaction_top_left
    zone_right, zone_bottom = interaction_bottom_right
    zone_width = zone_right - zone_left
    zone_height = zone_bottom - zone_top
    
    # Clamp frame coordinates to interaction zone bounds
    clamped_x = max(zone_left, min(zone_right, frame_x))
    clamped_y = max(zone_top, min(zone_bottom, frame_y))
    
    # Calculate relative position within interaction zone (0.0 to 1.0)
    if zone_width > 0:
        relative_x = (clamped_x - zone_left) / zone_width
    else:
        relative_x = 0.5
    
    if zone_height > 0:
        relative_y = (clamped_y - zone_top) / zone_height
    else:
        relative_y = 0.5
    
    # Map relative position to full screen coordinates
    screen_x = int(relative_x * screen_width)
    screen_y = int(relative_y * screen_height)
    
    # Clamp to screen bounds
    screen_x = max(0, min(screen_width - 1, screen_x))
    screen_y = max(0, min(screen_height - 1, screen_y))
    
    return screen_x, screen_y


def main() -> None:
    """Run the capture/processing/render loop."""
    camera = Camera()
    hand_tracker = HandTracker()
    mouse_controller = MouseController()
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

    interaction_margin = 80
    interaction_enabled = True

    frame_idx = 0
    last_status_lines: Optional[List[str]] = None
    prev_left_active = False
    prev_right_active = False
    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                # Camera may still be warming up; skip this iteration.
                continue
            # Mirror the frame so on-screen motion matches user perspective
            frame = cv2.flip(frame, 1)

            hands: List[HandData] = hand_tracker.process_frame(frame)

            frame_height, frame_width = frame.shape[:2]

            # Build a scaled interaction box to keep fingers in frame near edges
            box_width = int(frame_width * INTERACTION_SCALE)
            box_height = int(frame_height * INTERACTION_SCALE)
            center_x = frame_width // 2
            center_y = frame_height // 2
            top_left = (
                max(0, center_x - box_width // 2),
                max(0, center_y - box_height // 2),
            )
            bottom_right = (
                min(frame_width, center_x + box_width // 2),
                min(frame_height, center_y + box_height // 2),
            )

            pointer_index = 8
            middle_index = 12
            index_positions: List[Optional[Point]] = []
            index_corrected_norms: List[Optional[Tuple[float, float, float]]] = []
            index_inside_flags: List[bool] = []
            middle_inside_flags: List[bool] = []
            palm_centers: List[Optional[Tuple[float, float, float]]] = []
            palm_positions: List[Optional[Point]] = []

            for hand in hands:
                index_pointer: Optional[Point] = None
                index_corrected: Optional[Tuple[float, float, float]] = None
                palm_center: Optional[Tuple[float, float, float]] = None
                palm_pos: Optional[Point] = None
                palm_inside = False

                # Calculate palm center from wrist and MCP joints
                if len(hand.landmarks) > 9:
                    # Use wrist (0) and middle MCP (9) for palm center
                    wrist = hand.landmarks[0]
                    middle_mcp = hand.landmarks[9]
                    # Palm center is midpoint between wrist and middle MCP
                    palm_center = (
                        (wrist[0] + middle_mcp[0]) / 2.0,
                        (wrist[1] + middle_mcp[1]) / 2.0,
                        (wrist[2] + middle_mcp[2]) / 2.0,
                    )
                    palm_x = int(palm_center[0] * frame_width)
                    palm_y = int(palm_center[1] * frame_height)
                    palm_pos = (palm_x, palm_y)
                    palm_inside = inside_box_margin(
                        palm_x, palm_y, top_left, bottom_right, margin=25
                    )

                if len(hand.landmarks) > pointer_index:
                    px, py, _ = hand.landmarks[pointer_index]
                    ix = int(px * frame_width)
                    iy = int(py * frame_height)
                    index_pointer = (ix, iy)
                    index_inside_flags.append(
                        inside_box_margin(ix, iy, top_left, bottom_right, margin=25)
                        or palm_inside
                    )
                    index_corrected = correct_fingertip(
                        hand.landmarks[7], hand.landmarks[pointer_index], 1.55
                    )
                else:
                    index_inside_flags.append(palm_inside)

                index_positions.append(index_pointer)
                index_corrected_norms.append(index_corrected)
                palm_centers.append(palm_center)
                palm_positions.append(palm_pos)

                if len(hand.landmarks) > middle_index:
                    mx, my, _ = hand.landmarks[middle_index]
                    middle_x = int(mx * frame_width)
                    middle_y = int(my * frame_height)
                    middle_inside_flags.append(
                        inside_box_margin(middle_x, middle_y, top_left, bottom_right, margin=25)
                        or palm_inside
                    )
                else:
                    middle_inside_flags.append(palm_inside)

            # Select cursor hand based on palm center position
            cursor_hand_idx = _select_cursor_hand(
                palm_positions, top_left, bottom_right, margin=25
            )
            if cursor_hand_idx is None and hands:
                cursor_hand_idx = 0
            cursor_hand_label = (
                hands[cursor_hand_idx].handedness if cursor_hand_idx is not None else "None"
            )

            gesture_result: GestureResult = recognize_gestures(
                hands,
                index_inside_flags=index_inside_flags,
                middle_inside_flags=middle_inside_flags,
            )
            gesture = gesture_result.gesture
            gesture_details = gesture_result.details
            active = bool(gesture_details.get("active", False))
            conf_raw = gesture_details.get("pinch_confidence", 0.0)
            try:
                pinch_confidence_value = float(conf_raw)
            except (TypeError, ValueError):
                pinch_confidence_value = 0.0
            is_left = gesture == GestureType.LEFT_CLICK
            is_right = gesture == GestureType.RIGHT_CLICK
            gesture_hand_idx = int(gesture_details.get("hand_index", -1))
            gesture_hand_label = (
                hands[gesture_hand_idx].handedness
                if 0 <= gesture_hand_idx < len(hands)
                else "None"
            )
            mode_toggle = gesture_result.details.get("mode_toggle")
            if mode_toggle == "ENABLE":
                interaction_enabled = True
            elif mode_toggle == "DISABLE":
                interaction_enabled = False
                if mouse_controller.is_dragging():
                    mouse_controller.end_drag()

            zone_text = "Outside Zone"

            if (
                cursor_hand_idx is not None
                and cursor_hand_idx < len(hands)
                and palm_centers[cursor_hand_idx] is not None
                and palm_positions[cursor_hand_idx] is not None
            ):
                cursor_position = palm_positions[cursor_hand_idx]
                cursor_pointer_inside = inside_box_margin(
                    cursor_position[0], cursor_position[1], top_left, bottom_right, margin=25
                )

                if cursor_pointer_inside and interaction_enabled:
                    # Use palm center for cursor movement (more stable than fingertip)
                    palm_norm = palm_centers[cursor_hand_idx]
                    if palm_norm is not None:
                        # Map interaction zone to full screen for edge-to-edge control
                        screen_x, screen_y = map_interaction_zone_to_screen(
                            palm_norm,
                            frame_width,
                            frame_height,
                            top_left,
                            bottom_right,
                            screen_width,
                            screen_height,
                        )
                        smoothed_x, smoothed_y = mouse_controller.smooth(
                            screen_x, screen_y
                        )
                        mouse_controller.move_cursor(smoothed_x, smoothed_y)
                        cv2.circle(frame, cursor_position, 8, (0, 255, 0), -1)
                        zone_text = "Inside Interaction Zone"
            if not interaction_enabled:
                zone_text = "AIR INPUT DISABLED (show open palm to enable)"

            if interaction_enabled:
                # ---------- LEFT CLICK ----------
                if is_left:
                    if active and not prev_left_active:
                        mouse_controller.mouse_down(button="left")
                    if not active and prev_left_active:
                        mouse_controller.mouse_up(button="left")
                    prev_left_active = active
                else:
                    if prev_left_active:
                        mouse_controller.mouse_up(button="left")
                        prev_left_active = False

                # ---------- RIGHT CLICK ----------
                if is_right:
                    if active and not prev_right_active:
                        mouse_controller.mouse_down(button="right")
                    if not active and prev_right_active:
                        mouse_controller.mouse_up(button="right")
                    prev_right_active = active
                else:
                    if prev_right_active:
                        mouse_controller.mouse_up(button="right")
                        prev_right_active = False
            needs_release = (not interaction_enabled) or (
                not active and not (is_left or is_right)
            )
            if needs_release and mouse_controller.is_dragging():
                current_button = mouse_controller.current_button()
                mouse_controller.mouse_up(button=current_button or "left")
                mouse_controller._dragging = False

            draw_interaction_box(frame, top_left, bottom_right)

            for hand in hands:
                draw_skeleton(frame, hand)

            draw_hand_landmarks(frame, hands, inside_flags=index_inside_flags)

            def _fmt3(val: object) -> object:
                try:
                    return f"{float(val):.3f}"
                except (TypeError, ValueError):
                    return val

            gesture_display = gesture_result.gesture.value
            if active:
                if is_left:
                    gesture_display = f"{gesture_display} (left clicking)"
                elif is_right:
                    gesture_display = f"{gesture_display} (right clicking)"
                else:
                    gesture_display = f"{gesture_display} (clicking)"
            elif pinch_confidence_value > PINCH_CONFIDENCE_CLICK_THRESHOLD:
                gesture_display = f"{gesture_display} (clicking)"

            status_lines = [
                f"Hands detected: {len(hands)}",
                zone_text,
                f"Interaction Enabled: {interaction_enabled}",
                f"Cursor Hand: {cursor_hand_label}",
                f"Gesture Hand: {gesture_hand_label}",
                f"Gesture: {gesture_display}",
                f"Dragging: {mouse_controller.is_dragging()}",
            ]
            # ===== DEBUG VALUES (minimal & truncated) =====
            dbg = gesture_result.details
            status_lines.extend(
                [
                    f"combined: {_fmt3(dbg.get('combined', None))}",
                    f"tip: {_fmt3(dbg.get('tip_dist', None))} / {_fmt3(dbg.get('thr_tip', None))}",
                    f"cluster: {_fmt3(dbg.get('cluster_dist', None))} / {_fmt3(dbg.get('thr_cluster', None))}",
                    f"pinch_conf: {_fmt3(dbg.get('pinch_confidence', None))}",
                    f"active: {dbg.get('active', None)}",
                ]
            )
            # ==========================================
            status_lines.append("Press 'q' to quit")
            if frame_idx % 2 == 0 or last_status_lines is None:
                last_status_lines = status_lines
            overlay_status_text(frame, last_status_lines or status_lines)

            frame_idx += 1
            cv2.imshow("Air Touchscreen Preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        hand_tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

