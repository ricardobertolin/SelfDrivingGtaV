"""
Self-Driving GTA V  —  entry point.

Run with GTA V in windowed mode at 800x600 anywhere on your screen.
The window position is detected automatically. Press Q to stop cleanly.
"""

import ctypes
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    CENTER_LINE_COLOR, HUD_COLOR, KEY_A, KEY_D, KEY_S, KEY_W,
    KILL_VK, LANE_COLOR, MIN_LINES_FOR_STEERING, ROI_BOTTOM_Y, ROI_TOP_Y,
    SHOW_HUD, SMOOTHING_WINDOW,
)
from controller import KeyController
from lane_detection import DetectionResult, LaneDetector
from screen_capture import find_gta_window, grab_screen
from steering import SteeringController

Lane = Tuple[int, int, int, int]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_vk_down(vk: int) -> bool:
    """True if the given Windows Virtual Key is currently held."""
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def draw_lanes_on_frame(
    frame: np.ndarray,
    left:  Optional[Lane],
    right: Optional[Lane],
) -> None:
    if left:
        cv2.line(frame, (left[0],  left[1]),  (left[2],  left[3]),  LANE_COLOR, 5)
    if right:
        cv2.line(frame, (right[0], right[1]), (right[2], right[3]), LANE_COLOR, 5)
    if left and right:
        cx_top = (left[0] + right[0]) // 2
        cx_bot = (left[2] + right[2]) // 2
        cv2.line(frame, (cx_top, ROI_TOP_Y), (cx_bot, ROI_BOTTOM_Y),
                 CENTER_LINE_COLOR, 3)


def draw_hud(
    frame:    np.ndarray,
    offset:   float,
    action:   str,
    throttle: str,
    fps:      float,
    fill:     int,
) -> None:
    lines = [
        f"FPS:    {fps:5.1f}",
        f"Offset: {offset:+.0f} px",
        f"Steer:  {action.upper()}",
        f"Speed:  {throttle.upper()}",
        f"Buffer: {fill}/{SMOOTHING_WINDOW}",
    ]
    # Semi-transparent dark background strip
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (210, 146), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    y = 25
    for text in lines:
        cv2.putText(frame, text, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, HUD_COLOR, 2, cv2.LINE_AA)
        y += 26


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    detector = LaneDetector()
    steering = SteeringController()
    keys     = KeyController()

    fps      = 0.0
    offset   = 0.0
    action   = 'straight'
    throttle = 'accelerate'

    print("Self-Driving GTA V")

    # Locate GTA V now so a clear error is shown before the countdown.
    bbox = find_gta_window()
    print(f"GTA V window found — capture region: {bbox}")

    print("Switching to game in 3 seconds … alt-tab into GTA V now.")
    time.sleep(3)
    print("Running. Press Q to stop.")

    last_time = time.time()

    try:
        while True:
            # --- Kill switch (global Q key) ---
            if is_vk_down(KILL_VK):
                print("Kill switch activated.")
                break

            # --- Capture & detect ---
            frame  = grab_screen()
            result: DetectionResult = detector.detect(frame)

            # --- Decide steering and speed ---
            if (result.line_count >= MIN_LINES_FOR_STEERING
                    and result.left_lane is not None
                    and result.right_lane is not None):
                offset   = steering.calculate_offset(result.left_lane, result.right_lane)
                action   = steering.decide_action(offset)
                throttle = steering.decide_speed(offset)
            else:
                # Not enough confidence — coast with no lateral input
                action   = 'straight'
                throttle = 'coast'

            # --- Lateral keys ---
            if action == 'left':
                keys.press(KEY_A)
                keys.release(KEY_D)
            elif action == 'right':
                keys.press(KEY_D)
                keys.release(KEY_A)
            else:
                keys.release(KEY_A)
                keys.release(KEY_D)

            # --- Throttle / brake keys ---
            if throttle == 'accelerate':
                keys.press(KEY_W)
                keys.release(KEY_S)
            elif throttle == 'brake':
                keys.press(KEY_S)
                keys.release(KEY_W)
            else:  # coast
                keys.release(KEY_W)
                keys.release(KEY_S)

            # --- FPS ---
            now      = time.time()
            fps      = 1.0 / max(now - last_time, 1e-9)
            last_time = now

            # --- Visualise ---
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            draw_lanes_on_frame(display, result.left_lane, result.right_lane)
            if SHOW_HUD:
                draw_hud(display, offset, action, throttle, fps, result.buffer_fill)

            cv2.imshow('Self-Driving GTA V', display)

            # Q inside the OpenCV window also quits
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as exc:
        print(f"Unhandled exception: {exc}")
        raise
    finally:
        keys.release_all()
        cv2.destroyAllWindows()
        print("All keys released. Exited cleanly.")


if __name__ == '__main__':
    main()
