"""
Self-Driving GTA V  —  entry point.

GTA V must be in Windowed mode at 800×600.  The window is moved to
(GTA_WINDOW_X, GTA_WINDOW_Y) at startup; the debug overlay appears at
(OVERLAY_WINDOW_X, OVERLAY_WINDOW_Y) so the two windows never overlap.

Global keys (active even when GTA V has focus):
  Q  — kill switch
  C  — toggle AI controller     (full AI ↔ full manual)
  L  — toggle camera-lock (car-body centroid correction)
"""

import ctypes
import math
import time
from typing import Optional

import cv2
import numpy as np

from camera_controller import CameraController
from config import (
    AI_CTRL_VK, CAMERA_LOCK_VK,
    CENTER_LINE_COLOR, DANGER_ZONE, GTA_WINDOW_X, GTA_WINDOW_Y,
    HUD_COLOR, KEY_A, KEY_D, KEY_E, KEY_S, KEY_W, KILL_VK,
    LANE_COLOR, LANE_STALE_COLOR, MIN_LINES_FOR_STEERING,
    ACCEL_HOLD_FRAMES, ACCEL_SKIP_FRAMES,
    BRAKE_HOLD_FRAMES, BRAKE_SKIP_FRAMES,
    ML_CTRL_VK, ML_MODEL_PATH, ML_OFFSET_MODEL_PATH, ML_THRESHOLD,
    OBSTACLE_DETECTION_ENABLED,
    OBSTACLE_ZONE_COLOR_BLOCKED, OBSTACLE_ZONE_COLOR_CLEAR,
    OVERLAY_WINDOW_X, OVERLAY_WINDOW_Y,
    HONK_VK, RECORD_VK, REVERSE_ENABLED,
    SHOW_CAMERA_OVERLAY, SHOW_HUD, SHOW_OBSTACLE_ZONE, SMOOTHING_WINDOW,
    STEER_HOLD_FRAMES, STEER_SKIP_FRAMES,
)
from data_collector import DataCollector

# ML components are optional — require `pip install torch`
try:
    import cv2 as _cv2_ml  # already imported below; just check torch
    import torch as _torch_check  # noqa: F401
    from model import DrivingCNN, OffsetCNN, INPUT_H, INPUT_W
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False
    DrivingCNN = OffsetCNN = None   # type: ignore[assignment,misc]
    INPUT_W = INPUT_H = 0
    print("Warning: PyTorch not found — ML mode disabled.  "
          "Run:  pip install torch")
from controller import KeyController
from lane_detection import DetectionResult, LaneDetector
from obstacle_detection import ObstacleDetector
from screen_capture import find_gta_window, get_gta_window_pos, grab_screen, move_gta_window
from steering import SteeringController

_AI_ON_COLOR    = (0, 255, 100)
_CAM_LOCK_COLOR = (0, 200, 255)
_OVERLAY_TITLE  = 'Self-Driving GTA V'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_vk_down(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def _rising_edge(cur: bool, prev: bool) -> bool:
    return cur and not prev


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_lanes_on_frame(
    frame:       np.ndarray,
    left:        Optional[np.ndarray],
    right:       Optional[np.ndarray],
    left_fresh:  bool,
    right_fresh: bool,
) -> None:
    def _draw_curve(pts: np.ndarray, color: tuple, thickness: int) -> None:
        cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, color, thickness)

    if left is not None:
        _draw_curve(left,  LANE_COLOR if left_fresh  else LANE_STALE_COLOR, 5)
    if right is not None:
        _draw_curve(right, LANE_COLOR if right_fresh else LANE_STALE_COLOR, 5)

    if left is not None and right is not None:
        centre = ((left.astype(np.float32) + right.astype(np.float32)) / 2).astype(np.int32)
        _draw_curve(centre, CENTER_LINE_COLOR, 3)


def draw_danger_zone(frame: np.ndarray, blocked: bool) -> None:
    x1, y1, x2, y2 = DANGER_ZONE
    if blocked:
        # Semi-transparent red fill over the danger zone
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 200), -1)
        cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)
        # Thick red border
        cv2.rectangle(frame, (x1, y1), (x2, y2), OBSTACLE_ZONE_COLOR_BLOCKED, 3)
        # Large warning text centred above the zone
        label  = '!! OBSTACLE !!'
        font   = cv2.FONT_HERSHEY_SIMPLEX
        scale  = 0.9
        thick  = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        tx = (x1 + x2 - tw) // 2
        ty = max(y1 - 10, th + 4)
        cv2.putText(frame, label, (tx, ty), font, scale,
                    OBSTACLE_ZONE_COLOR_BLOCKED, thick + 1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (x1, y1), (x2, y2), OBSTACLE_ZONE_COLOR_CLEAR, 2)


def draw_heading_arrow(frame: np.ndarray, heading_deg: float) -> None:
    h, w = frame.shape[:2]
    base = (w // 2, h - 50)
    rad  = math.radians(heading_deg)
    tip  = (int(base[0] + 90 * math.sin(rad)), int(base[1] - 90 * math.cos(rad)))
    cv2.arrowedLine(frame, base, tip, HUD_COLOR, 3, tipLength=0.28)
    lx = tip[0] + (8 if heading_deg >= 0 else -80)
    cv2.putText(frame, f"{heading_deg:+.1f}d", (lx, tip[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, HUD_COLOR, 1, cv2.LINE_AA)


_REC_COLOR = (0, 0, 220)    # red  — recording indicator
_ML_COLOR  = (220, 0, 220)  # magenta — ML mode indicator


def draw_hud(
    frame:          np.ndarray,
    offset:         float,
    action:         str,
    throttle:       str,
    ai_enabled:     bool,
    camera_lock:    bool,
    obstacle_ahead: bool,
    heading_deg:    Optional[float],
    fps:            float,
    fill:           int,
    cam_x_err:      float = 0.0,
    cam_y_err:      float = 0.0,
    recording:      bool  = False,
    record_count:   int   = 0,
    ml_enabled:     bool  = False,
) -> None:
    h_text  = f"{heading_deg:+.1f} deg" if heading_deg is not None else "--"
    ob_text = ("BLOCKED" if obstacle_ahead else "clear") if OBSTACLE_DETECTION_ENABLED else "OFF"
    rec_text = f"REC  {record_count} fr" if recording else "off"
    ml_text  = "ON" if ml_enabled else "off"
    lines = [
        (f"FPS:        {fps:5.1f}",                             HUD_COLOR),
        (f"Offset:     {offset:+.0f} px",                       HUD_COLOR),
        (f"Heading:    {h_text}",                                HUD_COLOR),
        (f"Steer:      {action.upper()}",                        HUD_COLOR),
        (f"Speed:      {throttle.upper()}",                      HUD_COLOR),
        (f"Controller: {'ON' if ai_enabled else 'OFF'}",         _AI_ON_COLOR if ai_enabled else HUD_COLOR),
        (f"ML model:   {ml_text}",                               _ML_COLOR if ml_enabled else HUD_COLOR),
        (f"Camera:     {'LOCK' if camera_lock else 'free'}",     _CAM_LOCK_COLOR if camera_lock else HUD_COLOR),
        (f"Cam err:    x={cam_x_err:+.0f} y={cam_y_err:+.0f}",  _CAM_LOCK_COLOR if camera_lock else HUD_COLOR),
        (f"Obstacle:   {ob_text}",                               OBSTACLE_ZONE_COLOR_BLOCKED if obstacle_ahead else HUD_COLOR),
        (f"Buffer:     {fill}/{SMOOTHING_WINDOW}",               HUD_COLOR),
        (f"Record:     {rec_text}",                              _REC_COLOR if recording else HUD_COLOR),
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (245, 338), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    y = 25
    for text, color in lines:
        cv2.putText(frame, text, (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        y += 26

    # Prominent recording dot in the top-right corner
    if recording:
        w_f = frame.shape[1]
        cv2.circle(frame, (w_f - 20, 20), 10, _REC_COLOR, -1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    detector  = LaneDetector()
    obstacle  = ObstacleDetector()
    steering  = SteeringController()
    keys      = KeyController()
    cam_ctrl  = CameraController()
    collector = DataCollector()

    # Load ML models — offset regression is preferred; keys BC is the fallback.
    offset_model = OffsetCNN.load(ML_OFFSET_MODEL_PATH) if _ML_AVAILABLE else None
    ml_model     = DrivingCNN.load(ML_MODEL_PATH)       if _ML_AVAILABLE else None

    fps            = 0.0
    offset         = 0.0
    action         = 'straight'
    throttle       = 'coast'
    ai_enabled     = False
    camera_lock    = False
    obstacle_ahead = False
    recording      = False
    ml_enabled     = False
    cam_x_err      = 0.0
    cam_y_err      = 0.0
    centroid       = None
    contour        = None
    result         = DetectionResult(None, None, False, False, 0, 0, None)
    _ai_prev        = False
    _cam_prev       = False
    _rec_prev       = False
    _ml_prev        = False
    _overlay_placed = False
    _steer_tick     = 0
    _brake_tick     = 0
    _accel_tick     = 0

    print("Self-Driving GTA V")
    print("  C = lane-AI on/off   M = ML model on/off")
    print("  R = record (manual play only)   L = camera lock   Q = quit")

    find_gta_window()
    _original_gta_pos = get_gta_window_pos()

    print(f"Moving GTA V window to ({GTA_WINDOW_X}, {GTA_WINDOW_Y}) …")
    move_gta_window(GTA_WINDOW_X, GTA_WINDOW_Y)
    bbox = find_gta_window()
    print(f"Capture region: {bbox}")
    _gta_cx = (bbox[0] + bbox[2]) // 2
    _gta_cy = (bbox[1] + bbox[3]) // 2

    print(f"Debug overlay will appear at ({OVERLAY_WINDOW_X}, {OVERLAY_WINDOW_Y}).")
    print("Switching in 3 s … alt-tab into GTA V now.")
    time.sleep(3)

    last_time = time.time()

    try:
        while True:
            # ---- Kill switch ----
            if is_vk_down(KILL_VK):
                print("Kill switch activated.")
                break

            # ---- Toggle AI / lane-detection control (C) ----
            ai_now = is_vk_down(AI_CTRL_VK)
            if _rising_edge(ai_now, _ai_prev):
                ai_enabled = not ai_enabled
                if ai_enabled:
                    ml_enabled = False   # mutual exclusion
                print(f"Lane-AI: {'ON' if ai_enabled else 'OFF'}")
            _ai_prev = ai_now

            # ---- Toggle ML model control (M) ----
            ml_now = is_vk_down(ML_CTRL_VK)
            if _rising_edge(ml_now, _ml_prev):
                if not _ML_AVAILABLE:
                    print("ML mode unavailable — install PyTorch first.")
                elif ml_model is None:
                    print("No trained model found — run train_model.py first.")
                else:
                    ml_enabled = not ml_enabled
                    if ml_enabled:
                        ai_enabled = False   # mutual exclusion
                    print(f"ML model: {'ON' if ml_enabled else 'OFF'}")
            _ml_prev = ml_now

            # ---- Toggle recording (R) — only allowed during manual play ----
            rec_now = is_vk_down(RECORD_VK)
            if _rising_edge(rec_now, _rec_prev):
                if ai_enabled or ml_enabled:
                    print("Recording blocked while AI/ML is active — drive manually.")
                elif not recording:
                    recording = True
                    collector.start()
                else:
                    recording = False
                    collector.stop()
            _rec_prev = rec_now

            # ---- Obstacle capture on honk (E) ----
            # While E is held the user is confirming an obstacle is present.
            # Each frame is saved to training_data/obstacles/ with label=1
            # so those scenes can later train an obstacle classifier.
            # This is separate from the R-key offset recording.
            if is_vk_down(HONK_VK) and not (ai_enabled or ml_enabled):
                collector.record_obstacle(frame)

            # ---- Toggle camera lock (L) ----
            cam_now = is_vk_down(CAMERA_LOCK_VK)
            if _rising_edge(cam_now, _cam_prev):
                camera_lock = not camera_lock
                if camera_lock:
                    cam_ctrl.enable(_gta_cx, _gta_cy)
                else:
                    cam_ctrl.disable()
                print(f"Camera lock: {'ON' if camera_lock else 'OFF'}")
            _cam_prev = cam_now

            # ---- Capture frame ----
            frame = grab_screen()
            obstacle_ahead = obstacle.detect(frame) if OBSTACLE_DETECTION_ENABLED else False

            # ---- Record training data ----
            # Always runs the lane detector so we have fresh offset labels.
            # Offset samples are only saved when both lanes are freshly detected
            # (not extrapolated from the buffer) to keep labels reliable.
            if recording:
                result = detector.detect(frame)
                if (result.left_fresh and result.right_fresh
                        and result.left_lane is not None
                        and result.right_lane is not None):
                    rec_offset = steering.calculate_offset(result.left_lane,
                                                           result.right_lane)
                    collector.record_offset(frame, rec_offset)

            # ---- Decide action -----------------------------------------------
            if ml_enabled:
                small = cv2.resize(frame, (INPUT_W, INPUT_H),
                                   interpolation=cv2.INTER_AREA)

                if offset_model is not None:
                    # Offset regression: predict lane offset → existing steering logic
                    offset       = offset_model.predict_offset(small)
                    action       = steering.decide_action(offset)
                    speed_advice = steering.decide_speed(offset)
                elif ml_model is not None:
                    # Fallback: behavioural-cloning key prediction
                    pred         = ml_model.predict(small, ML_THRESHOLD)
                    action       = ('left'  if pred['A'] else
                                    'right' if pred['D'] else 'straight')
                    speed_advice = 'accelerate' if pred['W'] else 'coast'
                    offset       = 0.0
                else:
                    action = 'straight'; speed_advice = 'coast'; offset = 0.0

                throttle = 'brake' if obstacle_ahead else speed_advice
                if not recording:
                    result = detector.detect(frame)

            else:
                # Lane-detection AI
                result = detector.detect(frame)
                if (result.line_count >= MIN_LINES_FOR_STEERING
                        and result.left_lane is not None
                        and result.right_lane is not None):
                    offset       = steering.calculate_offset(result.left_lane,
                                                             result.right_lane)
                    action       = steering.decide_action(offset)
                    speed_advice = steering.decide_speed(offset)
                else:
                    action       = 'straight'
                    speed_advice = 'coast'

                throttle = 'brake' if obstacle_ahead else speed_advice

            # ---- Apply keys ----
            if ai_enabled or ml_enabled:
                _steer_tick += 1
                _period   = STEER_HOLD_FRAMES + max(STEER_SKIP_FRAMES, 0)
                _steer_on = (_period == 0) or ((_steer_tick % _period) < STEER_HOLD_FRAMES)

                if action == 'left' and _steer_on:
                    keys.press(KEY_A);   keys.release(KEY_D)
                elif action == 'right' and _steer_on:
                    keys.press(KEY_D);   keys.release(KEY_A)
                else:
                    keys.release(KEY_A); keys.release(KEY_D)

                if throttle == 'accelerate':
                    _accel_tick += 1
                    _ap = ACCEL_HOLD_FRAMES + max(ACCEL_SKIP_FRAMES, 0)
                    if (_ap == 0) or ((_accel_tick % _ap) < ACCEL_HOLD_FRAMES):
                        keys.press(KEY_W)
                    else:
                        keys.release(KEY_W)
                    keys.release(KEY_S)
                    _brake_tick = 0
                elif throttle == 'brake':
                    # Alternating S / W: hold brake, then briefly accelerate to
                    # prevent the car from stopping and engaging reverse.
                    _brake_tick += 1
                    _bp      = BRAKE_HOLD_FRAMES + max(BRAKE_SKIP_FRAMES, 0)
                    _in_hold = (_bp > 0) and ((_brake_tick % _bp) < BRAKE_HOLD_FRAMES)
                    if REVERSE_ENABLED:
                        if _in_hold:
                            keys.press(KEY_S); keys.release(KEY_W)
                        else:
                            keys.release(KEY_S); keys.release(KEY_W)
                    else:
                        if _in_hold:
                            keys.press(KEY_S); keys.release(KEY_W)
                        else:
                            keys.release(KEY_S); keys.press(KEY_W)
                else:
                    keys.release(KEY_W); keys.release(KEY_S)
                    _brake_tick = 0
                    _accel_tick = 0

                # Auto-honk when obstacle detected
                if obstacle_ahead:
                    keys.press(KEY_E)
                else:
                    keys.release(KEY_E)
            else:
                keys.release(KEY_A); keys.release(KEY_D)
                keys.release(KEY_W); keys.release(KEY_S)
                keys.release(KEY_E)

            # ---- Camera alignment ----
            cam_x_err, cam_y_err, centroid, contour = cam_ctrl.update(frame)

            # ---- FPS ----
            now       = time.time()
            fps       = 1.0 / max(now - last_time, 1e-9)
            last_time = now

            # ---- Visualise ----
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if OBSTACLE_DETECTION_ENABLED and SHOW_OBSTACLE_ZONE:
                draw_danger_zone(display, obstacle_ahead)
            draw_lanes_on_frame(
                display,
                result.left_lane,  result.right_lane,
                result.left_fresh, result.right_fresh,
            )
            if result.heading_deg is not None:
                draw_heading_arrow(display, result.heading_deg)
            if SHOW_CAMERA_OVERLAY:
                cam_ctrl.draw(display, centroid, contour, cam_x_err, cam_y_err)
            if SHOW_HUD:
                draw_hud(display, offset, action, throttle,
                         ai_enabled, camera_lock, obstacle_ahead,
                         result.heading_deg, fps, result.buffer_fill,
                         cam_x_err, cam_y_err,
                         recording=recording,
                         record_count=collector.frame_count,
                         ml_enabled=ml_enabled)

            cv2.imshow(_OVERLAY_TITLE, display)

            if not _overlay_placed:
                cv2.moveWindow(_OVERLAY_TITLE, OVERLAY_WINDOW_X, OVERLAY_WINDOW_Y)
                _overlay_placed = True

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as exc:
        print(f"Unhandled exception: {exc}")
        raise
    finally:
        if recording:
            collector.stop()
        cam_ctrl.disable()
        keys.release_all()
        cv2.destroyAllWindows()
        if _original_gta_pos is not None:
            move_gta_window(*_original_gta_pos)
            print(f"GTA V window restored to {_original_gta_pos}.")
        print("All keys released. Exited cleanly.")


if __name__ == '__main__':
    main()
