"""
Camera alignment controller for the GTA V self-driving project.

Keeps the third-person camera locked to the ideal position:
  - horizontally centred behind the car  (rotation correction)
  - high enough to see the road clearly  (height correction)

Detection — rotation-robust 1-D projection method
--------------------------------------------------
Instead of taking the centroid of the single largest contour (which shifts
unpredictably whenever the car yaws and shows a different body profile) we
compute two independent 1-D weighted averages of edge energy:

  Glass / window suppression (applied first):
    Pixels where L > 195 (specular reflections on glass or chrome) or
    S < 25 (transparent glass showing road / sky, or grey road surface)
    are zeroed in the edge map.  This removes the rear-window frame — the
    most common false-positive — before any projection is computed.

  Horizontal (cx):
    Column-sum projection over the bottom 60% of CAR_ROI (chassis/body
    band, below the road horizon), then softly Gaussian-weighted toward the
    ROI centre.  The weighted average of that histogram gives the car's
    lateral centre and is stable across yaw angles.

  Vertical (cy):
    Row-sum projection over the full glass-suppressed ROI, preserving the
    same semantic as CAR_TARGET_Y (mid-body centroid when aligned).

The largest contour is still found for on-screen visualisation but is
filtered to only show shapes that span ≥ 12% of the ROI width, which
rejects the narrow rear-window rectangle.

Direction feedback
------------------
When the camera is misaligned, coloured arrows and labels are drawn near
the centre of the overlay:
  → ROTATE RIGHT   /   ← ROTATE LEFT
  ↓ PITCH DOWN     /   ↑ PITCH UP
  LOCKED           (when within deadzone on both axes)

A thin vertical guide line is also drawn inside CAR_ROI at the detected
column centre (cx) so you can see exactly which column the system locked on.

Error conventions:
  cam_x_err = cx − frame_centre_x   (+ → car right of centre → rotate cam right)
  cam_y_err = cy − CAR_TARGET_Y      (+ → car too low in frame → pitch cam down)

Correction: `mouse_event(MOUSEEVENTF_MOVE, dx, dy)` injects synthetic mouse
movement that GTA V reads as camera input.  The OS cursor is snapped to the
GTA V window centre before each correction to cancel any physical mouse input.

Tune in config.py
-----------------
  CAR_ROI            — bounding box of the detection zone
  CAR_TARGET_Y       — ideal roof centroid y
  CAM_X_GAIN         — mouse px per px of horizontal error
  CAM_Y_GAIN         — mouse px per px of vertical error (negate if inverted)
  CAM_MIN_MOVE       — minimum |pixels| sent per tick
  CAMERA_DEADZONE_PX — errors smaller than this are ignored
  CAMERA_ALERT_PX    — indicator turns red beyond this error magnitude
"""

import ctypes
import ctypes.wintypes
import math
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    CAM_MIN_MOVE, CAM_X_GAIN, CAM_Y_GAIN,
    CAMERA_ALERT_PX, CAMERA_DEADZONE_PX,
    CAR_MIN_CONTOUR_PX, CAR_ROI, CAR_TARGET_Y,
)


class CameraController:
    """One instance lives for the full duration of the script."""

    def __init__(self) -> None:
        self._anchor: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def enable(self, anchor_x: int, anchor_y: int) -> None:
        """Activate camera lock.  anchor is the GTA V window centre."""
        self._anchor = (anchor_x, anchor_y)

    def disable(self) -> None:
        """Deactivate camera lock and free the cursor."""
        self._anchor = None

    # ------------------------------------------------------------------ #
    #  Per-frame update                                                    #
    # ------------------------------------------------------------------ #

    def update(self, frame: np.ndarray) \
            -> Tuple[float, float, Optional[Tuple[float, float]], Optional[np.ndarray]]:
        """
        Detect car, compute alignment errors, send mouse correction if active.

        Returns (cam_x_err, cam_y_err, centroid, contour).
        Both errors are 0.0 when the car cannot be detected.
        contour is an (N,1,2) int32 array for drawing, or None.
        """
        centroid, contour = self._detect(frame)
        cam_x_err = 0.0
        cam_y_err = 0.0

        if centroid is not None:
            cam_x_err = centroid[0] - frame.shape[1] / 2.0
            cam_y_err = centroid[1] - CAR_TARGET_Y

        if self._anchor is not None:
            # Snap cursor to the fixed anchor point every frame so GTA V sees
            # zero player mouse input before we inject our own correction.
            ctypes.windll.user32.SetCursorPos(self._anchor[0], self._anchor[1])

            if centroid is not None:
                dx = self._scale(cam_x_err, CAM_X_GAIN)
                dy = self._scale(cam_y_err, CAM_Y_GAIN)
                if dx or dy:
                    ctypes.windll.user32.mouse_event(
                        0x0001,                  # MOUSEEVENTF_MOVE
                        ctypes.c_long(dx),
                        ctypes.c_long(dy),
                        0, 0,
                    )

        return cam_x_err, cam_y_err, centroid, contour

    # ------------------------------------------------------------------ #
    #  Debug drawing                                                       #
    # ------------------------------------------------------------------ #

    def draw(self, frame: np.ndarray,
             centroid: Optional[Tuple[float, float]],
             contour:  Optional[np.ndarray],
             cam_x_err: float, cam_y_err: float) -> None:
        """
        Overlay the ROI box, detected contour, target crosshair, centroid,
        a vertical guide line at the detected column centre, and directional
        arrows indicating which way to move the camera.

        Box / indicator colour:
          GREEN  — within deadzone on both axes (aligned)
          ORANGE — outside deadzone, within alert (correcting)
          RED    — beyond CAMERA_ALERT_PX (far off)
          GRAY   — car not detected
        """
        x1, y1, x2, y2 = CAR_ROI
        h, w = frame.shape[:2]

        if centroid is None:
            color = (120, 120, 120)
        elif abs(cam_x_err) > CAMERA_ALERT_PX or abs(cam_y_err) > CAMERA_ALERT_PX:
            color = (0, 0, 255)
        elif abs(cam_x_err) > CAMERA_DEADZONE_PX or abs(cam_y_err) > CAMERA_DEADZONE_PX:
            color = (0, 140, 255)
        else:
            color = (0, 220, 0)

        # ROI bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        # Largest contour (visualisation only — not used for centroid)
        if contour is not None:
            cv2.drawContours(frame, [contour], -1, color, 2)

        # Target crosshair at ideal car position
        cx_t = (x1 + x2) // 2
        cy_t = int(CAR_TARGET_Y)
        cv2.drawMarker(frame, (cx_t, cy_t), color, cv2.MARKER_CROSS, 16, 1)

        if centroid is not None:
            cx, cy = int(centroid[0]), int(centroid[1])

            # Vertical guide line through the detected column centre
            cv2.line(frame, (cx, y1), (cx, y2), color, 1)

            # Centroid dot and line to target
            cv2.circle(frame, (cx, cy), 5, color, -1)
            cv2.line(frame, (cx_t, cy_t), (cx, cy), color, 1)

            # Numeric error readout
            cv2.putText(
                frame,
                f"cam dx={cam_x_err:+.0f} dy={cam_y_err:+.0f}",
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
            )

            # ── Directional guidance arrows ──────────────────────────────
            # Drawn near the centre of the overlay so they are always visible.
            origin = (w // 2, h // 2)
            arm    = 62          # arrow length in pixels
            font   = cv2.FONT_HERSHEY_SIMPLEX
            fscale = 0.54
            thick  = 2

            x_ok = abs(cam_x_err) <= CAMERA_DEADZONE_PX
            y_ok = abs(cam_y_err) <= CAMERA_DEADZONE_PX

            if x_ok and y_ok:
                cv2.putText(frame, "LOCKED",
                            (origin[0] - 42, origin[1] + 7),
                            font, 0.95, (0, 220, 0), 2, cv2.LINE_AA)
            else:
                if not x_ok:
                    go_right = cam_x_err > 0
                    tip_h = (origin[0] + (arm if go_right else -arm), origin[1])
                    cv2.arrowedLine(frame, origin, tip_h, color, 3, tipLength=0.28)
                    label = "ROTATE RIGHT" if go_right else "ROTATE LEFT"
                    lx = tip_h[0] + 6 if go_right else tip_h[0] - 108
                    cv2.putText(frame, label, (lx, origin[1] - 7),
                                font, fscale, color, thick, cv2.LINE_AA)

                if not y_ok:
                    go_down = cam_y_err > 0
                    tip_v = (origin[0], origin[1] + (arm if go_down else -arm))
                    cv2.arrowedLine(frame, origin, tip_v, color, 3, tipLength=0.28)
                    label = "PITCH DOWN" if go_down else "PITCH UP"
                    ly = tip_v[1] + 18 if go_down else tip_v[1] - 5
                    cv2.putText(frame, label, (origin[0] + 8, ly),
                                font, fscale, color, thick, cv2.LINE_AA)

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _detect(frame: np.ndarray) \
            -> tuple[Optional[Tuple[float, float]], Optional[np.ndarray]]:
        """
        Rotation-robust car centroid via 1-D projection of Canny edges.

        Pipeline:
          1. HLS L-channel + Gaussian blur.
          2. Canny edges.
          3. Glass/window suppression: zero pixels where L > 195 (specular
             glass/chrome) or S < 25 (transparent window pane or grey road).
             Eliminates the rear-window frame contour before it can bias the
             projection.
          4. Horizon suppression: top 25% of ROI zeroed (background horizon).
             Side-margin suppression: outer 8% of columns zeroed (ROI noise).
          5. Horizontal (cx): column-sum projection of the bottom 60%,
             multiplied by a Gaussian centred on the ROI to down-weight
             stray road/background edges near the boundaries.
          6. Vertical (cy): row-sum projection of the full cleaned image,
             preserving the CAR_TARGET_Y semantic.
          7. Contour (visualisation): largest contour whose bounding-rect
             width > 12% of ROI width; rejects the narrow window rectangle.
        """
        x1, y1, x2, y2 = CAR_ROI
        roi   = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        hls   = cv2.cvtColor(roi, cv2.COLOR_RGB2HLS)
        L     = hls[:, :, 1]
        S     = hls[:, :, 2]

        light = cv2.GaussianBlur(L, (5, 5), 0)
        edges = cv2.Canny(light, 40, 120)

        # ── Glass / window suppression ─────────────────────────────────────
        # L > 195: bright specular reflections (rear window, chrome trim)
        # S < 25:  transparent window glass or grey road surface
        glass = (L > 195) | (S < 25)
        edges[glass] = 0

        # ── Horizon and side-margin suppression ───────────────────────────
        edges[: int(roi_h * 0.25), :] = 0          # road horizon / sky
        margin = int(roi_w * 0.08)
        edges[:, :margin]         = 0               # left ROI boundary noise
        edges[:, roi_w - margin:] = 0               # right ROI boundary noise

        # ── Horizontal centre: column projection, bottom 60% ──────────────
        bot_start = int(roi_h * 0.40)
        col_w = edges[bot_start:, :].sum(axis=0).astype(np.float64)
        # Gaussian weight: softly prefers the ROI centre so background edges
        # at the frame boundaries don't pull the estimate sideways.
        gauss = np.exp(
            -0.5 * ((np.arange(roi_w) - roi_w / 2.0) / (roi_w * 0.30)) ** 2
        )
        col_wg = col_w * gauss
        if col_wg.sum() < max(CAR_MIN_CONTOUR_PX // 4, 50):
            return None, None
        cx = float(np.average(np.arange(roi_w), weights=col_wg)) + x1

        # ── Vertical centre: row projection, full cleaned image ────────────
        row_w = edges.sum(axis=1).astype(np.float64)
        if row_w.sum() < max(CAR_MIN_CONTOUR_PX // 4, 50):
            return None, None
        cy = float(np.average(np.arange(roi_h), weights=row_w)) + y1

        # ── Largest wide contour — visualisation only ─────────────────────
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = None
        if cnts:
            wide = [c for c in cnts
                    if cv2.boundingRect(c)[2] > roi_w * 0.12
                    and cv2.contourArea(c) >= CAR_MIN_CONTOUR_PX]
            if wide:
                best = max(wide, key=cv2.contourArea)
                contour = best + np.array([[[x1, y1]]], dtype=np.int32)

        return (cx, cy), contour

    @staticmethod
    def _scale(err: float, gain: float) -> int:
        """Apply gain + minimum-kick; return 0 inside deadzone."""
        if abs(err) <= CAMERA_DEADZONE_PX:
            return 0
        raw = gain * err
        return int(math.copysign(max(abs(raw), CAM_MIN_MOVE), raw))
