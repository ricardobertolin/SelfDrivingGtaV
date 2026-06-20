import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    BLUR_KERNEL,
    CANNY_SIGMA, CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID,
    FRAME_WIDTH,
    HOUGH_MAX_LINE_GAP, HOUGH_MIN_LINE_LEN, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
    LANE_CURVE_POINTS, LANE_MAX_JUMP_PX, LANE_MIN_YSPAN_PX, MORPH_CLOSE_ITERS,
    ROI_BOTTOM_Y, ROI_TOP_Y, ROI_VERTICES,
    SLOPE_MAX, SLOPE_MIN, SMOOTHING_WINDOW, TYPICAL_LANE_WIDTH,
    WHITE_LANE_HIGH, WHITE_LANE_LOW, YELLOW_LANE_HIGH, YELLOW_LANE_LOW,
)

# Polynomial representation: coefficients [a, b, c] for  x = a·y² + b·y + c
# (fitting x as a function of y handles vertical-ish lines naturally)
Poly  = np.ndarray          # shape (3,)
LanePoints = np.ndarray     # shape (LANE_CURVE_POINTS, 2)  dtype int32

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
_Y_VALS       = np.linspace(ROI_TOP_Y, ROI_BOTTOM_Y, LANE_CURVE_POINTS)


@dataclass
class DetectionResult:
    left_lane:   Optional[LanePoints]  # Nx2 int32 curve, or None
    right_lane:  Optional[LanePoints]
    left_fresh:  bool   # True → detected this frame; False → from buffer / extrapolated
    right_fresh: bool
    line_count:  int
    buffer_fill: int
    heading_deg: Optional[float]       # 0=straight, +right, −left


class LaneDetector:
    """
    Detects left and right lane markings and returns polynomial curves.

    Internal smoothing buffers store polynomial coefficients [a, b, c] so
    the smooth output bends with the road rather than forcing a straight line.
    """

    def __init__(self) -> None:
        self._left_buf:  deque = deque(maxlen=SMOOTHING_WINDOW)
        self._right_buf: deque = deque(maxlen=SMOOTHING_WINDOW)
        self._clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> DetectionResult:
        edges     = self._preprocess(frame)
        masked    = self._apply_roi(edges)
        raw_lines = cv2.HoughLinesP(
            masked, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LEN,
            maxLineGap=HOUGH_MAX_LINE_GAP,
        )
        line_count = len(raw_lines) if raw_lines is not None else 0

        left_segs, right_segs = self._classify(raw_lines)

        left_poly  = self._fit_poly(left_segs)
        right_poly = self._fit_poly(right_segs)

        # Temporal consistency gate — discard detections that jump too far
        if left_poly is not None and self._left_buf:
            if abs(self._x_at_y(left_poly,  ROI_BOTTOM_Y) -
                   self._x_at_y(self._mean_poly(self._left_buf), ROI_BOTTOM_Y)) > LANE_MAX_JUMP_PX:
                left_poly = None
        if right_poly is not None and self._right_buf:
            if abs(self._x_at_y(right_poly, ROI_BOTTOM_Y) -
                   self._x_at_y(self._mean_poly(self._right_buf), ROI_BOTTOM_Y)) > LANE_MAX_JUMP_PX:
                right_poly = None

        left_fresh  = left_poly  is not None
        right_fresh = right_poly is not None

        if left_poly  is not None: self._left_buf.append(left_poly)
        if right_poly is not None: self._right_buf.append(right_poly)

        # Smoothed polynomials from buffer
        s_left_p  = self._mean_poly(self._left_buf)  if self._left_buf  else None
        s_right_p = self._mean_poly(self._right_buf) if self._right_buf else None

        # Extrapolate the missing side by shifting the known one horizontally
        if s_left_p is None and s_right_p is not None:
            s_left_p  = self._shift_poly(s_right_p, -TYPICAL_LANE_WIDTH)
        elif s_right_p is None and s_left_p is not None:
            s_right_p = self._shift_poly(s_left_p,  +TYPICAL_LANE_WIDTH)

        # Sanity check: left lane must sit to the LEFT of right lane at the car.
        # Crossing = bad detection → clear both buffers for fast recovery.
        if s_left_p is not None and s_right_p is not None:
            if self._x_at_y(s_left_p, ROI_BOTTOM_Y) >= self._x_at_y(s_right_p, ROI_BOTTOM_Y) - 30:
                self._left_buf.clear()
                self._right_buf.clear()
                s_left_p = s_right_p = None
                left_fresh = right_fresh = False

        # Convert polynomials → (x, y) point arrays for drawing
        s_left  = self._poly_to_points(s_left_p)  if s_left_p  is not None else None
        s_right = self._poly_to_points(s_right_p) if s_right_p is not None else None

        buf_fill = (len(self._left_buf) + len(self._right_buf)) // 2

        # Heading: angle of the centre polynomial from vertical (down-to-up direction)
        heading_deg: Optional[float] = None
        if s_left_p is not None and s_right_p is not None:
            cp     = (s_left_p + s_right_p) / 2.0
            cx_top = self._x_at_y(cp, ROI_TOP_Y)
            cx_bot = self._x_at_y(cp, ROI_BOTTOM_Y)
            heading_deg = math.degrees(
                math.atan2(cx_top - cx_bot, ROI_BOTTOM_Y - ROI_TOP_Y)
            )

        return DetectionResult(
            s_left, s_right, left_fresh, right_fresh,
            line_count, buf_fill, heading_deg,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE → adaptive Canny → HLS colour mask → morphological close."""
        hls = cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)

        l_enh  = self._clahe.apply(hls[:, :, 1])
        blurrd = cv2.GaussianBlur(l_enh, BLUR_KERNEL, 0)
        lo, hi = self._auto_canny(blurrd)
        edges  = cv2.Canny(blurrd, lo, hi)

        white  = cv2.inRange(hls, WHITE_LANE_LOW,  WHITE_LANE_HIGH)
        yellow = cv2.inRange(hls, YELLOW_LANE_LOW, YELLOW_LANE_HIGH)

        combined = cv2.bitwise_or(edges, cv2.bitwise_or(white, yellow))
        return cv2.morphologyEx(
            combined, cv2.MORPH_CLOSE, _MORPH_KERNEL, iterations=MORPH_CLOSE_ITERS
        )

    @staticmethod
    def _auto_canny(img: np.ndarray) -> Tuple[int, int]:
        v    = float(np.median(img))
        lo   = int(max(0,   (1.0 - CANNY_SIGMA) * v))
        hi   = int(min(255, (1.0 + CANNY_SIGMA) * v))
        return max(lo, 10), max(hi, lo + 20)

    # ------------------------------------------------------------------
    # ROI & segment classification
    # ------------------------------------------------------------------

    def _apply_roi(self, img: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [ROI_VERTICES], 255)
        return cv2.bitwise_and(img, mask)

    def _classify(self, raw_lines) -> Tuple[list, list]:
        """
        Split Hough segments by slope sign AND x-position.

        Slope alone is insufficient: a noisy segment with the wrong sign on
        the wrong side of the frame can flip the entire lane estimate.
        The x-position gate adds a second independent constraint:
          left  lane: slope < 0  AND  mean_x < FRAME_WIDTH * 0.65
          right lane: slope > 0  AND  mean_x > FRAME_WIDTH * 0.35
        The 30 % overlap zone (0.35–0.65) keeps tight-curve lanes visible
        when they drift toward the centre without accepting cross-over junk.
        """
        left_segs:  list = []
        right_segs: list = []
        if raw_lines is None:
            return left_segs, right_segs
        for seg in raw_lines:
            x1, y1, x2, y2 = seg[0]
            dx = x2 - x1
            if dx == 0:
                continue
            slope  = (y2 - y1) / dx
            if abs(slope) < SLOPE_MIN or abs(slope) > SLOPE_MAX:
                continue
            mean_x = (x1 + x2) / 2.0
            if slope < 0 and mean_x < FRAME_WIDTH * 0.65:
                left_segs.append((slope, x1, y1, x2, y2))
            elif slope > 0 and mean_x > FRAME_WIDTH * 0.35:
                right_segs.append((slope, x1, y1, x2, y2))
        return left_segs, right_segs

    # ------------------------------------------------------------------
    # Polynomial fitting
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_poly(segs: list) -> Optional[Poly]:
        """
        Fit x = a·y² + b·y + c through all inlier segment endpoints,
        weighted by each segment's pixel length.

        Longer Hough segments are more reliable (they represent continuous
        lane paint rather than noise blips), so they pull the fitted curve
        more than short fragments.  IQR slope filtering removes outliers
        before fitting.  Falls back to degree-1 when fewer than 6 points
        (stored as [0, b, c] so the buffer always holds 3-element arrays).
        """
        if not segs:
            return None

        slopes = np.array([s[0] for s in segs], dtype=np.float32)

        # IQR slope filter
        if len(slopes) >= 4:
            q1, q3  = np.percentile(slopes, [25, 75])
            iqr     = q3 - q1
            inliers = [s for s, sl in zip(segs, slopes)
                       if q1 - 1.5 * iqr <= sl <= q3 + 1.5 * iqr]
            if inliers:
                segs = inliers

        # Vertical-span gate: segments that are all bunched within a narrow
        # y-band produce a polynomial that is mostly extrapolated — unreliable.
        ys_all = [s[2] for s in segs] + [s[4] for s in segs]
        if max(ys_all) - min(ys_all) < LANE_MIN_YSPAN_PX:
            return None

        xs: list = []
        ys: list = []
        ws: list = []   # weight = pixel length of the segment
        for s in segs:
            x1, y1, x2, y2 = s[1], s[2], s[3], s[4]
            length = math.hypot(x2 - x1, y2 - y1)
            xs += [x1, x2]
            ys += [y1, y2]
            ws += [length, length]

        if len(xs) < 2:
            return None

        try:
            deg  = 2 if len(xs) >= 6 else 1
            raw  = np.polyfit(ys, xs, deg, w=ws)
            poly = np.array([0.0, raw[0], raw[1]]) if deg == 1 \
                   else raw.astype(np.float64)
            return poly
        except np.linalg.LinAlgError:
            return None

    # ------------------------------------------------------------------
    # Polynomial helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _x_at_y(poly: Poly, y: float) -> float:
        return float(np.polyval(poly, y))

    @staticmethod
    def _mean_poly(buf: deque) -> Poly:
        return np.array(list(buf), dtype=np.float64).mean(axis=0)

    @staticmethod
    def _shift_poly(poly: Poly, dx: float) -> Poly:
        """Shift x = f(y) horizontally: only the constant term (c) changes."""
        shifted    = poly.copy()
        shifted[2] += dx
        return shifted

    @staticmethod
    def _poly_to_points(poly: Poly) -> LanePoints:
        x_vals = np.polyval(poly, _Y_VALS)
        return np.column_stack([x_vals, _Y_VALS]).astype(np.int32)
