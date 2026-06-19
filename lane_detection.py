from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    BLUR_KERNEL, CANNY_HIGH, CANNY_LOW,
    HOUGH_MAX_LINE_GAP, HOUGH_MIN_LINE_LEN, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
    ROI_BOTTOM_Y, ROI_TOP_Y, ROI_VERTICES,
    SLOPE_MAX, SLOPE_MIN, SMOOTHING_WINDOW, TYPICAL_LANE_WIDTH,
)

# (x_top, y_top, x_bot, y_bot) — both endpoints extended to ROI_TOP_Y / ROI_BOTTOM_Y
Lane = Tuple[int, int, int, int]


@dataclass
class DetectionResult:
    left_lane:    Optional[Lane]
    right_lane:   Optional[Lane]
    line_count:   int   # raw Hough lines found this frame
    buffer_fill:  int   # avg number of entries in smoothing buffers


class LaneDetector:
    def __init__(self) -> None:
        self._left_buf:  deque = deque(maxlen=SMOOTHING_WINDOW)
        self._right_buf: deque = deque(maxlen=SMOOTHING_WINDOW)

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

        left  = self._fit_lane(left_segs)
        right = self._fit_lane(right_segs)

        # Push confirmed detections into rolling buffers
        if left is not None:
            self._left_buf.append(left)
        if right is not None:
            self._right_buf.append(right)

        # Smooth from buffers; fall back to extrapolation when one side is empty
        s_left  = self._smooth(self._left_buf)  if self._left_buf  else None
        s_right = self._smooth(self._right_buf) if self._right_buf else None

        if s_left is None and s_right is not None:
            s_left  = self._shift_lane(s_right, -TYPICAL_LANE_WIDTH)
        elif s_right is None and s_left is not None:
            s_right = self._shift_lane(s_left,  +TYPICAL_LANE_WIDTH)

        buf_fill = (len(self._left_buf) + len(self._right_buf)) // 2

        return DetectionResult(s_left, s_right, line_count, buf_fill)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)   # blur BEFORE Canny
        return cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

    def _apply_roi(self, img: np.ndarray) -> np.ndarray:
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, [ROI_VERTICES], 255)
        return cv2.bitwise_and(img, mask)

    def _classify(self, raw_lines) -> Tuple[list, list]:
        """Split Hough segments into left (negative slope) / right (positive slope)."""
        left_segs:  list = []
        right_segs: list = []

        if raw_lines is None:
            return left_segs, right_segs

        for seg in raw_lines:
            x1, y1, x2, y2 = seg[0]
            dx = x2 - x1
            if dx == 0:
                continue  # skip perfectly vertical lines (undefined slope)
            slope = (y2 - y1) / dx
            if abs(slope) < SLOPE_MIN or abs(slope) > SLOPE_MAX:
                continue  # discard near-horizontal and near-vertical noise
            if slope < 0:
                left_segs.append((slope, x1, y1, x2, y2))
            else:
                right_segs.append((slope, x1, y1, x2, y2))

        return left_segs, right_segs

    def _fit_lane(self, segs: list) -> Optional[Lane]:
        """Average all segments on one side and extend to ROI_TOP_Y / ROI_BOTTOM_Y."""
        if not segs:
            return None

        slopes = [s[0] for s in segs]
        # Collect all (x, y) points from every segment
        xs = [s[1] for s in segs] + [s[3] for s in segs]
        ys = [s[2] for s in segs] + [s[4] for s in segs]

        m = float(np.mean(slopes))
        if abs(m) < 1e-6:
            return None

        b = float(np.mean(ys)) - m * float(np.mean(xs))

        # x = (y - b) / m
        x_top = int((ROI_TOP_Y    - b) / m)
        x_bot = int((ROI_BOTTOM_Y - b) / m)
        return (x_top, ROI_TOP_Y, x_bot, ROI_BOTTOM_Y)

    @staticmethod
    def _smooth(buf: deque) -> Lane:
        arr  = np.array(list(buf), dtype=np.float32)
        mean = arr.mean(axis=0)
        return (int(mean[0]), int(mean[1]), int(mean[2]), int(mean[3]))

    @staticmethod
    def _shift_lane(lane: Lane, dx: int) -> Lane:
        """Horizontally shift a lane by dx pixels (extrapolate missing side)."""
        return (lane[0] + dx, lane[1], lane[2] + dx, lane[3])
