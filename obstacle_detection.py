"""
Basic obstacle detector using Canny edge density inside a forward danger zone.

A vehicle directly ahead produces a notably higher edge density in the center
strip of the frame compared with clear asphalt or distant scenery. This is a
simple heuristic — tune OBSTACLE_EDGE_DENSITY_THRESHOLD in config.py to
adjust sensitivity for different lighting conditions or camera angles.
"""

import cv2
import numpy as np

from config import (
    BLUR_KERNEL, CANNY_HIGH, CANNY_LOW,
    DANGER_ZONE, OBSTACLE_EDGE_DENSITY_THRESHOLD,
)


class ObstacleDetector:
    def detect(self, frame: np.ndarray) -> bool:
        """
        Return True if an obstacle is likely blocking the road ahead.

        Uses the same blur + Canny pipeline as lane detection so the
        thresholds are consistent, but operates on the full (unmasked)
        frame to cover the danger zone regardless of the lane ROI.
        """
        gray    = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
        edges   = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)

        x1, y1, x2, y2 = DANGER_ZONE
        region = edges[y1:y2, x1:x2]
        if region.size == 0:
            return False

        density = np.count_nonzero(region) / region.size
        return bool(density > OBSTACLE_EDGE_DENSITY_THRESHOLD)
