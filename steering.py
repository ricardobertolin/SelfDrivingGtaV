import numpy as np

from config import FRAME_WIDTH, SPEED_BRAKE_THRESHOLD, STEERING_DEADZONE


class SteeringController:
    """
    Proportional (P) controller: maps lane-centre offset to a discrete action.

    Lanes are now Nx2 point arrays; the bottom row (index -1) gives the
    x position closest to the car, which is what we steer toward.
    """

    def calculate_offset(
        self,
        left_lane:  np.ndarray,
        right_lane: np.ndarray,
    ) -> float:
        """
        Return the signed pixel offset of the lane centre from the image centre
        at the bottom of the ROI (closest point to the car).

        Positive  → lane centre is right of image centre → car is left of lane  → steer right
        Negative  → lane centre is left  of image centre → car is right of lane → steer left
        """
        left_x       = float(left_lane[-1, 0])
        right_x      = float(right_lane[-1, 0])
        lane_centre  = (left_x + right_x) / 2.0
        image_centre = FRAME_WIDTH / 2.0
        return lane_centre - image_centre

    def decide_action(self, offset: float) -> str:
        if offset > STEERING_DEADZONE:
            return 'right'
        if offset < -STEERING_DEADZONE:
            return 'left'
        return 'straight'

    def decide_speed(self, offset: float) -> str:
        abs_off = abs(offset)
        if abs_off <= STEERING_DEADZONE:
            return 'accelerate'
        if abs_off <= SPEED_BRAKE_THRESHOLD:
            return 'coast'
        return 'brake'
