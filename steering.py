from typing import Optional, Tuple

from config import FRAME_WIDTH, SPEED_BRAKE_THRESHOLD, STEERING_DEADZONE

Lane = Tuple[int, int, int, int]  # (x_top, y_top, x_bot, y_bot)


class SteeringController:
    """
    Simple proportional (P) controller that maps lane-center offset to a
    discrete steering action: 'left', 'right', or 'straight'.
    """

    def calculate_offset(
        self,
        left_lane:  Optional[Lane],
        right_lane: Optional[Lane],
    ) -> float:
        """
        Return the signed pixel distance between the detected lane center
        and the image center at ROI_BOTTOM_Y (the nearest point to the car).

        Positive  → lane center is right  of image center → car drifting left  → steer right
        Negative  → lane center is left   of image center → car drifting right → steer left
        """
        if left_lane is None or right_lane is None:
            return 0.0

        # x_bot is index 2 in (x_top, y_top, x_bot, y_bot)
        lane_center  = (left_lane[2] + right_lane[2]) / 2.0
        image_center = FRAME_WIDTH / 2.0
        return lane_center - image_center

    def decide_action(self, offset: float) -> str:
        """Map offset to 'left', 'right', or 'straight' with a deadzone."""
        if offset > STEERING_DEADZONE:
            return 'right'   # lane center is right of car → steer right
        if offset < -STEERING_DEADZONE:
            return 'left'    # lane center is left of car  → steer left
        return 'straight'

    def decide_speed(self, offset: float) -> str:
        """
        Map offset magnitude to a throttle state.

        'accelerate' — W held   (small offset, on course)
        'coast'      — W and S released (moderate correction, slow naturally)
        'brake'      — S held, W released (large offset, sharp correction)
        """
        abs_offset = abs(offset)
        if abs_offset <= STEERING_DEADZONE:
            return 'accelerate'
        if abs_offset <= SPEED_BRAKE_THRESHOLD:
            return 'coast'
        return 'brake'
