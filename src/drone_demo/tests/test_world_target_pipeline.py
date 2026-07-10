import unittest
from unittest.mock import patch

import numpy as np

from drone_demo.laser_controller import LaserController
from drone_demo.target_filters import WorldKalmanFilter


class WorldTargetPipelineTest(unittest.TestCase):
    def _controller(self):
        controller = LaserController.__new__(LaserController)
        controller.world_target_filter = WorldKalmanFilter(
            process_noise=1.0,
            measurement_noise=0.01,
        )
        controller.world_prediction_time = 0.15
        controller.world_filter_max_measurement_age = 0.5
        controller._target_measurement_pending = True
        controller._world_filter_last_update_time = None
        controller._last_aim_source = "unknown"
        return controller

    def test_new_depth_measurement_updates_world_filter_once(self):
        controller = self._controller()
        measurement_calls = 0

        def measured_position():
            nonlocal measurement_calls
            measurement_calls += 1
            controller._last_aim_source = "depth"
            return 1.0, 2.0, 3.0

        controller._measured_target_world_position = measured_position

        with patch("drone_demo.laser_controller.time.monotonic", return_value=10.0):
            first = controller.target_world_position()
            second = controller.target_world_position()

        self.assertEqual(measurement_calls, 1)
        self.assertFalse(controller._target_measurement_pending)
        self.assertEqual(first, (1.0, 2.0, 3.0))
        self.assertEqual(second, (1.0, 2.0, 3.0))

    def test_fallback_measurement_does_not_change_filter_state(self):
        controller = self._controller()

        def depth_position():
            controller._last_aim_source = "depth"
            return 1.0, 2.0, 3.0

        controller._measured_target_world_position = depth_position
        with patch("drone_demo.laser_controller.time.monotonic", return_value=10.0):
            controller.target_world_position()

        state_before_fallback = controller.world_target_filter.state.copy()
        controller._target_measurement_pending = True

        def fallback_position():
            controller._last_aim_source = "fallback"
            return 50.0, 50.0, 50.0

        controller._measured_target_world_position = fallback_position
        with patch("drone_demo.laser_controller.time.monotonic", return_value=10.1):
            predicted = controller.target_world_position()

        np.testing.assert_array_equal(
            controller.world_target_filter.state,
            state_before_fallback,
        )
        self.assertNotEqual(predicted, (50.0, 50.0, 50.0))
        self.assertTrue(controller._target_measurement_pending)
        self.assertEqual(controller._last_aim_source, "world_kalman_hold")


if __name__ == "__main__":
    unittest.main()
