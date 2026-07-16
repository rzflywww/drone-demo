import unittest

from drone_demo.radio_jamming_controller import (
    IDLE,
    JAMMING,
    SUCCESS,
    ContinuousJammingTimer,
    point_inside_cone,
)
from drone_demo.target_geometry import pixel_to_world_direction


class ContinuousJammingTimerTest(unittest.TestCase):
    def test_timer_succeeds_after_five_continuous_seconds(self):
        timer = ContinuousJammingTimer(5.0)

        self.assertEqual(timer.update(False, 10.0), IDLE)
        self.assertEqual(timer.update(True, 10.0), JAMMING)
        self.assertEqual(timer.update(True, 14.999), JAMMING)
        self.assertEqual(timer.update(True, 15.0), SUCCESS)
        self.assertEqual(timer.update(False, 30.0), SUCCESS)

    def test_leaving_cone_resets_continuous_coverage_timer(self):
        timer = ContinuousJammingTimer(5.0)

        self.assertEqual(timer.update(True, 10.0), JAMMING)
        self.assertEqual(timer.update(True, 14.0), JAMMING)
        self.assertEqual(timer.update(False, 14.5), IDLE)
        self.assertEqual(timer.elapsed(14.5), 0.0)
        self.assertEqual(timer.update(True, 20.0), JAMMING)
        self.assertEqual(timer.update(True, 24.999), JAMMING)
        self.assertEqual(timer.update(True, 25.0), SUCCESS)


class ConeCoverageTest(unittest.TestCase):
    def test_point_must_be_within_length_and_expanding_radius(self):
        apex = (0.0, 0.0, 0.0)
        axis = (0.0, 0.0, 1.0)

        self.assertTrue(point_inside_cone((0.0, 0.0, 5.0), apex, axis, 10.0, 4.0))
        self.assertTrue(point_inside_cone((1.9, 0.0, 5.0), apex, axis, 10.0, 4.0))
        self.assertFalse(point_inside_cone((2.1, 0.0, 5.0), apex, axis, 10.0, 4.0))
        self.assertFalse(point_inside_cone((0.0, 0.0, -1.0), apex, axis, 10.0, 4.0))
        self.assertFalse(point_inside_cone((0.0, 0.0, 10.1), apex, axis, 10.0, 4.0))


class FixedDistanceProjectionTest(unittest.TestCase):
    def test_center_pixel_uses_camera_forward_direction(self):
        direction = pixel_to_world_direction(
            640.0,
            360.0,
            1280.0,
            720.0,
            1.047,
            (0.0, 0.0, 0.0),
        )

        self.assertAlmostEqual(direction[0], 1.0)
        self.assertAlmostEqual(direction[1], 0.0)
        self.assertAlmostEqual(direction[2], 0.0)


if __name__ == "__main__":
    unittest.main()
