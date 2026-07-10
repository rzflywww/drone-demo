import unittest

from drone_demo.target_filters import PixelKalmanFilter, create_target_filter


class TargetFilterFactoryTest(unittest.TestCase):
    def test_none_is_a_true_bypass(self):
        self.assertIsNone(create_target_filter("none"))

    def test_kalman_filter_is_created_only_when_selected(self):
        target_filter = create_target_filter(
            "kalman",
            kalman_process_noise=123.0,
            kalman_measurement_noise=7.0,
        )

        self.assertIsInstance(target_filter, PixelKalmanFilter)
        self.assertEqual(target_filter.process_noise, 123.0)
        self.assertEqual(target_filter.measurement_noise, 7.0)

    def test_unknown_filter_is_rejected(self):
        with self.assertRaises(ValueError):
            create_target_filter("unknown")


if __name__ == "__main__":
    unittest.main()
