import unittest

from drone_demo.target_filters import (
    WorldKalmanFilter,
    create_world_target_filter,
)


class TargetFilterFactoryTest(unittest.TestCase):
    def test_unknown_filter_is_rejected(self):
        with self.assertRaises(ValueError):
            create_world_target_filter("unknown")

    def test_world_filter_is_created_only_when_selected(self):
        self.assertIsNone(create_world_target_filter("none"))

        target_filter = create_world_target_filter(
            "kalman",
            kalman_process_noise=2.0,
            kalman_measurement_noise=0.01,
        )

        self.assertIsInstance(target_filter, WorldKalmanFilter)
        self.assertEqual(target_filter.dimensions, 3)
        self.assertEqual(target_filter.process_noise, 2.0)
        self.assertEqual(target_filter.measurement_noise, 0.01)

    def test_world_filter_predicts_in_three_dimensions(self):
        target_filter = WorldKalmanFilter(
            process_noise=0.1,
            measurement_noise=0.001,
        )
        for index in range(41):
            timestamp = index * 0.1
            target_filter.update(
                timestamp,
                2.0 * timestamp,
                3.0 - 0.5 * timestamp,
                timestamp,
            )

        predicted = target_filter.predict(0.2)

        self.assertAlmostEqual(predicted[0], 4.2, delta=0.02)
        self.assertAlmostEqual(predicted[1], 8.4, delta=0.02)
        self.assertAlmostEqual(predicted[2], 0.9, delta=0.02)


if __name__ == "__main__":
    unittest.main()
