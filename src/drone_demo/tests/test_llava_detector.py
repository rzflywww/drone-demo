import unittest

from drone_demo.llava_detector import target_center_from_response


class LlavaResponseTest(unittest.TestCase):
    def test_valid_center_is_returned(self):
        center = target_center_from_response(
            {"found": True, "center": [320.0, 180.0]},
            640,
            360,
        )
        self.assertEqual(center, (320.0, 180.0))

    def test_missing_target_returns_none(self):
        center = target_center_from_response(
            {"found": False, "center": None},
            640,
            360,
        )
        self.assertIsNone(center)

    def test_center_is_clamped_to_image(self):
        center = target_center_from_response(
            {"found": True, "center": [-20.0, 500.0]},
            640,
            360,
        )
        self.assertEqual(center, (0.0, 359.0))

    def test_invalid_center_is_rejected(self):
        with self.assertRaises(ValueError):
            target_center_from_response(
                {"found": True, "center": [320.0]},
                640,
                360,
            )


if __name__ == "__main__":
    unittest.main()
