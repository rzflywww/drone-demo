import unittest

from drone_demo.llava_detector import (
    generation_request_data,
    target_center_from_response,
)


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

    def test_generation_fields_are_encoded_for_multipart_request(self):
        result = generation_request_data(
            42,
            "locate the drone",
            True,
            0.95,
            0.7,
            128,
        )
        self.assertEqual(
            result,
            {
                "request_id": "42",
                "prompt": "locate the drone",
                "do_sample": "true",
                "temperature": "0.95",
                "top_p": "0.7",
                "max_new_tokens": "128",
            },
        )


if __name__ == "__main__":
    unittest.main()
