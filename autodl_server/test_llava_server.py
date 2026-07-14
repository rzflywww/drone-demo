import unittest

from llava_server import parse_model_answer


class ModelAnswerTest(unittest.TestCase):
    def test_zero_to_one_bbox_is_converted_to_pixels(self):
        result = parse_model_answer(
            '{"found":true,"bbox":[0.25,0.25,0.75,0.75]}',
            640,
            360,
        )
        self.assertEqual(result["bbox"], [160.0, 90.0, 480.0, 270.0])
        self.assertEqual(result["center"], [320.0, 180.0])

    def test_zero_to_thousand_bbox_remains_compatible(self):
        result = parse_model_answer(
            '{"found":true,"bbox":[250,250,750,750]}',
            640,
            360,
        )
        self.assertEqual(result["bbox"], [160.0, 90.0, 480.0, 270.0])
        self.assertEqual(result["center"], [320.0, 180.0])

    def test_not_found_response(self):
        result = parse_model_answer(
            '{"found":false,"bbox":null}',
            640,
            360,
        )
        self.assertEqual(
            result,
            {"found": False, "bbox": None, "center": None},
        )

    def test_lora_pixel_center_is_extracted_from_text(self):
        result = parse_model_answer(
            "laser strikes should be used, the center position is[[827, 324]].",
            1280,
            720,
        )
        self.assertEqual(
            result,
            {"found": True, "bbox": None, "center": [827.0, 324.0]},
        )

    def test_normalized_json_center_is_supported(self):
        result = parse_model_answer(
            '{"found":true,"center":[0.5,0.5]}',
            1280,
            720,
        )
        self.assertEqual(result["center"], [640.0, 360.0])


if __name__ == "__main__":
    unittest.main()
