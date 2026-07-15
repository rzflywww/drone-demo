import unittest
from unittest.mock import MagicMock, call, patch

from drone_demo.llava_detector import (
    RemoteLlavaDetector,
    answer_contains_trigger,
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

    def test_laser_trigger_phrase_is_case_insensitive(self):
        self.assertTrue(
            answer_contains_trigger(
                "The selected plan is LASER STRIKES.",
                "laser strikes",
            )
        )

    def test_laser_trigger_requires_the_complete_phrase(self):
        self.assertFalse(
            answer_contains_trigger(
                "Use radio jamming instead.",
                "laser strikes",
            )
        )

    def test_laser_trigger_rejects_empty_values(self):
        self.assertFalse(answer_contains_trigger(None, "laser strikes"))
        self.assertFalse(answer_contains_trigger("laser strikes", ""))


class YoloHandoffTriggerTest(unittest.TestCase):
    def setUp(self):
        self.detector = RemoteLlavaDetector.__new__(RemoteLlavaDetector)
        self.detector.auto_start_laser = True
        self.detector._trigger_handled = False
        self.detector._tracking_handed_off = False
        self.detector._managed_processes = {}
        self.detector.laser_trigger_phrase = "laser strikes"
        self.detector.ros_node = MagicMock()
        self.detector.target_pub = MagicMock()
        self.detector._wait_for_nodes = MagicMock(return_value=True)

    @patch("drone_demo.llava_detector.subprocess.Popen")
    @patch("drone_demo.llava_detector.shutil.which", return_value="/usr/bin/ros2")
    def test_match_starts_laser_and_yolo_once(self, _which, popen):
        laser_process = MagicMock(pid=123)
        yolo_process = MagicMock(pid=124)
        popen.side_effect = [laser_process, yolo_process]
        self.detector._node_is_running = MagicMock(return_value=False)

        first = self.detector._maybe_handoff_to_yolo("Use laser strikes.")
        second = self.detector._maybe_handoff_to_yolo("Use laser strikes.")

        popen.assert_has_calls(
            [
                call(
                    [
                        "/usr/bin/ros2",
                        "run",
                        "drone_demo",
                        "laser_controller",
                        "--ros-args",
                        "-p",
                        "world_target_filter:=kalman",
                        "-p",
                        "world_prediction_time:=0.15",
                    ],
                    start_new_session=True,
                ),
                call(
                    ["/usr/bin/ros2", "run", "drone_demo", "yolo_detector"],
                    start_new_session=True,
                ),
            ]
        )
        self.assertEqual(popen.call_count, 2)
        self.assertTrue(first)
        self.assertFalse(second)
        self.assertTrue(self.detector._tracking_handed_off)
        self.detector.ros_node.destroy_publisher.assert_called_once()
        self.assertIsNone(self.detector.target_pub)

    @patch("drone_demo.llava_detector.subprocess.Popen")
    def test_running_nodes_are_not_started_again(self, popen):
        self.detector._node_is_running = MagicMock(return_value=True)

        result = self.detector._maybe_handoff_to_yolo("Use laser strikes.")

        popen.assert_not_called()
        self.assertTrue(result)
        self.assertTrue(self.detector._tracking_handed_off)

    @patch("drone_demo.llava_detector.requests.post")
    def test_handoff_stops_further_llava_requests(self, post):
        self.detector._tracking_handed_off = True

        self.assertFalse(self.detector.process_latest_frame())
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
