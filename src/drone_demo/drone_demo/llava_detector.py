#!/usr/bin/env python3
"""Send Gazebo camera frames to a remote LLaVA service and publish its target."""

import argparse
import math
import time

import cv2
import numpy as np
import rclpy
import requests
from geometry_msgs.msg import Point
from gz.msgs10 import image_pb2
from gz.transport13 import Node as GzNode


PIXEL_FORMAT_RGB_INT8 = 3
DEFAULT_CAMERA_TOPIC = "/ground_camera"
DEFAULT_TARGET_TOPIC = "/laser_target_pixel"
DEFAULT_SERVER_URL = "http://127.0.0.1:8000"


def target_center_from_response(result, image_width, image_height):
    """Validate a locate response and return a clamped pixel center."""
    if not isinstance(result, dict):
        raise ValueError("server response must be a JSON object")
    if result.get("found") is not True:
        return None

    center = result.get("center")
    if not isinstance(center, list) or len(center) != 2:
        raise ValueError(f"invalid center in server response: {center!r}")

    x, y = [float(value) for value in center]
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"non-finite target center: {center!r}")

    width = max(1, int(image_width))
    height = max(1, int(image_height))
    return (
        min(max(x, 0.0), width - 1.0),
        min(max(y, 0.0), height - 1.0),
    )


class RemoteLlavaDetector:
    def __init__(
        self,
        server_url,
        camera_topic,
        target_topic,
        timeout,
        jpeg_quality,
        interval,
    ):
        self.server_url = server_url.rstrip("/")
        self.locate_url = f"{self.server_url}/locate"
        self.timeout = max(0.1, float(timeout))
        self.jpeg_quality = min(max(int(jpeg_quality), 1), 100)
        self.interval = max(0.0, float(interval))

        self.ros_node = rclpy.create_node("llava_target_publisher")
        self.target_pub = self.ros_node.create_publisher(Point, target_topic, 10)
        self.gz_node = GzNode()
        self.gz_node.subscribe(
            msg_type=image_pb2.Image,
            topic=camera_topic,
            callback=self._image_callback,
        )

        self._latest_frame = None
        self._latest_frame_id = 0
        self._processed_frame_id = 0
        self._last_request_time = 0.0

        self.ros_node.get_logger().info(
            f"LLaVA bridge: camera={camera_topic}, target={target_topic}, "
            f"server={self.locate_url}"
        )

    def close(self):
        self.ros_node.destroy_node()

    def check_server(self):
        response = requests.get(
            f"{self.server_url}/health",
            timeout=self.timeout,
        )
        response.raise_for_status()
        health = response.json()
        self.ros_node.get_logger().info(f"LLaVA server health: {health}")

    def _image_callback(self, msg):
        if msg.pixel_format_type != PIXEL_FORMAT_RGB_INT8:
            return

        width, height = int(msg.width), int(msg.height)
        expected_size = width * height * 3
        if width <= 0 or height <= 0 or len(msg.data) != expected_size:
            return

        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
        self._latest_frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        self._latest_frame_id += 1

    def process_latest_frame(self):
        frame = self._latest_frame
        frame_id = self._latest_frame_id
        if frame is None or frame_id == self._processed_frame_id:
            return False

        now = time.monotonic()
        if now - self._last_request_time < self.interval:
            return False

        self._processed_frame_id = frame_id
        self._last_request_time = now
        height, width = frame.shape[:2]

        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not encoded_ok:
            raise RuntimeError("failed to encode Gazebo frame as JPEG")

        response = requests.post(
            self.locate_url,
            data={"request_id": str(frame_id)},
            files={"image": ("frame.jpg", encoded.tobytes(), "image/jpeg")},
            timeout=self.timeout,
        )
        response.raise_for_status()
        result = response.json()

        response_id = int(result.get("request_id", frame_id))
        if response_id != frame_id:
            raise ValueError(
                f"response request_id {response_id} does not match {frame_id}"
            )

        center = target_center_from_response(result, width, height)
        if center is None:
            self.ros_node.get_logger().info(
                f"Frame {frame_id}: no drone found"
            )
            return True

        msg = Point(x=center[0], y=center[1], z=0.0)
        self.target_pub.publish(msg)
        self.ros_node.get_logger().info(
            f"Frame {frame_id}: published target ({center[0]:.1f}, {center[1]:.1f})"
        )
        return True

    def run(self):
        self.check_server()
        while rclpy.ok():
            rclpy.spin_once(self.ros_node, timeout_sec=0.0)
            try:
                processed = self.process_latest_frame()
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                self.ros_node.get_logger().error(f"LLaVA request failed: {exc}")
                processed = True
            if not processed:
                time.sleep(0.01)


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Publish drone pixels returned by a remote LLaVA HTTP service"
    )
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--camera-topic", default=DEFAULT_CAMERA_TOPIC)
    parser.add_argument("--target-topic", default=DEFAULT_TARGET_TOPIC)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Minimum seconds between requests",
    )
    parsed, ros_args = parser.parse_known_args(args)

    rclpy.init(args=ros_args)
    detector = RemoteLlavaDetector(
        parsed.server_url,
        parsed.camera_topic,
        parsed.target_topic,
        parsed.timeout,
        parsed.jpeg_quality,
        parsed.interval,
    )
    try:
        detector.run()
    except KeyboardInterrupt:
        pass
    finally:
        detector.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
