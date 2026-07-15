#!/usr/bin/env python3
"""Send Gazebo camera frames to a remote LLaVA service and publish its target."""

import argparse
import math
import os
import shutil
import signal
import subprocess
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
DEFAULT_TEMPERATURE = 0.95
DEFAULT_TOP_P = 0.7
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_LASER_TRIGGER_PHRASE = "laser strikes"
DEFAULT_PROMPT = """You are a counter-drone system operator. When a drone
appears in an image, you need to analyze the surrounding environment and select
the most appropriate strike plan. Available strike methods include: radio
jamming, laser strikes, and high-energy microwave strikes. Please design a
suitable strike plan."""


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


def generation_request_data(
    request_id,
    prompt,
    do_sample,
    temperature,
    top_p,
    max_new_tokens,
):
    """Build the multipart fields understood by the remote inference service."""
    return {
        "request_id": str(request_id),
        "prompt": prompt,
        "do_sample": "true" if do_sample else "false",
        "temperature": str(temperature),
        "top_p": str(top_p),
        "max_new_tokens": str(max_new_tokens),
    }


def answer_contains_trigger(raw_answer, trigger_phrase):
    """Return whether a model answer contains the configured trigger phrase."""
    if not isinstance(raw_answer, str) or not trigger_phrase:
        return False
    return trigger_phrase.casefold() in raw_answer.casefold()


class RemoteLlavaDetector:
    def __init__(
        self,
        server_url,
        camera_topic,
        target_topic,
        timeout,
        jpeg_quality,
        interval,
        prompt,
        do_sample,
        temperature,
        top_p,
        max_new_tokens,
        auto_start_laser,
        laser_trigger_phrase,
        laser_startup_timeout,
    ):
        self.server_url = server_url.rstrip("/")
        self.locate_url = f"{self.server_url}/locate"
        self.timeout = max(0.1, float(timeout))
        self.jpeg_quality = min(max(int(jpeg_quality), 1), 100)
        self.interval = max(0.0, float(interval))
        self.prompt = prompt
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_new_tokens = int(max_new_tokens)
        self.auto_start_laser = bool(auto_start_laser)
        self.laser_trigger_phrase = str(laser_trigger_phrase)
        self.laser_startup_timeout = max(0.0, float(laser_startup_timeout))
        self._trigger_handled = False
        self._tracking_handed_off = False
        self._managed_processes = {}

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
            f"server={self.locate_url}, sample={self.do_sample}, "
            f"temperature={self.temperature}, top_p={self.top_p}, "
            f"auto_laser={self.auto_start_laser}, "
            f"laser_trigger={self.laser_trigger_phrase!r}"
        )

    def close(self):
        self._stop_managed_processes()
        self.ros_node.destroy_node()

    def _node_is_running(self, node_name):
        return any(
            name == node_name
            for name, _namespace in self.ros_node.get_node_names_and_namespaces()
        )

    def _start_managed_node(self, executable, node_name, extra_args=None):
        if self._node_is_running(node_name):
            self.ros_node.get_logger().info(
                f"Trigger action reusing existing ROS node {node_name}"
            )
            return True

        existing = self._managed_processes.get(executable)
        if existing is not None and existing.poll() is None:
            return True

        ros2 = shutil.which("ros2")
        if ros2 is None:
            self.ros_node.get_logger().error(
                f"Cannot start {executable}: the ros2 executable was not found"
            )
            return False

        try:
            command = [ros2, "run", "drone_demo", executable]
            if extra_args:
                command.extend(extra_args)
            process = subprocess.Popen(
                command,
                start_new_session=True,
            )
        except OSError as exc:
            self.ros_node.get_logger().error(
                f"Failed to start {executable}: {exc}"
            )
            return False

        self._managed_processes[executable] = process
        self.ros_node.get_logger().warn(
            f"Started {executable} pid={process.pid}"
        )
        return True

    def _wait_for_nodes(self, node_names):
        deadline = time.monotonic() + self.laser_startup_timeout
        while rclpy.ok() and time.monotonic() < deadline:
            if all(self._node_is_running(name) for name in node_names):
                return True
            for executable, process in self._managed_processes.items():
                return_code = process.poll()
                if return_code is not None:
                    self.ros_node.get_logger().error(
                        f"{executable} exited during startup with code {return_code}"
                    )
                    return False
            rclpy.spin_once(self.ros_node, timeout_sec=0.05)
        return all(self._node_is_running(name) for name in node_names)

    def _maybe_handoff_to_yolo(self, raw_answer):
        if not self.auto_start_laser or self._trigger_handled:
            return False
        if not answer_contains_trigger(raw_answer, self.laser_trigger_phrase):
            return False

        self._trigger_handled = True
        self.ros_node.get_logger().warn(
            f"Trigger {self.laser_trigger_phrase!r} matched; "
            "starting laser_controller and yolo_detector"
        )

        laser_started = self._start_managed_node(
            "laser_controller",
            "laser_controller",
            [
                "--ros-args",
                "-p",
                "world_target_filter:=kalman",
                "-p",
                "world_prediction_time:=0.15",
            ],
        )
        yolo_started = self._start_managed_node(
            "yolo_detector",
            "yolo_target_publisher",
        )
        nodes_ready = laser_started and yolo_started and self._wait_for_nodes(
            {"laser_controller", "yolo_target_publisher"}
        )
        if not nodes_ready:
            self.ros_node.get_logger().error(
                "YOLO handoff failed; keeping LLaVA target publishing active"
            )
            self._stop_managed_processes()
            self._trigger_handled = False
            return False

        self._tracking_handed_off = True
        self.ros_node.destroy_publisher(self.target_pub)
        self.target_pub = None
        self.ros_node.get_logger().warn(
            "YOLO handoff complete; LLaVA target publishing and inference stopped"
        )
        return True

    @staticmethod
    def _stop_process(process):
        if process is None or process.poll() is not None:
            return

        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=2.0)
                except ProcessLookupError:
                    pass
            except ProcessLookupError:
                pass
        except ProcessLookupError:
            pass

    def _stop_managed_processes(self):
        for executable in ("yolo_detector", "laser_controller"):
            self._stop_process(self._managed_processes.get(executable))
        self._managed_processes.clear()

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
        if self._tracking_handed_off:
            return False

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
            data=generation_request_data(
                frame_id,
                self.prompt,
                self.do_sample,
                self.temperature,
                self.top_p,
                self.max_new_tokens,
            ),
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

        if self._maybe_handoff_to_yolo(result.get("raw_answer")):
            return True

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
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-sample", dest="do_sample", action="store_false")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
    )
    parser.add_argument(
        "--laser-trigger-phrase",
        default=DEFAULT_LASER_TRIGGER_PHRASE,
    )
    parser.add_argument(
        "--auto-start-laser",
        dest="auto_start_laser",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-auto-laser",
        dest="auto_start_laser",
        action="store_false",
    )
    parser.add_argument("--laser-startup-timeout", type=float, default=5.0)
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
        parsed.prompt,
        parsed.do_sample,
        parsed.temperature,
        parsed.top_p,
        parsed.max_new_tokens,
        parsed.auto_start_laser,
        parsed.laser_trigger_phrase,
        parsed.laser_startup_timeout,
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
