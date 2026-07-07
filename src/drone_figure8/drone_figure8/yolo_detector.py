#!/usr/bin/env python3
"""
Real-time YOLO11n object detection on Gazebo ground_camera feed.

Subscribes to /ground_camera via gz.transport13 Python bindings,
runs YOLO11n inference, and displays annotated video with cv2.imshow.

Usage:
    # With Gazebo running (after source venv + ROS 2 setup):
    ros2 run drone_figure8 yolo_detector

    # Or directly:
    python3 yolo_detector.py [--model PATH] [--conf 0.25] [--topic /ground_camera]

    # Publish detected target center to the laser controller:
    ros2 run drone_figure8 yolo_detector --ros-args
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from gz.msgs10 import image_pb2
from gz.transport13 import Node as GzNode

# --- Constants ---
PIXEL_FORMAT_RGB_INT8 = 3
DEFAULT_MODEL = "/home/rzfly/ultralytics-8.3.39/ultralytics/yolo11n.pt"
DEFAULT_TOPIC = "/ground_camera"
DEFAULT_CONF = 0.25
DEFAULT_TARGET_TOPIC = "/laser_target_pixel"
DEFAULT_PREDICTION_TIME = 0.15
YOLO_VENV_SITE_PACKAGES = Path(
    "/home/rzfly/drone_ws/yolo_venv/lib/python3.12/site-packages"
)


def import_yolo(model_path):
    if YOLO_VENV_SITE_PACKAGES.exists():
        site_packages = str(YOLO_VENV_SITE_PACKAGES)
        if site_packages not in sys.path:
            sys.path.insert(0, site_packages)

    try:
        from ultralytics import YOLO

        return YOLO
    except ModuleNotFoundError as exc:
        if exc.name != "ultralytics":
            raise

    candidates = [
        Path(DEFAULT_MODEL).expanduser().resolve().parents[1],
        Path(model_path).expanduser().resolve().parents[1],
    ]
    for candidate in candidates:
        if (candidate / "ultralytics" / "__init__.py").exists():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            from ultralytics import YOLO

            print(f"Using local Ultralytics source: {candidate_text}")
            return YOLO

    raise ModuleNotFoundError(
        "No module named 'ultralytics'. Install it with `pip install ultralytics` "
        "or place the Ultralytics source tree at /home/rzfly/ultralytics-8.3.39."
    )


class PixelKalmanFilter:
    """Constant-velocity Kalman filter for image-space target centers."""

    def __init__(self, process_noise=800.0, measurement_noise=25.0):
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.initialized = False
        self.last_time = None
        self.state = np.zeros((4, 1), dtype=np.float64)
        self.covariance = np.eye(4, dtype=np.float64) * 1000.0

    def reset(self):
        self.initialized = False
        self.last_time = None
        self.state.fill(0.0)
        self.covariance = np.eye(4, dtype=np.float64) * 1000.0

    def update(self, x, y, timestamp):
        if not self.initialized:
            self.state = np.array([[x], [y], [0.0], [0.0]], dtype=np.float64)
            self.covariance = np.diag([25.0, 25.0, 2500.0, 2500.0]).astype(np.float64)
            self.initialized = True
            self.last_time = timestamp
            return

        dt = max(1e-3, min(timestamp - self.last_time, 1.0))
        self.last_time = timestamp

        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = self.process_noise
        process = q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

        self.state = transition @ self.state
        self.covariance = transition @ self.covariance @ transition.T + process

        measurement = np.array([[x], [y]], dtype=np.float64)
        observation = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        noise = np.eye(2, dtype=np.float64) * self.measurement_noise
        residual = measurement - observation @ self.state
        residual_cov = observation @ self.covariance @ observation.T + noise
        gain = self.covariance @ observation.T @ np.linalg.inv(residual_cov)

        self.state = self.state + gain @ residual
        identity = np.eye(4, dtype=np.float64)
        self.covariance = (identity - gain @ observation) @ self.covariance

    def predict(self, lead_time):
        if not self.initialized:
            return None
        lead_time = max(0.0, float(lead_time))
        x = self.state[0, 0] + self.state[2, 0] * lead_time
        y = self.state[1, 0] + self.state[3, 0] * lead_time
        return float(x), float(y)


class YOLODetector:
    """Subscribes to Gazebo camera topic and runs YOLO detection in real-time."""

    def __init__(
        self,
        model_path,
        conf_threshold,
        topic,
        target_topic,
        publish_target,
        class_id,
        class_name,
        prediction_time,
        kalman_process_noise,
        kalman_measurement_noise,
    ):
        self.topic = topic
        self.conf_threshold = conf_threshold
        self.target_topic = target_topic
        self.publish_target = publish_target
        self.class_id = class_id
        self.class_name = class_name.lower() if class_name else None
        self.prediction_time = max(0.0, float(prediction_time))
        self.kalman = PixelKalmanFilter(kalman_process_noise, kalman_measurement_noise)
        self.frame_count = 0
        self.fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
        self._last_target_log_time = 0.0

        self.ros_node = None
        self.target_pub = None
        if self.publish_target:
            self.ros_node = rclpy.create_node("yolo_target_publisher")
            self.target_pub = self.ros_node.create_publisher(Point, self.target_topic, 10)
            print(f"Publishing YOLO target centers to {self.target_topic}")

        # Load YOLO model
        YOLO = import_yolo(model_path)
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        print(f"YOLO model loaded: {model_path} ({len(self.class_names)} classes)")

        # Create gz transport node and subscribe
        self.gz_node = GzNode()
        self.gz_node.subscribe(
            msg_type=image_pb2.Image,
            topic=self.topic,
            callback=self._image_callback,
        )
        print(f"Subscribed to {self.topic}, waiting for frames...")
        if self.prediction_time > 0.0:
            print(f"Kalman prediction enabled: lead_time={self.prediction_time:.3f}s")
        else:
            print("Kalman prediction disabled")

        # Latest frame storage (thread-safe via GIL for CPython)
        self._latest_frame = None
        self._frame_ready = False

    def close(self):
        if self.ros_node is not None:
            self.ros_node.destroy_node()
            self.ros_node = None

    def _image_callback(self, msg):
        """Called by gz.transport13 when a new Image message arrives."""
        # Validate pixel format (RGB_INT8 = 3)
        if msg.pixel_format_type != PIXEL_FORMAT_RGB_INT8:
            return

        w, h = msg.width, msg.height
        if w == 0 or h == 0:
            return

        # Convert raw bytes to numpy array
        # msg.data is bytes in RGB format, shape (h, w, 3)
        frame = np.frombuffer(msg.data, dtype=np.uint8).reshape((h, w, 3))

        # Store as BGR for OpenCV/YOLO
        self._latest_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self._frame_ready = True

    def _box_matches_filter(self, cls_id):
        if self.class_id is not None and cls_id != self.class_id:
            return False
        if self.class_name is not None:
            name = self._class_label(cls_id).lower()
            if self.class_name not in name:
                return False
        return True

    def _class_label(self, cls_id):
        if hasattr(self.class_names, "get"):
            return str(self.class_names.get(cls_id, cls_id))
        if 0 <= cls_id < len(self.class_names):
            return str(self.class_names[cls_id])
        return str(cls_id)

    def _extract_detections(self, results):
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                if not self._box_matches_filter(cls_id):
                    continue

                xyxy = (float(x1), float(y1), float(x2), float(y2))
                detections.append({
                    "xyxy": xyxy,
                    "center": ((float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0),
                    "conf": conf,
                    "cls_id": cls_id,
                })

        detections.sort(key=lambda item: item["conf"], reverse=True)
        return detections

    def _update_prediction(self, detection, frame_shape):
        if detection is None:
            return None

        cx, cy = detection["center"]
        now = time.time()
        self.kalman.update(cx, cy, now)
        predicted = self.kalman.predict(self.prediction_time)
        if predicted is None:
            return None

        frame_h, frame_w = frame_shape[:2]
        px = min(max(predicted[0], 0.0), max(0.0, frame_w - 1.0))
        py = min(max(predicted[1], 0.0), max(0.0, frame_h - 1.0))
        return px, py

    def _publish_target(self, detection, target_center):
        if not self.publish_target or self.target_pub is None or detection is None:
            return

        cx, cy = target_center if target_center is not None else detection["center"]
        msg = Point()
        msg.x = cx
        msg.y = cy
        msg.z = 0.0
        self.target_pub.publish(msg)

        now = time.time()
        if now - self._last_target_log_time >= 1.0:
            label = self._class_label(detection["cls_id"])
            print(
                f"Published target center: ({cx:.1f}, {cy:.1f}) "
                f"class={label} conf={detection['conf']:.2f}"
            )
            self._last_target_log_time = now

    def _annotate_frame(self, frame, detections, target_center):
        """Draw bounding boxes, labels, and selected target center."""
        for index, detection in enumerate(detections):
            x1, y1, x2, y2 = [int(value) for value in detection["xyxy"]]
            cx, cy = detection["center"]
            conf = detection["conf"]
            cls_id = detection["cls_id"]
            label = f"{self._class_label(cls_id)} {conf:.2f}"
            color = (0, 255, 255) if index == 0 else (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(cx), int(cy)), 5, color, -1)

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame, (x1, y1 - th - 4), (x1 + tw, y1), color, -1
            )
            cv2.putText(
                frame,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        if detections:
            cx, cy = detections[0]["center"]
            tx, ty = target_center if target_center is not None else detections[0]["center"]
            if target_center is not None:
                cv2.circle(frame, (int(tx), int(ty)), 7, (255, 0, 255), 2)
                cv2.line(
                    frame,
                    (int(cx), int(cy)),
                    (int(tx), int(ty)),
                    (255, 0, 255),
                    2,
                )
            cv2.putText(
                frame,
                f"detect: ({cx:.1f}, {cy:.1f})",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"laser target: ({tx:.1f}, {ty:.1f})",
                (10, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 255),
                2,
            )

        return frame

    def run(self):
        """Main loop: wait for frames, run YOLO, display results."""
        print("Starting detection loop. Press 'q' to quit.")

        while True:
            # Wait for a new frame
            if not self._frame_ready:
                time.sleep(0.001)  # 1ms poll -- avoid busy-wait
                continue

            self._frame_ready = False
            frame = self._latest_frame
            if frame is None:
                continue

            self.frame_count += 1

            # Run YOLO inference
            results = self.model(frame, conf=self.conf_threshold, verbose=False)
            detections = self._extract_detections(results)
            selected_detection = detections[0] if detections else None
            target_center = self._update_prediction(selected_detection, frame.shape)
            self._publish_target(selected_detection, target_center)

            # Annotate frame
            annotated = self._annotate_frame(frame.copy(), detections, target_center)

            # Compute and display FPS
            self._fps_frame_count += 1
            now = time.time()
            elapsed = now - self._last_fps_time
            if elapsed >= 1.0:
                self.fps = self._fps_frame_count / elapsed
                self._fps_frame_count = 0
                self._last_fps_time = now

            cv2.putText(
                annotated,
                f"FPS: {self.fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            # Display
            cv2.imshow("YOLO11 Detection - Ground Camera", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        print(f"Stopped after {self.frame_count} frames.")


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Real-time YOLO11n detection on Gazebo ground camera"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Path to YOLO model weights",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help="Confidence threshold",
    )
    parser.add_argument(
        "--topic",
        default=DEFAULT_TOPIC,
        help="Gazebo camera topic",
    )
    parser.add_argument(
        "--target-topic",
        default=DEFAULT_TARGET_TOPIC,
        help="ROS 2 topic used to publish detected target center",
    )
    parser.add_argument(
        "--no-publish-target",
        action="store_true",
        help="Only display detections; do not publish target center to ROS 2",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Only use detections with this YOLO class id",
    )
    parser.add_argument(
        "--class-name",
        default=None,
        help="Only use detections whose class name contains this text",
    )
    parser.add_argument(
        "--prediction-time",
        type=float,
        default=DEFAULT_PREDICTION_TIME,
        help="Seconds to predict target motion ahead before publishing to laser; use 0 to disable",
    )
    parser.add_argument(
        "--kalman-process-noise",
        type=float,
        default=800.0,
        help="Higher values make velocity estimates adapt faster",
    )
    parser.add_argument(
        "--kalman-measurement-noise",
        type=float,
        default=25.0,
        help="Higher values smooth noisier detections more strongly",
    )
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    detector = YOLODetector(
        args.model,
        args.conf,
        args.topic,
        args.target_topic,
        not args.no_publish_target,
        args.class_id,
        args.class_name,
        args.prediction_time,
        args.kalman_process_noise,
        args.kalman_measurement_noise,
    )
    try:
        detector.run()
    except KeyboardInterrupt:
        pass
    finally:
        detector.close()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
