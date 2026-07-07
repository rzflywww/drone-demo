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
import time

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
    ):
        self.topic = topic
        self.conf_threshold = conf_threshold
        self.target_topic = target_topic
        self.publish_target = publish_target
        self.class_id = class_id
        self.class_name = class_name.lower() if class_name else None
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
        from ultralytics import YOLO

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

    def _publish_target(self, detection):
        if not self.publish_target or self.target_pub is None or detection is None:
            return

        cx, cy = detection["center"]
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

    def _annotate_frame(self, frame, detections):
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
            cv2.putText(
                frame,
                f"laser target: ({cx:.1f}, {cy:.1f})",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
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
            self._publish_target(detections[0] if detections else None)

            # Annotate frame
            annotated = self._annotate_frame(frame.copy(), detections)

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
