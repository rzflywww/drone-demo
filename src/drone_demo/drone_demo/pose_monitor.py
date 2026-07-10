#!/usr/bin/env python3
"""Print a Gazebo entity's world pose from the world pose topic."""

import argparse
import math
import time

from gz.msgs10 import pose_v_pb2
from gz.transport13 import Node as GzNode


def quaternion_to_yaw(orientation):
    """Return yaw in radians from a protobuf quaternion."""
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class PoseMonitor:
    def __init__(self, entity_name, topic, rate, once):
        self.entity_name = entity_name
        self.topic = topic
        self.period = 1.0 / max(0.1, rate)
        self.once = once
        self.latest_pose = None
        self.latest_update = 0.0
        self.last_print = 0.0

        self.gz_node = GzNode()
        self.gz_node.subscribe(
            msg_type=pose_v_pb2.Pose_V,
            topic=self.topic,
            callback=self._on_pose_v,
        )

    def _on_pose_v(self, msg):
        for pose in msg.pose:
            if pose.name == self.entity_name:
                self.latest_pose = pose
                self.latest_update = time.monotonic()
                break

    def run(self):
        print(
            f"Listening for entity '{self.entity_name}' on {self.topic}. "
            "Press Ctrl+C to stop."
        )
        while True:
            now = time.monotonic()
            if self.latest_pose is None:
                if now - self.last_print >= 2.0:
                    print("Waiting for pose data...")
                    self.last_print = now
                time.sleep(0.05)
                continue

            if now - self.last_print < self.period:
                time.sleep(0.01)
                continue

            pose = self.latest_pose
            pos = pose.position
            yaw = quaternion_to_yaw(pose.orientation)
            age = now - self.latest_update
            print(
                f"{self.entity_name}: "
                f"x={pos.x:8.3f} m  y={pos.y:8.3f} m  z={pos.z:8.3f} m  "
                f"yaw={math.degrees(yaw):7.2f} deg  age={age:4.2f}s",
                flush=True,
            )
            self.last_print = now

            if self.once:
                return


def main():
    parser = argparse.ArgumentParser(
        description="Print a Gazebo entity's world coordinates."
    )
    parser.add_argument(
        "--entity",
        default="quadcopter",
        help="Gazebo entity name to monitor",
    )
    parser.add_argument(
        "--world",
        default="drone_world",
        help="Gazebo world name, used to build the default pose topic",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Gazebo Pose_V topic. Defaults to /world/<world>/pose/info",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=2.0,
        help="Print rate in Hz",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one pose and exit",
    )
    args = parser.parse_args()

    topic = args.topic or f"/world/{args.world}/pose/info"
    monitor = PoseMonitor(args.entity, topic, args.rate, args.once)
    try:
        monitor.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
