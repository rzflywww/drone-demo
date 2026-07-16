#!/usr/bin/env python3
"""Aim a simulated radio-jamming cone from LLaVA target pixels."""

import time

import rclpy
from geometry_msgs.msg import Point, Pose, Vector3
from gz.msgs10 import pose_v_pb2
from gz.transport13 import Node as GzNode
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from simulation_interfaces.srv import SetEntityState
from std_msgs.msg import Bool

from drone_demo.scene_config import scene_defaults_from_sdf
from drone_demo.target_geometry import (
    direction_to_quaternion,
    normalize,
    pixel_to_world_direction,
)


IDLE = "idle"
JAMMING = "jamming"
SUCCESS = "success"


class ContinuousJammingTimer:
    """Latch success after a target has been continuously jammed long enough."""

    def __init__(self, duration):
        self.duration = max(0.0, float(duration))
        self.started_at = None
        self.succeeded = False

    def update(self, covered, now):
        if self.succeeded:
            return SUCCESS
        if not covered:
            self.started_at = None
            return IDLE
        if self.started_at is None:
            self.started_at = float(now)
        if float(now) - self.started_at >= self.duration:
            self.succeeded = True
            return SUCCESS
        return JAMMING

    def elapsed(self, now):
        if self.started_at is None:
            return 0.0
        return max(0.0, min(float(now) - self.started_at, self.duration))


def point_inside_cone(point, apex, axis, length, end_radius):
    """Return whether a world-space point is inside an apex-origin cone."""
    length = max(0.0, float(length))
    end_radius = max(0.0, float(end_radius))
    if length <= 0.0:
        return False

    axis = normalize(axis)
    relative = tuple(float(point[i]) - float(apex[i]) for i in range(3))
    axial_distance = sum(relative[i] * axis[i] for i in range(3))
    if axial_distance < 0.0 or axial_distance > length:
        return False

    distance_squared = sum(value * value for value in relative)
    radial_squared = max(0.0, distance_squared - axial_distance * axial_distance)
    allowed_radius = end_radius * axial_distance / length
    return radial_squared <= allowed_radius * allowed_radius


class RadioJammingController(Node):
    def __init__(self):
        super().__init__("radio_jamming_controller")

        numeric_parameter = ParameterDescriptor(dynamic_typing=True)
        scene_defaults = scene_defaults_from_sdf()
        self.declare_parameter("rate", 20.0, numeric_parameter)
        self.declare_parameter("jamming_duration", 5.0, numeric_parameter)
        self.declare_parameter("jamming_aim_distance", 15.0, numeric_parameter)
        self.declare_parameter("jamming_cone_length", 22.0, numeric_parameter)
        self.declare_parameter("jamming_cone_radius", 4.5, numeric_parameter)
        self.declare_parameter("pose_timeout", 0.5, numeric_parameter)
        self.declare_parameter("target_topic", "/countermeasure_target_pixel")
        self.declare_parameter("success_topic", "/radio_jamming/success")
        self.declare_parameter("beam_entity", "radio_jamming_beam")
        self.declare_parameter("target_entity", "quadcopter")
        self.declare_parameter("pose_topic", "/world/drone_world/pose/info")
        self.declare_parameter(
            "image_width", scene_defaults["image_width"], numeric_parameter
        )
        self.declare_parameter(
            "image_height", scene_defaults["image_height"], numeric_parameter
        )
        self.declare_parameter(
            "horizontal_fov", scene_defaults["horizontal_fov"], numeric_parameter
        )
        self.declare_parameter("camera_x", scene_defaults["camera_x"], numeric_parameter)
        self.declare_parameter("camera_y", scene_defaults["camera_y"], numeric_parameter)
        self.declare_parameter("camera_z", scene_defaults["camera_z"], numeric_parameter)
        self.declare_parameter(
            "camera_roll", scene_defaults["camera_roll"], numeric_parameter
        )
        self.declare_parameter(
            "camera_pitch", scene_defaults["camera_pitch"], numeric_parameter
        )
        self.declare_parameter(
            "camera_yaw", scene_defaults["camera_yaw"], numeric_parameter
        )
        self.declare_parameter("jammer_x", scene_defaults["jammer_x"], numeric_parameter)
        self.declare_parameter("jammer_y", scene_defaults["jammer_y"], numeric_parameter)
        self.declare_parameter("jammer_z", scene_defaults["jammer_z"], numeric_parameter)

        self.rate = max(0.1, float(self.get_parameter("rate").value))
        self.aim_distance = max(
            0.1,
            float(self.get_parameter("jamming_aim_distance").value),
        )
        self.cone_length = max(
            0.1,
            float(self.get_parameter("jamming_cone_length").value),
        )
        self.cone_radius = max(
            0.0,
            float(self.get_parameter("jamming_cone_radius").value),
        )
        self.pose_timeout = max(
            0.0,
            float(self.get_parameter("pose_timeout").value),
        )
        self.image_width = float(self.get_parameter("image_width").value)
        self.image_height = float(self.get_parameter("image_height").value)
        self.horizontal_fov = float(self.get_parameter("horizontal_fov").value)
        self.camera_pos = tuple(
            float(self.get_parameter(name).value)
            for name in ("camera_x", "camera_y", "camera_z")
        )
        self.camera_rpy = tuple(
            float(self.get_parameter(name).value)
            for name in ("camera_roll", "camera_pitch", "camera_yaw")
        )
        self.jammer_pos = tuple(
            float(self.get_parameter(name).value)
            for name in ("jammer_x", "jammer_y", "jammer_z")
        )
        self.beam_entity = str(self.get_parameter("beam_entity").value)
        self.target_entity = str(self.get_parameter("target_entity").value)
        pose_topic = str(self.get_parameter("pose_topic").value)
        target_topic = str(self.get_parameter("target_topic").value)
        success_topic = str(self.get_parameter("success_topic").value)
        self.jamming_timer = ContinuousJammingTimer(
            self.get_parameter("jamming_duration").value
        )

        self.target_x = self.image_width / 2.0
        self.target_y = self.image_height / 2.0
        self._target_received = False
        self._drone_pos = None
        self._last_drone_pose_time = 0.0
        self._pending_request = False
        self._success_published = False
        self._last_log_time = 0.0

        self.create_subscription(Point, target_topic, self._on_target_pixel, 10)
        self.gz_node = GzNode()
        self.gz_node.subscribe(
            msg_type=pose_v_pb2.Pose_V,
            topic=pose_topic,
            callback=self._on_pose_v,
        )
        success_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.success_pub = self.create_publisher(Bool, success_topic, success_qos)

        service_name = "/gzserver/set_entity_state"
        self.set_state_client = self.create_client(SetEntityState, service_name)
        if not self.set_state_client.wait_for_service(timeout_sec=30):
            raise RuntimeError(f"Service {service_name} not available")

        self.timer = self.create_timer(1.0 / self.rate, self.timer_callback)
        self.get_logger().info(
            f"Radio jamming ready: target={target_topic}, "
            f"aim_distance={self.aim_distance:.1f}m, "
            f"cone={self.cone_length:.1f}x{self.cone_radius:.1f}m, "
            f"duration={self.jamming_timer.duration:.1f}s, "
            f"pose={pose_topic}:{self.target_entity}"
        )

    def _on_target_pixel(self, msg):
        self.target_x = float(msg.x)
        self.target_y = float(msg.y)
        self._target_received = True

    def _on_pose_v(self, msg):
        for pose in msg.pose:
            if pose.name == self.target_entity:
                self._drone_pos = (
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                )
                self._last_drone_pose_time = time.monotonic()
                break

    def fixed_distance_aim_point(self):
        ray = pixel_to_world_direction(
            self.target_x,
            self.target_y,
            self.image_width,
            self.image_height,
            self.horizontal_fov,
            self.camera_rpy,
        )
        return tuple(
            self.camera_pos[index] + ray[index] * self.aim_distance
            for index in range(3)
        )

    def timer_callback(self):
        now = time.monotonic()
        if not self._target_received:
            return

        aim_point = self.fixed_distance_aim_point()
        direction = normalize(
            tuple(aim_point[index] - self.jammer_pos[index] for index in range(3))
        )
        pose_is_fresh = (
            self._drone_pos is not None
            and now - self._last_drone_pose_time <= self.pose_timeout
        )
        covered = pose_is_fresh and point_inside_cone(
            self._drone_pos,
            self.jammer_pos,
            direction,
            self.cone_length,
            self.cone_radius,
        )
        state = self.jamming_timer.update(covered, now)

        if state == SUCCESS and not self._success_published:
            self.success_pub.publish(Bool(data=True))
            self._success_published = True
            self.get_logger().warn(
                f"Radio jamming succeeded after {self.jamming_timer.duration:.1f}s "
                "of continuous geometric coverage"
            )

        if now - self._last_log_time >= 1.0 and state != SUCCESS:
            if covered:
                elapsed = self.jamming_timer.elapsed(now)
                message = (
                    f"Target inside cone: {elapsed:.1f}/"
                    f"{self.jamming_timer.duration:.1f}s"
                )
            elif pose_is_fresh:
                message = "Target outside cone; continuous timer reset"
            else:
                message = "Waiting for a fresh Gazebo target pose; timer reset"
            self.get_logger().info(message)
            self._last_log_time = now

        if self._pending_request:
            return

        request = SetEntityState.Request()
        request.entity = self.beam_entity
        request.state = request.state or type(request.state)()
        request.state.pose = Pose(
            position=Point(
                x=self.jammer_pos[0],
                y=self.jammer_pos[1],
                z=self.jammer_pos[2],
            ),
            orientation=direction_to_quaternion(*direction),
        )
        request.state.twist = request.state.twist or type(request.state.twist)()
        request.state.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
        request.state.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)

        self._pending_request = True
        future = self.set_state_client.call_async(request)
        future.add_done_callback(self._on_set_done)

    def _on_set_done(self, future):
        self._pending_request = False
        if future.exception() is not None:
            self.get_logger().error(f"SetEntityState failed: {future.exception()}")

    def hide_beam(self):
        self.timer.cancel()
        request = SetEntityState.Request()
        request.entity = self.beam_entity
        request.state = request.state or type(request.state)()
        request.state.pose = Pose(
            position=Point(x=0.0, y=0.0, z=-100.0),
            orientation=direction_to_quaternion(0.0, 0.0, 1.0),
        )
        request.state.twist = request.state.twist or type(request.state.twist)()
        self.set_state_client.call_async(request)
        rclpy.spin_once(self, timeout_sec=0.1)


def main(args=None):
    # Keep the context alive during Ctrl+C cleanup so the beam can be hidden.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    controller = RadioJammingController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.hide_beam()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
