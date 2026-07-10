#!/usr/bin/env python3
"""激光指示器控制器 - 从武器平台指向地面摄像机图像中的像素位置。

独立节点，按需启动。输入地面摄像机画面中的像素坐标，例如 640x360
画面的中心点 (320, 180)，节点会把该像素反投影成世界坐标系下的射线，
优先使用深度图把该像素还原成世界坐标，再让武器平台炮口的 laser_beam 指向该点。

用法（在另一个终端中执行）:
    ros2 run drone_figure8 laser_controller --ros-args -p target_x:=320 -p target_y:=180
    ros2 topic pub --once /laser_target_pixel geometry_msgs/msg/Point "{x: 320.0, y: 180.0, z: 0.0}"
    # Ctrl+C 停止后激光束会被隐藏
"""

import math
import time

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from gz.msgs10 import image_pb2
from gz.transport13 import Node as GzNode
from simulation_interfaces.srv import SetEntityState


PIXEL_FORMAT_R_FLOAT32 = 13


def direction_to_quaternion(dx, dy, dz):
    """将方向向量转为四元数，使局部 +Z 轴指向该方向。

    给定方向 (dx, dy, dz)，返回一个四元数，其旋转将
    局部 Z 轴 (0,0,1) 映射到该单位方向向量上。
    """
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < 1e-10:
        return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

    dx, dy, dz = dx / norm, dy / norm, dz / norm

    # z_hat 与 direction 的点积
    dot = dz  # z_hat = (0,0,1)

    if dot > 0.99999:
        # 已指向 +Z
        return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

    if dot < -0.99999:
        # 指向 -Z，绕 X 轴旋转 180°
        return Quaternion(w=0.0, x=1.0, y=0.0, z=0.0)

    # 一般情况：旋转轴 = z_hat × d = (-dy, dx, 0)
    axis_x = -dy
    axis_y = dx
    axis_norm = math.sqrt(axis_x * axis_x + axis_y * axis_y)
    axis_x /= axis_norm
    axis_y /= axis_norm

    angle = math.acos(max(-1.0, min(1.0, dot)))
    half = angle / 2.0
    w = math.cos(half)
    s = math.sin(half)

    return Quaternion(
        w=w,
        x=s * axis_x,
        y=s * axis_y,
        z=0.0,
    )


def normalize(vector):
    norm = math.sqrt(sum(value * value for value in vector))
    if norm < 1e-10:
        raise ValueError("Cannot normalize a near-zero vector")
    return tuple(value / norm for value in vector)


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


class LaserController(Node):
    def __init__(self):
        super().__init__("laser_controller")

        numeric_parameter = ParameterDescriptor(dynamic_typing=True)

        self.declare_parameter("rate", 50.0, numeric_parameter)
        self.rate = self.get_parameter("rate").value

        self.declare_parameter("target_x", 320.0, numeric_parameter)
        self.declare_parameter("target_y", 180.0, numeric_parameter)
        self.declare_parameter("image_width", 640.0, numeric_parameter)
        self.declare_parameter("image_height", 360.0, numeric_parameter)
        self.declare_parameter("horizontal_fov", 1.047, numeric_parameter)
        self.declare_parameter("target_y_offset", 0.0, numeric_parameter)
        self.declare_parameter("laser_aim_distance", 15.0, numeric_parameter)
        self.declare_parameter("use_depth_camera", True, numeric_parameter)
        self.declare_parameter("depth_topic", "/ground_camera/depth")
        self.declare_parameter("depth_timeout", 0.5, numeric_parameter)
        self.declare_parameter("depth_sample_radius", 2, numeric_parameter)
        self.declare_parameter("depth_is_range", False, numeric_parameter)
        self.declare_parameter("log_estimated_world", True, numeric_parameter)
        self.declare_parameter("estimated_world_log_rate", 2.0, numeric_parameter)

        # 摄像机镜头世界坐标。默认值与 worlds/drone_world.sdf 中 ground_camera 对应。
        self.declare_parameter("camera_x", 7.8925, numeric_parameter)
        self.declare_parameter("camera_y", -7.8925, numeric_parameter)
        self.declare_parameter("camera_z", 1.5434, numeric_parameter)

        # 武器平台炮口世界坐标。默认值与 worlds/drone_world.sdf 中 weapon_platform 对应。
        self.declare_parameter("weapon_x", 7.9601, numeric_parameter)
        self.declare_parameter("weapon_y", -7.4600, numeric_parameter)
        self.declare_parameter("weapon_z", 1.5000, numeric_parameter)

        # 摄像机模型姿态。默认值与 worlds/drone_world.sdf 中 ground_camera 的 pose 对应。
        self.declare_parameter("camera_roll", 0.0, numeric_parameter)
        self.declare_parameter("camera_pitch", 0.044, numeric_parameter)
        self.declare_parameter("camera_yaw", 2.356, numeric_parameter)

        self.target_x = float(self.get_parameter("target_x").value)
        self.target_y = float(self.get_parameter("target_y").value)
        self.image_width = float(self.get_parameter("image_width").value)
        self.image_height = float(self.get_parameter("image_height").value)
        self.horizontal_fov = float(self.get_parameter("horizontal_fov").value)
        self.target_y_offset = float(self.get_parameter("target_y_offset").value)
        self.laser_aim_distance = float(self.get_parameter("laser_aim_distance").value)
        self.use_depth_camera = bool(self.get_parameter("use_depth_camera").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.depth_timeout = float(self.get_parameter("depth_timeout").value)
        self.depth_sample_radius = int(self.get_parameter("depth_sample_radius").value)
        self.depth_is_range = bool(self.get_parameter("depth_is_range").value)
        self.log_estimated_world = bool(self.get_parameter("log_estimated_world").value)
        self.estimated_world_log_rate = float(
            self.get_parameter("estimated_world_log_rate").value
        )
        self.camera_pos = (
            float(self.get_parameter("camera_x").value),
            float(self.get_parameter("camera_y").value),
            float(self.get_parameter("camera_z").value),
        )
        self.weapon_pos = (
            float(self.get_parameter("weapon_x").value),
            float(self.get_parameter("weapon_y").value),
            float(self.get_parameter("weapon_z").value),
        )
        self.camera_rpy = (
            float(self.get_parameter("camera_roll").value),
            float(self.get_parameter("camera_pitch").value),
            float(self.get_parameter("camera_yaw").value),
        )
        self._last_clamped_target = None
        self._latest_depth = None
        self._depth_width = 0
        self._depth_height = 0
        self._last_depth_time = 0.0
        self._last_depth_warn_time = 0.0
        self._last_depth_format_warn_time = 0.0
        self._last_estimated_world_log_time = 0.0
        self._last_aim_source = "unknown"
        self._last_aim_depth = None
        self._last_aim_ray_distance = None

        self.add_on_set_parameters_callback(self._on_parameters_set)
        self.create_subscription(Point, "laser_target_pixel", self._on_target_pixel, 10)

        self.gz_node = None
        if self.use_depth_camera:
            self.gz_node = GzNode()
            self.gz_node.subscribe(
                msg_type=image_pb2.Image,
                topic=self.depth_topic,
                callback=self._on_depth_image,
            )
            self.get_logger().info(f"Subscribed to Gazebo depth topic: {self.depth_topic}")

        # 创建 SetEntityState 客户端（更新激光束位姿）
        set_service = "/gzserver/set_entity_state"
        self.set_state_client = self.create_client(SetEntityState, set_service)
        if not self.set_state_client.wait_for_service(timeout_sec=30):
            self.get_logger().error(f"Service {set_service} not available!")
            raise RuntimeError(f"Service {set_service} not available")
        self.get_logger().info(f"Connected to {set_service}")

        # 定时器
        timer_period = 1.0 / self.rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self._pending = False

        self.get_logger().info(
            f"Laser controller started: "
            f"camera=({self.camera_pos[0]}, {self.camera_pos[1]}, {self.camera_pos[2]}), "
            f"weapon=({self.weapon_pos[0]}, {self.weapon_pos[1]}, {self.weapon_pos[2]}), "
            f"image={self.image_width:.0f}x{self.image_height:.0f}, "
            f"target=({self.target_x:.1f}, {self.target_y:.1f}), "
            f"depth={'enabled' if self.use_depth_camera else 'disabled'}, "
            f"estimated_world_log_rate={self.estimated_world_log_rate}Hz, "
            f"rate={self.rate}Hz"
        )

    def _on_parameters_set(self, parameters):
        for parameter in parameters:
            if parameter.name == "target_x":
                self.target_x = float(parameter.value)
            elif parameter.name == "target_y":
                self.target_y = float(parameter.value)
            elif parameter.name == "image_width":
                self.image_width = float(parameter.value)
            elif parameter.name == "image_height":
                self.image_height = float(parameter.value)
            elif parameter.name == "horizontal_fov":
                self.horizontal_fov = float(parameter.value)
            elif parameter.name == "target_y_offset":
                self.target_y_offset = float(parameter.value)
            elif parameter.name == "laser_aim_distance":
                self.laser_aim_distance = float(parameter.value)
            elif parameter.name == "use_depth_camera":
                self.use_depth_camera = bool(parameter.value)
            elif parameter.name == "depth_topic":
                self.depth_topic = str(parameter.value)
            elif parameter.name == "depth_timeout":
                self.depth_timeout = float(parameter.value)
            elif parameter.name == "depth_sample_radius":
                self.depth_sample_radius = int(parameter.value)
            elif parameter.name == "depth_is_range":
                self.depth_is_range = bool(parameter.value)
            elif parameter.name == "log_estimated_world":
                self.log_estimated_world = bool(parameter.value)
            elif parameter.name == "estimated_world_log_rate":
                self.estimated_world_log_rate = float(parameter.value)
            elif parameter.name == "camera_x":
                self.camera_pos = (float(parameter.value), self.camera_pos[1], self.camera_pos[2])
            elif parameter.name == "camera_y":
                self.camera_pos = (self.camera_pos[0], float(parameter.value), self.camera_pos[2])
            elif parameter.name == "camera_z":
                self.camera_pos = (self.camera_pos[0], self.camera_pos[1], float(parameter.value))
            elif parameter.name == "weapon_x":
                self.weapon_pos = (float(parameter.value), self.weapon_pos[1], self.weapon_pos[2])
            elif parameter.name == "weapon_y":
                self.weapon_pos = (self.weapon_pos[0], float(parameter.value), self.weapon_pos[2])
            elif parameter.name == "weapon_z":
                self.weapon_pos = (self.weapon_pos[0], self.weapon_pos[1], float(parameter.value))
            elif parameter.name == "camera_roll":
                self.camera_rpy = (float(parameter.value), self.camera_rpy[1], self.camera_rpy[2])
            elif parameter.name == "camera_pitch":
                self.camera_rpy = (self.camera_rpy[0], float(parameter.value), self.camera_rpy[2])
            elif parameter.name == "camera_yaw":
                self.camera_rpy = (self.camera_rpy[0], self.camera_rpy[1], float(parameter.value))

        return SetParametersResult(successful=True)

    def _on_target_pixel(self, msg):
        self.target_x = float(msg.x)
        self.target_y = float(msg.y)

    def _on_depth_image(self, msg):
        """Store the latest Gazebo depth image in meters."""
        if msg.pixel_format_type != PIXEL_FORMAT_R_FLOAT32:
            now = time.monotonic()
            if now - self._last_depth_format_warn_time >= 2.0:
                self.get_logger().warn(
                    f"Depth image pixel format {msg.pixel_format_type} is unsupported; "
                    "expected R_FLOAT32"
                )
                self._last_depth_format_warn_time = now
            return

        width, height = int(msg.width), int(msg.height)
        if width <= 0 or height <= 0:
            return

        depth_values = np.frombuffer(msg.data, dtype=np.float32)
        row_stride = int(msg.step // 4) if msg.step else width
        if row_stride < width or depth_values.size < row_stride * height:
            if depth_values.size < width * height:
                return
            row_stride = width

        depth = depth_values.reshape((height, row_stride))[:, :width].copy()
        self._latest_depth = depth
        self._depth_width = width
        self._depth_height = height
        self._last_depth_time = time.monotonic()

    def camera_center_direction(self):
        _, pitch, yaw = self.camera_rpy

        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cy = math.cos(yaw)
        sy = math.sin(yaw)

        # Gazebo/SDF camera looks along the sensor frame +X axis.
        return normalize((
            cy * cp,
            sy * cp,
            -sp,
        ))

    def pixel_to_world_direction(self, u, v):
        width = max(1.0, self.image_width)
        height = max(1.0, self.image_height)
        half_hfov_tan = math.tan(self.horizontal_fov / 2.0)

        clamped_u = min(max(float(u), 0.0), width)
        clamped_v = min(max(float(v) + self.target_y_offset, 0.0), height)
        clamped_target = (clamped_u, clamped_v)
        effective_target = (float(u), float(v) + self.target_y_offset)
        if clamped_target != effective_target and clamped_target != self._last_clamped_target:
            self.get_logger().warn(
                f"Target pixel ({effective_target[0]:.1f}, {effective_target[1]:.1f}) is outside "
                f"{width:.0f}x{height:.0f}; using ({clamped_u:.1f}, {clamped_v:.1f})"
            )
            self._last_clamped_target = clamped_target

        x_offset = ((clamped_u - width / 2.0) / (width / 2.0)) * half_hfov_tan
        y_offset = ((clamped_v - height / 2.0) / (width / 2.0)) * half_hfov_tan

        forward = self.camera_center_direction()
        world_up = (0.0, 0.0, 1.0)
        right = normalize(cross(forward, world_up))
        up = normalize(cross(right, forward))

        return normalize((
            forward[0] + x_offset * right[0] - y_offset * up[0],
            forward[1] + x_offset * right[1] - y_offset * up[1],
            forward[2] + x_offset * right[2] - y_offset * up[2],
        ))

    def depth_at_pixel(self, u, v):
        if not self.use_depth_camera or self._latest_depth is None:
            return None

        now = time.monotonic()
        if now - self._last_depth_time > max(0.0, self.depth_timeout):
            self._warn_depth_fallback("depth image timed out")
            return None

        if self._depth_width <= 0 or self._depth_height <= 0:
            return None

        x = int(round(min(max(float(u), 0.0), self._depth_width - 1)))
        y = int(round(min(max(float(v) + self.target_y_offset, 0.0), self._depth_height - 1)))
        radius = max(0, int(self.depth_sample_radius))
        x0 = max(0, x - radius)
        x1 = min(self._depth_width, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(self._depth_height, y + radius + 1)

        samples = self._latest_depth[y0:y1, x0:x1]
        valid = samples[np.isfinite(samples) & (samples > 0.0)]
        if valid.size == 0:
            self._warn_depth_fallback("no valid depth at target pixel")
            return None

        return float(np.median(valid))

    def target_world_position(self):
        cam = self.camera_pos
        camera_direction = self.pixel_to_world_direction(self.target_x, self.target_y)
        depth = self.depth_at_pixel(self.target_x, self.target_y)
        if depth is None:
            aim_distance = max(0.1, self.laser_aim_distance)
            self._last_aim_source = "fallback"
            self._last_aim_depth = None
            self._last_aim_ray_distance = aim_distance
            return (
                cam[0] + camera_direction[0] * aim_distance,
                cam[1] + camera_direction[1] * aim_distance,
                cam[2] + camera_direction[2] * aim_distance,
            )

        if self.depth_is_range:
            ray_distance = depth
        else:
            forward = self.camera_center_direction()
            cos_angle = max(1e-6, sum(camera_direction[i] * forward[i] for i in range(3)))
            ray_distance = depth / cos_angle

        self._last_aim_source = "depth"
        self._last_aim_depth = depth
        self._last_aim_ray_distance = ray_distance
        return (
            cam[0] + camera_direction[0] * ray_distance,
            cam[1] + camera_direction[1] * ray_distance,
            cam[2] + camera_direction[2] * ray_distance,
        )

    def log_estimated_world_position(self, aim_point):
        if not self.log_estimated_world:
            return

        now = time.monotonic()
        log_rate = max(0.1, float(self.estimated_world_log_rate))
        if now - self._last_estimated_world_log_time < 1.0 / log_rate:
            return

        if self._last_aim_depth is None:
            depth_text = "depth=none"
        else:
            depth_text = f"depth={self._last_aim_depth:.3f}m"

        self.get_logger().info(
            "estimated_target_world: "
            f"x={aim_point[0]:.3f}m, y={aim_point[1]:.3f}m, z={aim_point[2]:.3f}m, "
            f"pixel=({self.target_x:.1f}, {self.target_y:.1f}), "
            f"{depth_text}, ray_distance={self._last_aim_ray_distance:.3f}m, "
            f"source={self._last_aim_source}"
        )
        self._last_estimated_world_log_time = now

    def _warn_depth_fallback(self, reason):
        now = time.monotonic()
        if now - self._last_depth_warn_time < 2.0:
            return
        self.get_logger().warn(
            f"Using fallback laser_aim_distance={self.laser_aim_distance:.2f}m: {reason}"
        )
        self._last_depth_warn_time = now

    def timer_callback(self):
        if self._pending:
            return

        weapon = self.weapon_pos
        aim_point = self.target_world_position()
        self.log_estimated_world_position(aim_point)
        laser_direction = normalize((
            aim_point[0] - weapon[0],
            aim_point[1] - weapon[1],
            aim_point[2] - weapon[2],
        ))
        laser_orient = direction_to_quaternion(*laser_direction)

        # 更新激光束位姿
        set_req = SetEntityState.Request()
        set_req.entity = "laser_beam"

        set_req.state = set_req.state or type(set_req.state)()
        set_req.state.pose = Pose(
            position=Point(x=weapon[0], y=weapon[1], z=weapon[2]),
            orientation=laser_orient,
        )
        set_req.state.twist = set_req.state.twist or type(set_req.state.twist)()
        set_req.state.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
        set_req.state.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)

        self._pending = True
        set_future = self.set_state_client.call_async(set_req)
        set_future.add_done_callback(self._on_set_done)

    def _on_set_done(self, future):
        self._pending = False
        if future.exception() is not None:
            self.get_logger().error(f"SetEntityState failed: {future.exception()}")

    def hide_laser(self):
        """将激光束移到地面以下使其不可见。"""
        req = SetEntityState.Request()
        req.entity = "laser_beam"
        req.state = req.state or type(req.state)()
        req.state.pose = Pose(
            position=Point(x=0.0, y=0.0, z=-100.0),
            orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
        )
        req.state.twist = req.state.twist or type(req.state.twist)()
        self.set_state_client.call_async(req)
        # 给异步调用一点时间发送出去
        rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Laser hidden")


def main(args=None):
    rclpy.init(args=args)
    controller = LaserController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.hide_laser()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
