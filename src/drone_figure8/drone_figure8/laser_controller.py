#!/usr/bin/env python3
"""激光指示器控制器 - 从地面摄像机指向图像中的像素位置。

独立节点，按需启动。输入地面摄像机画面中的像素坐标，例如 1280x720
画面的中心点 (640, 360)，节点会把该像素反投影成世界坐标系下的射线，
然后用 SetEntityState 将 laser_beam 模型定位到摄像机处并指向该方向。

用法（在另一个终端中执行）:
    ros2 run drone_figure8 laser_controller --ros-args -p target_x:=640 -p target_y:=360
    ros2 topic pub --once /laser_target_pixel geometry_msgs/msg/Point "{x: 640.0, y: 360.0, z: 0.0}"
    # Ctrl+C 停止后激光束会被隐藏
"""

import math

import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from simulation_interfaces.srv import SetEntityState


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

        self.declare_parameter("target_x", 640.0, numeric_parameter)
        self.declare_parameter("target_y", 360.0, numeric_parameter)
        self.declare_parameter("image_width", 1280.0, numeric_parameter)
        self.declare_parameter("image_height", 720.0, numeric_parameter)
        self.declare_parameter("horizontal_fov", 1.047, numeric_parameter)
        self.declare_parameter("target_y_offset", 0.0, numeric_parameter)

        # 摄像机镜头世界坐标。默认值与 worlds/drone_world.sdf 中 ground_camera 对应。
        self.declare_parameter("camera_x", 7.8925, numeric_parameter)
        self.declare_parameter("camera_y", -7.8925, numeric_parameter)
        self.declare_parameter("camera_z", 1.5434, numeric_parameter)

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
        self.camera_pos = (
            float(self.get_parameter("camera_x").value),
            float(self.get_parameter("camera_y").value),
            float(self.get_parameter("camera_z").value),
        )
        self.camera_rpy = (
            float(self.get_parameter("camera_roll").value),
            float(self.get_parameter("camera_pitch").value),
            float(self.get_parameter("camera_yaw").value),
        )
        self._last_clamped_target = None

        self.add_on_set_parameters_callback(self._on_parameters_set)
        self.create_subscription(Point, "laser_target_pixel", self._on_target_pixel, 10)

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
            f"image={self.image_width:.0f}x{self.image_height:.0f}, "
            f"target=({self.target_x:.1f}, {self.target_y:.1f}), "
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
            elif parameter.name == "camera_x":
                self.camera_pos = (float(parameter.value), self.camera_pos[1], self.camera_pos[2])
            elif parameter.name == "camera_y":
                self.camera_pos = (self.camera_pos[0], float(parameter.value), self.camera_pos[2])
            elif parameter.name == "camera_z":
                self.camera_pos = (self.camera_pos[0], self.camera_pos[1], float(parameter.value))
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
        self.get_logger().info(
            f"Laser target pixel updated from topic: ({self.target_x:.1f}, {self.target_y:.1f})"
        )

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

    def timer_callback(self):
        if self._pending:
            return

        cam = self.camera_pos
        direction = self.pixel_to_world_direction(self.target_x, self.target_y)
        laser_orient = direction_to_quaternion(*direction)

        # 更新激光束位姿
        set_req = SetEntityState.Request()
        set_req.entity = "laser_beam"

        set_req.state = set_req.state or type(set_req.state)()
        set_req.state.pose = Pose(
            position=Point(x=cam[0], y=cam[1], z=cam[2]),
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
