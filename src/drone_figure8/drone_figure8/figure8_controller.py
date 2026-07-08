#!/usr/bin/env python3
"""Figure-8 trajectory controller for drone in Gazebo Harmonic.

Uses /gzserver/set_entity_state service (simulation_interfaces)
to move the quadcopter model along a lemniscate path.

Lemniscate of Gerono:
    x(t) = A * cos(ω*t)
    y(t) = A * sin(2*ω*t) / 2
    z(t) = H (constant altitude)
    yaw(t) = atan2(dy/dt, dx/dt)
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from simulation_interfaces.srv import SetEntityState


def euler_to_quaternion(roll, pitch, yaw):
    """Convert Euler angles (rad) to quaternion."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    return Quaternion(
        w=cr * cp * cy + sr * sp * sy,
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
    )


class Figure8Controller(Node):
    def __init__(self):
        super().__init__("figure8_controller")

        # 参数
        self.declare_parameter("amplitude", 3.0)
        self.declare_parameter("height", 2.0)
        self.declare_parameter("period", 12.0)
        self.declare_parameter("rate", 50.0)

        self.amplitude = float(self.get_parameter("amplitude").value)
        self.height = float(self.get_parameter("height").value)
        self.period = float(self.get_parameter("period").value)
        self.rate = float(self.get_parameter("rate").value)

        if self.amplitude <= 0.0:
            raise ValueError("amplitude must be greater than 0")
        if self.period <= 0.0:
            raise ValueError("period must be greater than 0")
        if self.rate <= 0.0:
            raise ValueError("rate must be greater than 0")

        self.omega = 2.0 * math.pi / self.period

        # 等待并创建 SetEntityState 客户端
        service_name = "/gzserver/set_entity_state"
        self.get_logger().info(f"Waiting for service: {service_name}")
        self.set_state_client = self.create_client(SetEntityState, service_name)
        if not self.set_state_client.wait_for_service(timeout_sec=30):
            self.get_logger().error(f"Service {service_name} not available!")
            raise RuntimeError(f"Service {service_name} not available")

        self.get_logger().info(f"Connected to {service_name}")

        # 定时器
        timer_period = 1.0 / self.rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.start_time = self.get_clock().now()
        self._pending_request = False

        self.get_logger().info(
            f"Figure-8 controller started: "
            f"amplitude={self.amplitude}m, height={self.height}m, "
            f"period={self.period}s, rate={self.rate}Hz"
        )

    def timer_callback(self):
        if self._pending_request:
            self.get_logger().debug("Skipping frame - previous request still pending")
            return

        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds * 1e-9

        # Lemniscate 轨迹
        x = self.amplitude * math.cos(self.omega * t)
        y = self.amplitude * math.sin(2.0 * self.omega * t) / 2.0
        z = self.height

        # 计算偏航角 (机头朝向运动切线方向)
        dx_dt = -self.amplitude * self.omega * math.sin(self.omega * t)
        dy_dt = self.amplitude * self.omega * math.cos(2.0 * self.omega * t)
        yaw = math.atan2(dy_dt, dx_dt)

        orientation = euler_to_quaternion(0.0, 0.0, yaw)

        # 构建请求
        req = SetEntityState.Request()
        req.entity = "quadcopter"

        req.state = req.state or type(req.state)()
        req.state.pose = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=orientation,
        )
        req.state.twist = req.state.twist or type(req.state.twist)()
        req.state.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
        req.state.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)

        self._pending_request = True
        future = self.set_state_client.call_async(req)
        future.add_done_callback(self._done_callback)

    def _done_callback(self, future):
        self._pending_request = False
        if future.exception() is not None:
            self.get_logger().error(f"Service call failed: {future.exception()}")


def main(args=None):
    rclpy.init(args=args)
    controller = Figure8Controller()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
