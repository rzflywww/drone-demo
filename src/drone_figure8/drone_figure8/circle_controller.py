#!/usr/bin/env python3
"""Circular trajectory controller for drone in Gazebo Harmonic.

Uses /gzserver/set_entity_state service (simulation_interfaces)
to move the quadcopter model along a horizontal circular path.

Circle:
    x(t) = center_x + R * cos(omega*t)
    y(t) = center_y +/- R * sin(omega*t)
    z(t) = H (constant altitude)
    yaw(t) = atan2(dy/dt, dx/dt)
"""

import math

import rclpy
from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from rclpy.node import Node
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


def get_bool_parameter(node, name):
    value = node.get_parameter(name).value
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


class CircleController(Node):
    def __init__(self):
        super().__init__("circle_controller")

        self.declare_parameter("radius", 3.0)
        self.declare_parameter("height", 2.0)
        self.declare_parameter("period", 12.0)
        self.declare_parameter("rate", 50.0)
        self.declare_parameter("center_x", 0.0)
        self.declare_parameter("center_y", 0.0)
        self.declare_parameter("clockwise", False)
        self.declare_parameter("entity_name", "quadcopter")

        self.radius = float(self.get_parameter("radius").value)
        self.height = float(self.get_parameter("height").value)
        self.period = float(self.get_parameter("period").value)
        self.rate = float(self.get_parameter("rate").value)
        self.center_x = float(self.get_parameter("center_x").value)
        self.center_y = float(self.get_parameter("center_y").value)
        self.clockwise = get_bool_parameter(self, "clockwise")
        self.entity_name = str(self.get_parameter("entity_name").value)

        if self.radius <= 0.0:
            raise ValueError("radius must be greater than 0")
        if self.period <= 0.0:
            raise ValueError("period must be greater than 0")
        if self.rate <= 0.0:
            raise ValueError("rate must be greater than 0")

        self.omega = 2.0 * math.pi / self.period
        self.direction = -1.0 if self.clockwise else 1.0

        service_name = "/gzserver/set_entity_state"
        self.get_logger().info(f"Waiting for service: {service_name}")
        self.set_state_client = self.create_client(SetEntityState, service_name)
        if not self.set_state_client.wait_for_service(timeout_sec=30):
            self.get_logger().error(f"Service {service_name} not available!")
            raise RuntimeError(f"Service {service_name} not available")

        self.get_logger().info(f"Connected to {service_name}")

        timer_period = 1.0 / self.rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.start_time = self.get_clock().now()
        self._pending_request = False

        direction_name = "clockwise" if self.clockwise else "counter-clockwise"
        self.get_logger().info(
            f"Circle controller started: radius={self.radius}m, "
            f"center=({self.center_x}, {self.center_y}), height={self.height}m, "
            f"period={self.period}s, rate={self.rate}Hz, direction={direction_name}"
        )

    def timer_callback(self):
        if self._pending_request:
            self.get_logger().debug("Skipping frame - previous request still pending")
            return

        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds * 1e-9
        phase = self.omega * t

        x = self.center_x + self.radius * math.cos(phase)
        y = self.center_y + self.direction * self.radius * math.sin(phase)
        z = self.height

        dx_dt = -self.radius * self.omega * math.sin(phase)
        dy_dt = self.direction * self.radius * self.omega * math.cos(phase)
        yaw = math.atan2(dy_dt, dx_dt)

        req = SetEntityState.Request()
        req.entity = self.entity_name

        req.state = req.state or type(req.state)()
        req.state.pose = Pose(
            position=Point(x=x, y=y, z=z),
            orientation=euler_to_quaternion(0.0, 0.0, yaw),
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
    controller = CircleController()
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
