"""Geometry helpers shared by countermeasure controllers."""

import math

from geometry_msgs.msg import Quaternion


def direction_to_quaternion(dx, dy, dz):
    """Return a quaternion that rotates local +Z onto a direction vector."""
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm < 1e-10:
        return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

    dx, dy, dz = dx / norm, dy / norm, dz / norm
    dot = dz
    if dot > 0.99999:
        return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    if dot < -0.99999:
        return Quaternion(w=0.0, x=1.0, y=0.0, z=0.0)

    axis_x = -dy
    axis_y = dx
    axis_norm = math.sqrt(axis_x * axis_x + axis_y * axis_y)
    axis_x /= axis_norm
    axis_y /= axis_norm
    half_angle = math.acos(max(-1.0, min(1.0, dot))) / 2.0
    scale = math.sin(half_angle)
    return Quaternion(
        w=math.cos(half_angle),
        x=scale * axis_x,
        y=scale * axis_y,
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


def camera_center_direction(camera_rpy):
    """Return the Gazebo camera sensor's local +X direction in world space."""
    _, pitch, yaw = camera_rpy
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return normalize((cy * cp, sy * cp, -sp))


def pixel_to_world_direction(
    u,
    v,
    image_width,
    image_height,
    horizontal_fov,
    camera_rpy,
):
    """Back-project an image pixel into a normalized world-space ray."""
    width = max(1.0, float(image_width))
    height = max(1.0, float(image_height))
    clamped_u = min(max(float(u), 0.0), width)
    clamped_v = min(max(float(v), 0.0), height)
    half_hfov_tan = math.tan(float(horizontal_fov) / 2.0)
    x_offset = ((clamped_u - width / 2.0) / (width / 2.0)) * half_hfov_tan
    y_offset = ((clamped_v - height / 2.0) / (width / 2.0)) * half_hfov_tan

    forward = camera_center_direction(camera_rpy)
    right = normalize(cross(forward, (0.0, 0.0, 1.0)))
    up = normalize(cross(right, forward))
    return normalize(
        (
            forward[0] + x_offset * right[0] - y_offset * up[0],
            forward[1] + x_offset * right[1] - y_offset * up[1],
            forward[2] + x_offset * right[2] - y_offset * up[2],
        )
    )
