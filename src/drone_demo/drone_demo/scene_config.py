"""Scene-derived defaults shared by controller nodes."""

import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path


FALLBACK_SCENE_DEFAULTS = {
    "camera_x": 7.8925,
    "camera_y": -7.8925,
    "camera_z": 1.5434,
    "camera_roll": 0.0,
    "camera_pitch": 0.044,
    "camera_yaw": 2.356,
    "weapon_x": 7.9601,
    "weapon_y": -7.4600,
    "weapon_z": 1.5000,
}


def parse_sdf_pose(text):
    values = [float(value) for value in text.split()] if text else []
    values.extend([0.0] * (6 - len(values)))
    return tuple(values[:6])


def rotation_matrix(roll, pitch, yaw):
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def transform_point(parent_pose, local_pose):
    px, py, pz, roll, pitch, yaw = parent_pose
    lx, ly, lz, _, _, _ = local_pose
    rotation = rotation_matrix(roll, pitch, yaw)
    return (
        px + rotation[0][0] * lx + rotation[0][1] * ly + rotation[0][2] * lz,
        py + rotation[1][0] * lx + rotation[1][1] * ly + rotation[1][2] * lz,
        pz + rotation[2][0] * lx + rotation[2][1] * ly + rotation[2][2] * lz,
    )


def direct_pose(element):
    pose_element = element.find("pose")
    return parse_sdf_pose(pose_element.text if pose_element is not None else "")


def find_default_world_file():
    env_world = os.environ.get("DRONE_FIGURE8_WORLD_FILE")
    if env_world and os.path.exists(env_world):
        return env_world

    try:
        from ament_index_python.packages import get_package_share_directory

        share_dir = get_package_share_directory("drone_demo")
        world_file = os.path.join(share_dir, "worlds", "drone_world.sdf")
        if os.path.exists(world_file):
            return world_file
    except Exception:
        pass

    source_world = Path(__file__).resolve().parents[1] / "worlds" / "drone_world.sdf"
    if source_world.exists():
        return str(source_world)

    return None


def scene_defaults_from_sdf(world_file=None):
    world_file = world_file or find_default_world_file()
    if not world_file:
        return FALLBACK_SCENE_DEFAULTS.copy()

    try:
        root = ET.parse(world_file).getroot()
        camera_model = root.find(".//model[@name='ground_camera']")
        weapon_model = root.find(".//model[@name='weapon_platform']")
        if camera_model is None or weapon_model is None:
            return FALLBACK_SCENE_DEFAULTS.copy()

        camera_pose = direct_pose(camera_model)
        camera_sensor = camera_model.find(".//sensor[@name='camera_sensor']")
        if camera_sensor is not None:
            sensor_pose = direct_pose(camera_sensor)
            camera_pos = transform_point(camera_pose, sensor_pose)
            camera_rpy = (
                camera_pose[3] + sensor_pose[3],
                camera_pose[4] + sensor_pose[4],
                camera_pose[5] + sensor_pose[5],
            )
        else:
            camera_pos = camera_pose[:3]
            camera_rpy = camera_pose[3:]

        weapon_pose = direct_pose(weapon_model)
        muzzle = weapon_model.find(".//visual[@name='muzzle']")
        muzzle_pose = (
            direct_pose(muzzle)
            if muzzle is not None
            else (0.0, 0.0, 1.5, 0.0, 0.0, 0.0)
        )
        weapon_pos = transform_point(weapon_pose, muzzle_pose)
    except (OSError, ET.ParseError, ValueError):
        return FALLBACK_SCENE_DEFAULTS.copy()

    return {
        "camera_x": camera_pos[0],
        "camera_y": camera_pos[1],
        "camera_z": camera_pos[2],
        "camera_roll": camera_rpy[0],
        "camera_pitch": camera_rpy[1],
        "camera_yaw": camera_rpy[2],
        "weapon_x": weapon_pos[0],
        "weapon_y": weapon_pos[1],
        "weapon_z": weapon_pos[2],
    }
