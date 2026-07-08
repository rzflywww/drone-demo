"""Launch drone simulation with ground camera view."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
)
from launch.substitutions import LaunchConfiguration
from ros_gz_sim.actions import GzServer


def generate_launch_description():
    pkg_dir = get_package_share_directory("drone_figure8")
    world_file = os.path.join(pkg_dir, "worlds", "drone_world.sdf")
    gui_config = os.path.join(pkg_dir, "worlds", "gui.config")

    world_file_arg = DeclareLaunchArgument(
        "world_file", default_value=world_file,
        description="Path to the SDF world file"
    )

    gz_server = GzServer(
        world_sdf_file=LaunchConfiguration("world_file"),
    )

    gz_gui = ExecuteProcess(
        cmd=["gz", "sim", "-g", "--gui-config", gui_config],
        output="screen",
    )

    return LaunchDescription([
        world_file_arg,
        gz_server,
        gz_gui,
    ])
