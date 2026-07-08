"""Launch figure-8 drone simulation with ground camera view."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from ros_gz_sim.actions import GzServer


def generate_launch_description():
    pkg_dir = get_package_share_directory("drone_figure8")
    world_file = os.path.join(pkg_dir, "worlds", "drone_world.sdf")
    gui_config = os.path.join(pkg_dir, "worlds", "gui.config")

    # 可调参数
    amplitude_arg = DeclareLaunchArgument(
        "amplitude", default_value="3.0",
        description="Figure-8 amplitude (meters)"
    )
    height_arg = DeclareLaunchArgument(
        "height", default_value="2.0",
        description="Flight altitude (meters)"
    )
    period_arg = DeclareLaunchArgument(
        "period", default_value="12.0",
        description="Time for one full figure-8 cycle (seconds)"
    )
    rate_arg = DeclareLaunchArgument(
        "rate", default_value="50.0",
        description="Controller update rate (Hz)"
    )
    start_controller_arg = DeclareLaunchArgument(
        "start_controller", default_value="false",
        description="If true, auto-start figure-8 controller after 6s delay"
    )
    world_file_arg = DeclareLaunchArgument(
        "world_file", default_value=world_file,
        description="Path to the SDF world file"
    )

    # Stage 1: GzServer（ROS2 服务）
    gz_server = GzServer(
        world_sdf_file=LaunchConfiguration("world_file"),
    )

    # Stage 2: Gazebo GUI (with custom config for camera ImageDisplay)
    gz_gui = ExecuteProcess(
        cmd=["gz", "sim", "-g", "--gui-config", gui_config],
        output="screen",
    )

    # Stage 3: 控制器（延迟 6 秒，等待 Gazebo 就绪；默认不启动，可手动启动）
    controller = TimerAction(
        period=6.0,
        actions=[
            Node(
                package="drone_figure8",
                executable="figure8_controller",
                name="figure8_controller",
                output="screen",
                parameters=[{
                    "amplitude": LaunchConfiguration("amplitude"),
                    "height": LaunchConfiguration("height"),
                    "period": LaunchConfiguration("period"),
                    "rate": LaunchConfiguration("rate"),
                }],
            )
        ],
        condition=IfCondition(LaunchConfiguration("start_controller")),
    )

    return LaunchDescription([
        amplitude_arg,
        height_arg,
        period_arg,
        rate_arg,
        world_file_arg,
        start_controller_arg,
        gz_server,
        gz_gui,
        controller,
    ])
