from setuptools import setup

package_name = "drone_demo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/sim.launch.py"]),
        ("share/" + package_name + "/worlds", [
            "worlds/drone_world.sdf",
            "worlds/gui.config",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="user",
    maintainer_email="user@example.com",
    description="Drone trajectory and tracking demo in Gazebo",
    license="MIT",
    test_suite="tests",
    entry_points={
        "console_scripts": [
            "figure8_controller = drone_demo.figure8_controller:main",
            "circle_controller = drone_demo.circle_controller:main",
            "laser_controller = drone_demo.laser_controller:main",
            "drone_pose_monitor = drone_demo.pose_monitor:main",
            "drone_record = drone_demo.record:main",
            "drone_camera_recorder = drone_demo.camera_recorder:main",
            "yolo_detector = drone_demo.yolo_detector:main",
        ],
    },
)
