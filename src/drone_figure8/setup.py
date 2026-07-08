from setuptools import setup

package_name = "drone_figure8"

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
    description="Drone trajectory simulation in Gazebo",
    license="MIT",
    entry_points={
        "console_scripts": [
            "figure8_controller = drone_figure8.figure8_controller:main",
            "circle_controller = drone_figure8.circle_controller:main",
            "laser_controller = drone_figure8.laser_controller:main",
            "drone_record = drone_figure8.record:main",
            "drone_camera_recorder = drone_figure8.camera_recorder:main",
            "yolo_detector = drone_figure8.yolo_detector:main",
        ],
    },
)
