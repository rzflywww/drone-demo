# Drone Demo

ROS 2 + Gazebo drone simulation demo. The project launches a simple quadcopter
model in Gazebo and drives it along a figure-8 trajectory. It also includes
helpers for camera snapshots/video recording and an optional YOLO detector.

## Tested Environment

This workspace was developed with:

- Ubuntu 24.04
- ROS 2 Jazzy
- Gazebo Harmonic / Gazebo Sim 8.x
- Python 3.12

Using the same versions is recommended. Other ROS/Gazebo versions may need code
or dependency changes.

## Install Dependencies

Install ROS 2 Jazzy first, then install the project dependencies:

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-desktop \
  ros-jazzy-ros-gz \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-simulation-interfaces \
  python3-colcon-common-extensions \
  python3-gz-msgs10 \
  python3-gz-transport13 \
  python3-opencv \
  python3-numpy \
  ffmpeg
```

Optional, only needed for YOLO detection:

```bash
pip install ultralytics
```

## Download And Build

```bash
git clone https://github.com/rzflywww/drone-demo.git drone_ws
cd drone_ws

source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

## Run The Simulation

```bash
ros2 launch drone_figure8 figure8.launch.py
```

Launch arguments:

```bash
ros2 launch drone_figure8 figure8.launch.py amplitude:=3.0 height:=2.0 period:=12.0 rate:=50.0
```

- `amplitude`: figure-8 path amplitude in meters
- `height`: flight altitude in meters
- `period`: seconds for one full figure-8 cycle
- `rate`: controller update rate in Hz

## Record Camera Output

Start the simulation first, then use the recording helper in another terminal:

```bash
cd drone_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
```

Take a snapshot:

```bash
ros2 run drone_figure8 drone_record snap -o snapshot.png
```

Record a video:

```bash
ros2 run drone_figure8 drone_record video -o drone_flight.mp4 -d 5
```

## YOLO Detection

Start the simulation first, then run:

```bash
ros2 run drone_figure8 yolo_detector --model /path/to/yolo11n.pt
```

The default model path in the code points to the original development machine,
so passing `--model` is recommended on a new computer.

To publish detected target centers for the laser controller:

```bash
ros2 run drone_figure8 yolo_detector --model /path/to/yolo11n.pt --ros-args
```

## Package Layout

```text
src/drone_figure8/
  drone_figure8/
    figure8_controller.py   # figure-8 trajectory controller
    laser_controller.py     # laser target controller
    record.py               # snapshot/video helper
    camera_recorder.py      # Gazebo camera frame converter
    yolo_detector.py        # optional YOLO detector
  launch/
    figure8.launch.py
  worlds/
    drone_world.sdf
    gui.config
```

## Notes

- `build/`, `install/`, `log/`, virtual environments, and Python cache files are
  intentionally ignored by Git.
- If Gazebo opens but no camera output is recorded, check the available topics:

```bash
gz topic --list
gz service --list
```
