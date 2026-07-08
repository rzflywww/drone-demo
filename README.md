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
## Run The Simulation

Start the Gazebo simulation. The figure-8 controller is **not** started
automatically by default, so the drone will hover at its initial pose until
you start the controller manually.

```bash
ros2 launch drone_figure8 figure8.launch.py
```

To keep the old auto-start behavior, pass `start_controller:=true`:

```bash
ros2 launch drone_figure8 figure8.launch.py start_controller:=true
```

Launch arguments:

```bash
ros2 launch drone_figure8 figure8.launch.py \
  amplitude:=3.0 height:=2.0 period:=12.0 rate:=50.0
```

- `start_controller`: if `true`, auto-start the figure-8 controller after 6s
- `amplitude`: figure-8 path amplitude in meters
- `height`: flight altitude in meters
- `period`: seconds for one full figure-8 cycle
- `rate`: controller update rate in Hz

### Start the figure-8 controller manually

After launching the simulation, open another terminal and run:

```bash
cd /home/rzfly/drone_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run drone_figure8 figure8_controller --ros-args \
  -p amplitude:=3.0 \
  -p height:=2.0 \
  -p period:=12.0 \
  -p rate:=50.0
```

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

To aim the laser at detected target centers, start the laser controller in a
separate terminal:

```bash
cd /home/rzfly/drone_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run drone_figure8 laser_controller
```

By default, the laser is emitted from the weapon platform next to the ground
camera. Override `weapon_x`, `weapon_y`, `weapon_z`, or `laser_aim_distance`
with ROS parameters if you move the platform.

Then start YOLO detection and publish target centers:

```bash
/home/rzfly/drone_ws/yolo_venv/bin/python3 \
  /home/rzfly/drone_ws/src/drone_figure8/drone_figure8/yolo_detector.py \
  --model /path/to/yolo11n.pt --prediction-time 0.15
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
