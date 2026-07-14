# Drone Trajectory And Tracking Demo

ROS 2 + Gazebo drone simulation demo. The project launches a simple quadcopter
model in Gazebo. The drone stays still after simulation startup, and trajectory
controllers can then be started manually to fly figure-8 or circular paths. It
also includes helpers for pose monitoring, camera snapshots/video recording,
YOLO detection, and laser target visualization.

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

Start Gazebo first. The launch file only starts the simulation, so the drone
stays at its initial pose until you manually start a trajectory controller from
another terminal.

```bash
ros2 launch drone_demo sim.launch.py
```

Launch argument:

```bash
ros2 launch drone_demo sim.launch.py world_file:=/path/to/drone_world.sdf
```

- `world_file`: path to the SDF world file

### Fly a figure-8

After launching the simulation, open another terminal and run:

```bash
cd /home/rzfly/drone_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run drone_demo figure8_controller --ros-args \
  -p amplitude:=3.0 \
  -p height:=2.0 \
  -p period:=12.0 \
  -p rate:=50.0
```

Figure-8 parameters:

- `amplitude`: figure-8 path amplitude in meters
- `height`: flight altitude in meters
- `period`: seconds for one full figure-8 cycle
- `rate`: controller update rate in Hz

### Fly a circle

After launching the simulation, open another terminal and run:

```bash
cd /home/rzfly/drone_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run drone_demo circle_controller --ros-args \
  -p radius:=3.0 \
  -p height:=2.0 \
  -p period:=12.0 \
  -p center_x:=0.0 \
  -p center_y:=0.0 \
  -p clockwise:=false \
  -p rate:=50.0
```

Circle parameters:

- `radius`: circle radius in meters
- `height`: flight altitude in meters
- `period`: seconds for one full circle
- `center_x`, `center_y`: circle center in world coordinates
- `clockwise`: if `true`, fly clockwise; otherwise counter-clockwise
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
ros2 run drone_demo drone_record snap -o snapshot.png
```

Record a video:

```bash
ros2 run drone_demo drone_record video -o drone_flight.mp4 -d 5
```

## YOLO Detection

Start the simulation first, then run:

```bash
ros2 run drone_demo yolo_detector
```

This workspace uses
`/home/rzfly/ultralytics-8.3.39/ultralytics/yolo11n.pt` by default. Override it
with `--model /path/to/yolo11n.pt` after moving the project to another machine.

To aim the laser at detected target centers, start the laser controller in a
separate terminal:

```bash
cd /home/rzfly/drone_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run drone_demo laser_controller
```

By default, the laser is emitted from the weapon platform next to the ground
camera. Override `weapon_x`, `weapon_y`, `weapon_z`, or `laser_aim_distance`
with ROS parameters if you move the platform.

The laser controller also subscribes to the Gazebo depth image topic
`/ground_camera/depth` by default. When a valid depth value is available at the
target pixel, the controller reconstructs the target's 3D world position before
aiming the weapon platform. If depth data is unavailable, it falls back to the
fixed `laser_aim_distance` approximation.

Then start YOLO detection and publish target centers:

```bash
/home/rzfly/drone_ws/yolo_venv/bin/python3 \
  /home/rzfly/drone_ws/src/drone_demo/drone_demo/yolo_detector.py \
  --model /path/to/yolo11n.pt
```

### Recommended World-Space Prediction

For laser aiming with depth, keep the YOLO output unfiltered and enable the
Kalman filter in the laser controller. The controller reconstructs the current
3D target position from the detection pixel and depth first, then filters and
predicts that position in world coordinates:

```text
current YOLO pixel + current depth -> 3D world measurement -> Kalman prediction -> laser aim
```

Start the laser controller with world-space filtering enabled:

```bash
ros2 run drone_demo laser_controller --ros-args \
  -p world_target_filter:=kalman \
  -p world_prediction_time:=0.15
```

YOLO always publishes the current raw detection center, so no filtering or
prediction arguments are needed:

```bash
ros2 run drone_demo yolo_detector
```

The world-space filter options are ROS parameters on `laser_controller`:

| Parameter | Default | Description |
| --- | --- | --- |
| `world_target_filter` | `none` | Select `none` for direct aiming or `kalman` for 3D filtering. |
| `world_prediction_time` | `0.15` | Seconds to predict the world position ahead; `0` keeps smoothing without extrapolation. |
| `world_kalman_process_noise` | `10.0` | Process noise for the meter-based position, velocity, and acceleration model. |
| `world_kalman_measurement_noise` | `0.04` | World-position measurement variance in square meters. |
| `world_filter_max_measurement_age` | `0.5` | Maximum measurement age included in the prediction horizon. |

Each new YOLO pixel updates the world filter at most once. Invalid depth and
the fixed-distance fallback are not inserted as Kalman measurements; the
controller temporarily holds the existing world prediction instead.

The former image-space Kalman prediction has been removed to prevent future
pixels from being combined with the current depth frame. Filtering and
prediction now happen only after the target has been reconstructed in 3D.

World-space filter implementations and their factory are kept in
`target_filters.py`. Additional filters should provide `update(...)` and
`predict(lead_time)`, then be registered in `TARGET_FILTER_NAMES` and
`create_world_target_filter()`.

## Remote LLaVA Detection

The first-stage LLaVA bridge sends Gazebo RGB frames to an HTTP service and
publishes the returned center on the same `/laser_target_pixel` topic used by
the laser controller. The AutoDL service supports a mock mode so networking can
be verified before loading a model. See
[`autodl_server/README.md`](autodl_server/README.md) for server setup,
SSH tunneling, and end-to-end checks.

After the AutoDL service and SSH tunnel are running, start the bridge with:

```bash
cd /home/rzfly/drone_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run drone_demo llava_detector --server-url http://127.0.0.1:8000
```

The bridge intentionally sends one request at a time. Use `--interval 2.0` to
limit requests while testing a slow model.

## Package Layout

```text
src/drone_demo/
  drone_demo/
    figure8_controller.py   # figure-8 trajectory controller
    circle_controller.py    # circular trajectory controller
    laser_controller.py     # laser target controller
    record.py               # snapshot/video helper
    camera_recorder.py      # Gazebo camera frame converter
    target_filters.py       # optional 3D world-target filters
    yolo_detector.py        # YOLO detector publishing raw target pixels
  launch/
    sim.launch.py
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
