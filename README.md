# Autonomous Perception-to-Navigation Pipeline (ROS2)

An end-to-end robotics perception-to-action pipeline, built incrementally
toward a target-following, obstacle-avoiding mobile robot: a YOLOv8 object
detector publishes what it sees as a ROS2 topic, and a decision node
subscribes to it and reacts — the perception → decision → action pattern
used in real robotic systems.

Built as a hands-on exploration of the Robotics & AI stack: computer
vision, ROS2 middleware, closed-loop control, and simulation.

## Roadmap

| Stage | Status | Notes |
|---|---|---|
| Object detection (YOLOv8 + OpenCV) | ✅ Done | `cv_detection/` — image/video/webcam |
| ROS2 node/topic pub-sub pattern | ✅ Done | `minimal_demo/` |
| Perception node → decision node (reactor) | ✅ Done | `yolo_detector/` |
| Closed-loop target following | 🔄 In progress | Proportional control on turtlesim driven by detection position |
| Obstacle avoidance (simulated LiDAR/depth) | 📋 Planned | Needs a 3D sim (Gazebo); blocked locally by WSL2 GPU/rendering limits |
| Performance evaluation (detection rate, latency, following error) | 📋 Planned | Logging harness around the existing nodes |

## What's in here

| Folder | What it does |
|---|---|
| `cv_detection/` | Standalone YOLOv8 + OpenCV object detection (image/video/webcam) |
| `ros2_ws/src/minimal_demo/` | A minimal ROS2 publisher/subscriber pair — the basic node/topic communication pattern |
| `ros2_ws/src/yolo_detector/` | The combined pipeline: a `detector` node runs YOLO and publishes detections to a topic; a `reactor` node subscribes and demonstrates decision logic based on what's detected |

## Why this architecture

Real robots split perception and decision-making into independent
processes that communicate over a message bus (ROS2's pub/sub model),
rather than one monolithic script. This repo is a small, working example
of that pattern:

```
 Camera / image  →  OpenCV (frame capture)  →  YOLOv8 (inference)
                                                     │
                                            ROS2 topic: /detected_objects
                                                     │
                                       reactor node (decision logic)
                                                     │
                                   → would drive a motor / actuator command
```

The `detector` node has no idea who (if anyone) is listening. The
`reactor` node has no idea a camera or neural network is involved. That
decoupling is what makes ROS2 useful for real hardware — swap the camera
for a different sensor, or the reactor for actual motor control, without
touching the other side.

## Running it

### 1. Object detection (no ROS2 required)

```bash
cd cv_detection
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python detect.py --source test.jpg --save
```
Outputs `annotated_output.jpg` with bounding boxes, labels, and confidence
scores. Also supports `--source 0` for a live webcam feed.

### 2. ROS2 minimal pub/sub

Requires ROS2 Humble installed ([instructions](https://docs.ros.org/en/humble/Installation.html)).

```bash
cd ros2_ws
colcon build --packages-select minimal_demo
source install/setup.bash

# Terminal 1
ros2 run minimal_demo talker

# Terminal 2
ros2 run minimal_demo listener
```

### 3. Full perception → decision pipeline

```bash
cd ros2_ws
colcon build --packages-select yolo_detector
source install/setup.bash

# Terminal 1 — perception
ros2 run yolo_detector detector --ros-args -p source:=/path/to/image.jpg

# Terminal 2 — decision logic
ros2 run yolo_detector reactor
```
The reactor prints what it saw and what action it would trigger (e.g.
`Person detected: would trigger "greet" behavior`) — a stand-in for real
actuator commands.

## Stack

- **YOLOv8** (Ultralytics) — pretrained on COCO, single-stage anchor-free
  object detector
- **OpenCV** — frame capture and visualization
- **ROS2 Humble** — node/topic pub-sub middleware, built with `colcon`
- **TurtleSim** — used to verify ROS2 topic → motion control end to end

## Notes

- Detection messages are passed as JSON-encoded `std_msgs/String` for
  simplicity; a production version would define a proper custom message
  type.
- Built and tested on Ubuntu 22.04 (WSL2) and Windows 11 (native, for
  live webcam access).
