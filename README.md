# Robotics Track Prep — Tonight's Build

Two independent pieces, both runnable tonight on your existing WSL2 Ubuntu
(same one you use for VLSI-ML work):

1. `cv_detection/` — YOLOv8 object detection with OpenCV (no ROS2 needed)
2. `ros2_ws/` — minimal ROS2 publisher/subscriber (nodes + topics)

Do the CV part first — it's faster to get a visible result and doesn't
depend on ROS2 being installed correctly.

---

## Part 1: OpenCV + YOLOv8 object detection

```bash
cd cv_detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Easiest first run: on a photo (avoids WSL2 webcam headaches)
python detect.py --source path/to/any_photo.jpg

# Or try live webcam (works if you're on WSL2 with usbipd-win camera
# passthrough set up, or just run this part on native Windows Python instead)
python detect.py --source 0
```

First run auto-downloads `yolov8n.pt` (~6MB) from Ultralytics. You'll see
console output listing what it detected per frame, and an
`annotated_output.jpg` (for image input) or live window (for webcam) with
bounding boxes.

**What to be able to explain from this:** the detect → NMS → draw pipeline
in the docstring at the top of `detect.py`, why YOLO is a single-stage
detector (as opposed to two-stage like Faster R-CNN — faster, slightly less
accurate), and what confidence threshold / NMS actually do.

---

## Part 2: ROS2 minimal publisher/subscriber

### Install ROS2 (Humble, on Ubuntu 22.04 in WSL2)

```bash
# Check your Ubuntu version first
lsb_release -a

locale  # verify UTF-8
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

sudo apt install software-properties-common
sudo add-apt-repository universe

sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install ros-humble-desktop -y   # includes Gazebo-related tools too

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

(If you're on Ubuntu 24.04 instead of 22.04, use `ros-jazzy-desktop` and the
`jazzy` codename throughout — same steps otherwise.)

Verify install:
```bash
ros2 doctor --report
```

### Build and run this package

```bash
cd ros2_ws
colcon build --packages-select minimal_demo
source install/setup.bash

# Terminal 1
ros2 run minimal_demo talker

# Terminal 2 (new terminal, also run: source install/setup.bash)
ros2 run minimal_demo listener
```

You should see the talker printing "Publishing: ..." and the listener
printing "Received: ..." for the same messages — that's the pub/sub model
working end to end.

**Useful commands to poke around with** (run while talker is running):
```bash
ros2 node list              # see both nodes once both are running
ros2 topic list             # see /robot_status
ros2 topic echo /robot_status   # watch messages without writing a subscriber
ros2 topic hz /robot_status     # confirm publish rate
```

### Gazebo (if time permits)

`ros-humble-desktop` already includes Gazebo integration packages. Just
confirm it launches:
```bash
gazebo
```
A full simulated robot in Gazebo is a multi-day task (URDF modeling, sensor
plugins, spawn scripts) — for tonight, being able to launch it and explain
what it's for (physics simulation before touching real hardware) is enough.

---

## Next step after tonight

Once both of these run cleanly, the natural "tie it together" project is a
ROS2 node that wraps `detect.py`'s YOLO inference and publishes detected
object labels on a topic (e.g. `/detected_objects`) instead of just
printing them — that's a legitimate small portfolio project combining
perception + ROS2, and reuses everything built here.
