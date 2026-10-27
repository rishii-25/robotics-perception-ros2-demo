"""
YOLO detector ROS2 node -- the "perception" side of the loop.

It owns the camera and the YOLO model, runs inference on a timer, and
PUBLISHES what it sees on the 'detected_objects' topic. Other nodes
(reactor.py, follower.py) subscribe without knowing anything about
cameras or neural networks.

Message on 'detected_objects' (std_msgs/String holding JSON, to keep the
demo dependency-free -- a production system would use a custom .msg):

    {
      "stamp": 1730012345.12,          # wall-clock time the frame was captured
      "frame_w": 640, "frame_h": 1080,
      "detections": [
        {"label": "person", "confidence": 0.91,
         "cx": 312.5, "cy": 540.0, "w": 140.0, "h": 420.0}   # box centre + size, px
      ]
    }

Sources (parameter 'source'):
    sample        -> the bus.jpg photo that ships with ultralytics
    0, 1, ...     -> webcam index
    path/to.jpg   -> a still image (read once and reused every tick)
    path/to.mp4   -> a video file (loops when it reaches the end)

Virtual camera (parameter 'virtual_pan:=true', image sources only):
    The photo is placed in a 360-degree world and the node subscribes to
    /turtle1/pose, so the frame it sees depends on which way the turtle
    faces. See camera.py.

Run:
    ros2 run yolo_detector detector --ros-args -p source:=sample
    ros2 run yolo_detector detector --ros-args -p source:=sample -p virtual_pan:=true
"""

import json
import os
import time

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO

from yolo_detector.camera import VirtualPanCamera, image_msg_to_bgr

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def resolve_source(source_param):
    if source_param == 'sample':
        from ultralytics.utils import ASSETS
        return str(ASSETS / 'bus.jpg')
    if source_param.isdigit():
        return int(source_param)
    return os.path.expanduser(source_param)


class Detector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        self.declare_parameter('source', 'sample')
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('model', 'yolov8n.pt')
        self.declare_parameter('rate_hz', 5.0)
        self.declare_parameter('virtual_pan', False)
        self.declare_parameter('fov_deg', 60.0)
        self.declare_parameter('scene_bearing_deg', -70.0)
        self.declare_parameter('view_width', 640)
        self.declare_parameter('pose_topic', '/turtle1/pose')
        self.declare_parameter('debug_image', '')
        self.declare_parameter('image_topic', '')

        gp = self.get_parameter
        self.source = resolve_source(gp('source').value)
        self.conf = float(gp('conf').value)
        model_name = gp('model').value
        rate_hz = float(gp('rate_hz').value)
        virtual_pan = bool(gp('virtual_pan').value)
        self.debug_image = gp('debug_image').value

        self.publisher_ = self.create_publisher(String, 'detected_objects', 10)

        self.get_logger().info(f'Loading model: {model_name}')
        self.model = YOLO(model_name)

        self.cap = None
        self.still = None
        self.vcam = None
        self.theta = 0.0
        self.last_debug_write = 0.0
        self.latest = None          # (Image msg, receive_time) from the ROS camera topic
        self.use_topic = False

        image_topic = gp('image_topic').value
        if image_topic:
            # Robot camera (e.g. Gazebo TurtleBot3 /camera/image_raw). Sensor-data
            # QoS (best effort) matches both best-effort and reliable publishers.
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
            self.create_subscription(Image, image_topic, self.on_image, qos_profile_sensor_data)
            self.use_topic = True
            self.get_logger().info(f'Using ROS camera topic: {image_topic}')
            self.timer = self.create_timer(1.0 / rate_hz, self.process_frame)
            return

        is_image = isinstance(self.source, str) and self.source.lower().endswith(IMAGE_EXTS)
        if is_image:
            self.still = cv2.imread(self.source)
            if self.still is None:
                raise RuntimeError(f'Could not read image: {self.source}')
            self.get_logger().info(f'Using still image: {self.source}')
        else:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise RuntimeError(f'Could not open source: {self.source}')
            self.get_logger().info(f'Opened video source: {self.source}')

        if virtual_pan:
            if self.still is None:
                raise RuntimeError('virtual_pan needs an image source (e.g. source:=sample)')
            from turtlesim.msg import Pose
            self.vcam = VirtualPanCamera(
                self.still,
                view_width=int(gp('view_width').value),
                fov_deg=float(gp('fov_deg').value),
                scene_bearing_deg=float(gp('scene_bearing_deg').value))
            self.create_subscription(Pose, gp('pose_topic').value, self.on_pose, 10)
            self.get_logger().info('Virtual pan camera ON: view follows turtle heading')

        # Timer-driven loop instead of a while-loop, so the executor can keep
        # servicing the pose subscription, parameters and Ctrl+C in between.
        self.timer = self.create_timer(1.0 / rate_hz, self.process_frame)

    def on_pose(self, msg):
        self.theta = msg.theta

    def on_image(self, msg):
        # Just keep the newest message. YOLO runs on the timer, so a 30 Hz
        # camera never queues up frames faster than we can process them, and
        # we only pay for converting the frames we actually use.
        self.latest = (msg, time.time())

    def grab_frame(self):
        if self.use_topic:
            if self.latest is None:
                return None, None
            msg, stamp = self.latest
            self.latest = None       # don't run YOLO twice on the same frame
            try:
                frame = image_msg_to_bgr(msg.height, msg.width, msg.step,
                                         msg.encoding, msg.data)
            except ValueError as e:
                self.get_logger().error(str(e), throttle_duration_sec=5.0)
                return None, None
            return frame, stamp
        now = time.time()
        if self.vcam is not None:
            return self.vcam.view(self.theta), now
        if self.still is not None:
            return self.still, now
        ok, frame = self.cap.read()
        if not ok and isinstance(self.source, str):
            # End of a video file: rewind and keep going.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return (frame, now) if ok else (None, None)

    def process_frame(self):
        # stamp = when the frame was captured/received; used downstream to
        # measure latency.
        frame, stamp = self.grab_frame()
        if frame is None:
            if self.cap is not None:
                self.get_logger().warn('No frame received, skipping this tick',
                                       throttle_duration_sec=5.0)
            return

        results = self.model(frame, conf=self.conf, verbose=False)
        names = results[0].names
        boxes = results[0].boxes

        detections = []
        if boxes is not None and len(boxes) > 0:
            for cls_id, confidence, xywh in zip(boxes.cls, boxes.conf, boxes.xywh):
                cx, cy, w, h = (round(float(v), 1) for v in xywh)
                detections.append({
                    'label': names[int(cls_id)],
                    'confidence': round(float(confidence), 2),
                    'cx': cx, 'cy': cy, 'w': w, 'h': h,
                })

        payload = {
            'stamp': stamp,
            'frame_w': int(frame.shape[1]),
            'frame_h': int(frame.shape[0]),
            'detections': detections,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.publisher_.publish(msg)

        labels = [d['label'] for d in detections]
        self.get_logger().info(f'{len(detections)} objects: {labels}',
                               throttle_duration_sec=2.0)

        # Debug image at most once per second: writing to /mnt/c from WSL is
        # slow and would otherwise drag the detector's frame rate down.
        if self.debug_image and time.time() - self.last_debug_write >= 1.0:
            self.last_debug_write = time.time()
            self.write_debug_image(results[0].plot())

    def write_debug_image(self, annotated):
        # Overwrites one file each frame so you can open it in Windows and
        # watch what the robot sees (cv2.imshow is unreliable inside WSL).
        # A debug image must never take the detector down, so any failure is
        # just a warning. The temp name includes our PID so two detectors
        # writing the same path can't rename each other's files.
        h, w = annotated.shape[:2]
        cv2.line(annotated, (w // 2, 0), (w // 2, h), (0, 255, 255), 2)
        tmp = f'{self.debug_image}.{os.getpid()}.tmp.jpg'
        try:
            if not cv2.imwrite(tmp, annotated):
                raise OSError(f'cv2.imwrite failed for {tmp}')
            os.replace(tmp, self.debug_image)
        except OSError as e:
            self.get_logger().warn(f'debug image not saved: {e}', throttle_duration_sec=10.0)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = Detector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
