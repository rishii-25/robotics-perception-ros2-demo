"""
YOLO detector ROS2 node.

This is the "perception" node: it owns the camera and the YOLO model,
runs inference on a timer, and PUBLISHES what it sees as a ROS2 topic
instead of just printing to a terminal. Any number of other nodes
(like reactor.py) can subscribe to this without knowing anything
about cameras or neural networks -- that decoupling is the whole
point of wrapping this in ROS2.

Message format published on 'detected_objects' (std_msgs/String,
containing a JSON-encoded list, to keep this demo dependency-free --
a production system would define a proper custom message type, e.g.
in a separate _msgs package):

    '[{"label": "person", "confidence": 0.92}, {"label": "tie", "confidence": 0.43}]'

Run:
    ros2 run yolo_detector detector --ros-args -p source:=0
    ros2 run yolo_detector detector --ros-args -p source:=test.jpg
"""

import json

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO


class Detector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # ROS2 parameters -- settable from the command line with --ros-args -p
        self.declare_parameter('source', '0')
        self.declare_parameter('conf', 0.4)
        self.declare_parameter('model', 'yolov8n.pt')
        self.declare_parameter('rate_hz', 5.0)

        source_param = self.get_parameter('source').get_parameter_value().string_value
        self.conf = self.get_parameter('conf').get_parameter_value().double_value
        model_name = self.get_parameter('model').get_parameter_value().string_value
        rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value

        # '0' (webcam index) needs to become an int for OpenCV; a file path stays a string
        self.source = int(source_param) if source_param.isdigit() else source_param

        self.publisher_ = self.create_publisher(String, 'detected_objects', 10)

        self.get_logger().info(f'Loading model: {model_name}')
        self.model = YOLO(model_name)

        self.get_logger().info(f'Opening source: {self.source}')
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            self.get_logger().error(f'Could not open source {self.source}')

        # Timer-driven loop: every (1/rate_hz) seconds, grab a frame and run inference.
        # A timer (not a tight while-loop) lets ROS2's executor keep servicing
        # callbacks, parameter updates, Ctrl+C shutdown, etc. in between frames.
        self.timer = self.create_timer(1.0 / rate_hz, self.process_frame)

    def process_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.get_logger().warn('No frame received, skipping this tick')
            return

        results = self.model(frame, conf=self.conf, verbose=False)
        names = results[0].names
        boxes = results[0].boxes

        detections = []
        if boxes is not None and len(boxes) > 0:
            for cls_id, confidence in zip(boxes.cls, boxes.conf):
                detections.append({
                    'label': names[int(cls_id)],
                    'confidence': round(float(confidence), 2),
                })

        msg = String()
        msg.data = json.dumps(detections)
        self.publisher_.publish(msg)

        if detections:
            self.get_logger().info(f'Published: {msg.data}')

    def destroy_node(self):
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
