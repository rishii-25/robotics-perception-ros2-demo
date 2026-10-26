"""
Reactor ROS2 node -- the "decision" side of the perception-to-action loop.

This node has NO IDEA there's a camera or a neural network involved.
It just subscribes to 'detected_objects' and reacts to whatever JSON
text shows up. In a real robot, this is where you'd publish a motor
command (e.g. a Twist on /cmd_vel, or a serial command to an STM32)
instead of just printing -- the print statements here are stand-ins
for "this is where actuation would happen."

Run (in a separate terminal from detector.py):
    ros2 run yolo_detector reactor
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Reactor(Node):
    def __init__(self):
        super().__init__('object_reactor')
        self.subscription = self.create_subscription(
            String,
            'detected_objects',
            self.callback,
            10)
        self.subscription  # prevent unused variable warning

    def callback(self, msg):
        try:
            detections = json.loads(msg.data)
        except json.JSONDecodeError:
            return

        if not detections:
            return

        labels = [d['label'] for d in detections]
        self.get_logger().info(f'Saw: {labels}')

        # This if/elif chain stands in for real robot decision logic --
        # swap these prints for actual commands (motor, arm, turtle, etc.)
        if 'person' in labels:
            self.get_logger().info('  -> Person detected: would trigger "greet" behavior')
        if 'cell phone' in labels or 'laptop' in labels:
            self.get_logger().info('  -> Device detected: would trigger "pick up" behavior')


def main(args=None):
    rclpy.init(args=args)
    node = Reactor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
