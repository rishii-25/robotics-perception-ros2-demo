"""
Minimal ROS2 subscriber node.

Concept: this node subscribes to the '/robot_status' topic and runs a
callback every time a new message arrives. It has no idea who published
the message or how many other subscribers exist -- that decoupling is
the whole point of the pub/sub model.

Run (in a SEPARATE terminal from talker.py, after sourcing the workspace):
    ros2 run minimal_demo listener
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Listener(Node):
    def __init__(self):
        super().__init__('listener')
        self.subscription = self.create_subscription(
            String,
            'robot_status',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'Received: "{msg.data}"')


def main(args=None):
    rclpy.init(args=args)
    node = Listener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
