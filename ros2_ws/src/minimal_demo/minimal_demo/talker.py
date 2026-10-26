"""
Minimal ROS2 publisher node.

Concept: this node publishes a String message on the topic '/robot_status'
every 0.5 seconds. Any number of other nodes (like listener.py) can
subscribe to this topic and receive the message independently.

Run (after building the workspace, see README.md):
    ros2 run minimal_demo talker
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class Talker(Node):
    def __init__(self):
        # Node name as it will appear in `ros2 node list`
        super().__init__('talker')

        # Create a publisher: message type String, topic name, queue size
        self.publisher_ = self.create_publisher(String, 'robot_status', 10)

        # Timer calls self.publish_message every 0.5 seconds
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.publish_message)
        self.count = 0

    def publish_message(self):
        msg = String()
        msg.data = f'Robot status update #{self.count}: all systems nominal'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.count += 1


def main(args=None):
    rclpy.init(args=args)
    node = Talker()
    rclpy.spin(node)  # keeps the node alive, processing callbacks
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
