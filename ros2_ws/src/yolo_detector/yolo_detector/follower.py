"""
Follower ROS2 node -- closes the perception -> decision -> actuation loop.

Two modes:

  turtlesim (default)  Turn-only. Runs the heading controller from control.py
                       and publishes the angular velocity, so the turtle turns
                       to face the target, like a humanoid turning its head.

  drive:=true          For a real or Gazebo mobile robot (TurtleBot3). Adds
                       the LiDAR (/scan) and odometry (/odom) and runs the
                       motion planner from navigation.py: fuse camera bearing
                       with LiDAR range, drive toward the person, stop at
                       follow_distance, steer around obstacles, remember where
                       the person was when they leave the camera view.

Every control tick is logged to a CSV. On shutdown (Ctrl+C) a summary of the
run is printed and saved as JSON: detection rate, perception latency, time to
acquire/center the target, following error, and in drive mode also collisions,
minimum clearance, task completion time and distance travelled.

Run:
    ros2 run yolo_detector follower                       # turtlesim
    ros2 run yolo_detector follower --ros-args -p drive:=true -p cmd_topic:=/cmd_vel \
        -p fov_deg:=62.2                                  # TurtleBot3
"""

import csv
import json
import math
import os
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import String

from yolo_detector.control import ControllerConfig, HeadingController, Metrics
from yolo_detector.navigation import DriveMetrics, MotionConfig, MotionPlanner, Scan

try:
    from rclpy.executors import ExternalShutdownException
except ImportError:  # older rclpy
    class ExternalShutdownException(Exception):
        pass

MAX_LINEAR = 0.26    # TurtleBot3 Waffle limits
MAX_ANGULAR = 1.82


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class Follower(Node):
    def __init__(self):
        super().__init__('target_follower')

        ctrl_defaults = ControllerConfig()
        nav_defaults = MotionConfig()
        for name, value in vars(ctrl_defaults).items():
            self.declare_parameter(name, value)
        for name, value in vars(nav_defaults).items():
            if name != 'camera_fov_deg':            # single source of truth: fov_deg
                self.declare_parameter(name, value)
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('cmd_topic', '/turtle1/cmd_vel')
        self.declare_parameter('log_csv', 'follower_log.csv')
        self.declare_parameter('drive', False)
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('odom_topic', '/odom')

        gp = self.get_parameter
        cfg = ControllerConfig(**{n: gp(n).value for n in vars(ctrl_defaults)})
        self.ctrl = HeadingController(cfg)
        self.start = time.time()
        self.metrics = Metrics(fov_deg=cfg.fov_deg)

        self.drive = bool(gp('drive').value)
        self.scan = None
        self.pose = None
        self.planner = None
        self.drive_metrics = None
        self.last_mode = None
        if self.drive:
            from nav_msgs.msg import Odometry
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import LaserScan
            nav_kwargs = {n: gp(n).value for n in vars(nav_defaults) if n != 'camera_fov_deg'}
            ncfg = MotionConfig(camera_fov_deg=cfg.fov_deg, **nav_kwargs)
            self.planner = MotionPlanner(ncfg)
            self.drive_metrics = DriveMetrics(ncfg.follow_distance)
            self.create_subscription(LaserScan, gp('scan_topic').value, self.on_scan,
                                     qos_profile_sensor_data)
            self.create_subscription(Odometry, gp('odom_topic').value, self.on_odom, 10)

        self.cmd_pub = self.create_publisher(Twist, gp('cmd_topic').value, 10)
        self.create_subscription(String, 'detected_objects', self.on_detections, 10)

        self.csv_path = os.path.abspath(gp('log_csv').value)
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(['t_s', 'state', 'mode', 'err_norm', 'linear_cmd', 'angular_cmd',
                           'd_target_m', 'corridor_free_m'])

        rate = float(gp('control_rate_hz').value)
        self.create_timer(1.0 / rate, self.on_tick)
        mode = 'DRIVE (camera + LiDAR)' if self.drive else 'turn-only'
        self.get_logger().info(
            f"Following '{cfg.target_label}' | {mode} | kp={cfg.kp} fov={cfg.fov_deg} "
            f"| cmd on {gp('cmd_topic').value} | logging to {self.csv_path}")

    # --- inputs ---------------------------------------------------------------
    def on_detections(self, msg):
        now = time.time()
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(payload, dict):
            return  # old list-style message from the v1 detector
        found = self.ctrl.update_detections(payload, now)
        self.metrics.on_detection(now, payload.get('stamp'), found, self.ctrl.err)

    def on_scan(self, msg):
        self.scan = Scan(angle_min=msg.angle_min, angle_increment=msg.angle_increment,
                         ranges=list(msg.ranges), range_min=msg.range_min,
                         range_max=msg.range_max)
        self.drive_metrics.on_scan(self.scan)

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (p.x, p.y, yaw)

    # --- control loop ---------------------------------------------------------
    def on_tick(self):
        now = time.time()
        ang_cmd, state = self.ctrl.compute(now)
        lin_cmd, info = 0.0, {}

        # Two followers on one robot fight each other (e.g. an old launch still
        # running in another terminal), so say so loudly.
        count = getattr(self, 'count_publishers', None)
        if count is not None and count(self.get_parameter('cmd_topic').value) > 1:
            self.get_logger().warn(
                f"Another node is also publishing to {self.get_parameter('cmd_topic').value}. "
                "Is an older launch still running? Check with: ros2 node list",
                throttle_duration_sec=5.0)

        if self.drive:
            if self.scan is None and now - self.start > 3.0:
                self.get_logger().warn(
                    f"No LiDAR data on {self.get_parameter('scan_topic').value} yet",
                    throttle_duration_sec=5.0)
            pred_err = (self.ctrl.predicted_cx() - 0.5) if self.ctrl.last_cx is not None else 0.0
            lin_cmd, ang_cmd = self.planner.plan(state, ang_cmd, pred_err, self.scan, now,
                                                 self.ctrl.last_w_norm, self.pose)
            lin_cmd = clamp(lin_cmd, -MAX_LINEAR, MAX_LINEAR)
            ang_cmd = clamp(ang_cmd, -MAX_ANGULAR, MAX_ANGULAR)
            self.ctrl.set_applied_command(ang_cmd)     # keep ego-motion prediction honest
            info = self.planner.last
            self.drive_metrics.on_tick(now, info, pred_err, self.pose)
            if info.get('mode') != self.last_mode:
                d = info.get('d_target')
                extra = f" target {d:.2f} m via {info.get('via')}" if d is not None else ''
                self.get_logger().info(f"[{state}] -> {info.get('mode')}{extra}")
                self.last_mode = info.get('mode')
        elif state != self.metrics.prev_state:
            self.get_logger().info(f'-> {state} (err={self.ctrl.err:+.3f})')
        self.metrics.on_tick(now, state)

        twist = Twist()
        twist.linear.x = float(lin_cmd)
        twist.angular.z = float(ang_cmd)
        self.cmd_pub.publish(twist)

        d = info.get('d_target')
        free = info.get('free')
        self.csv.writerow([round(now - self.start, 3), state, info.get('mode', ''),
                           round(self.ctrl.err, 4), round(lin_cmd, 4), round(ang_cmd, 4),
                           round(d, 3) if d is not None else '',
                           round(free, 3) if free is not None else ''])

    def finish(self):
        """Stop the robot, close the log, write the summary."""
        try:
            self.cmd_pub.publish(Twist())
        except Exception:
            pass
        self.csv_file.close()
        summary = self.metrics.summary()
        if self.drive:
            summary.update(self.drive_metrics.summary(self.metrics.start))
        summary_path = os.path.splitext(self.csv_path)[0] + '_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print('\n===== Follower run summary =====')
        for k, v in summary.items():
            print(f'  {k:34s} {v}')
        print(f'  saved: {summary_path}\n  log:   {self.csv_path}')


def main(args=None):
    rclpy.init(args=args)
    node = Follower()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
