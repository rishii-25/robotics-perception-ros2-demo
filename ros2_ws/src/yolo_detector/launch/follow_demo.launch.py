"""
One command to run the whole closed loop:

    turtlesim  --(/turtle1/pose)-->  detector (virtual pan camera + YOLOv8)
        ^                                   |
        |                          (detected_objects)
        |                                   v
        +-------(/turtle1/cmd_vel)------  follower (P-control)

Usage:
    ros2 launch yolo_detector follow_demo.launch.py
    ros2 launch yolo_detector follow_demo.launch.py target_label:=bus kp:=1.5
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = {
        'source': 'sample',
        'target_label': 'person',
        'kp': '2.0',
        'scene_bearing_deg': '-70.0',
        'fov_deg': '60.0',
        'debug_image': '',
        'log_csv': 'follower_log.csv',
    }
    declared = [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
    lc = {k: LaunchConfiguration(k) for k in args}

    def as_float(name):
        return ParameterValue(lc[name], value_type=float)

    def as_str(name):
        # Explicit type: an empty string or '0' would otherwise be parsed as YAML.
        return ParameterValue(lc[name], value_type=str)

    return LaunchDescription(declared + [
        Node(package='turtlesim', executable='turtlesim_node', name='sim'),
        Node(
            package='yolo_detector', executable='detector', name='yolo_detector',
            output='screen',
            parameters=[{
                'source': as_str('source'),
                'virtual_pan': True,
                'scene_bearing_deg': as_float('scene_bearing_deg'),
                'fov_deg': as_float('fov_deg'),
                'debug_image': as_str('debug_image'),
            }]),
        Node(
            package='yolo_detector', executable='follower', name='target_follower',
            output='screen',
            parameters=[{
                'target_label': as_str('target_label'),
                'kp': as_float('kp'),
                'fov_deg': as_float('fov_deg'),
                'log_csv': as_str('log_csv'),
            }]),
    ])
