"""
3D version: TurtleBot3 Waffle in Gazebo follows a person, avoiding the pillars.

    Gazebo camera --(/camera/image_raw)--> detector (YOLOv8) --(detected_objects)--+
    Gazebo LiDAR  --(/scan)-----------------------------------------------------> follower --(/cmd_vel)--> robot
    Gazebo odom   --(/odom)----------------------------------------------------->

Usage (after adding the person_standing model to ~/.gazebo/models):
    ros2 launch yolo_detector tb3_follow.launch.py
    ros2 launch yolo_detector tb3_follow.launch.py person_x:=1.5 person_y:=0.6     # harder spot
    ros2 launch yolo_detector tb3_follow.launch.py start_gazebo:=false             # Gazebo already up
    ros2 launch yolo_detector tb3_follow.launch.py conf:=0.25    # if YOLO misses the rendered person
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

WAFFLE_CAMERA_FOV_DEG = 62.2   # horizontal_fov 1.085595 rad in the Waffle's model


def generate_launch_description():
    args = {
        'start_gazebo': 'true',
        'person_model': os.path.expanduser('~/.gazebo/models/person_standing/model.sdf'),
        'person_x': '1.5',
        'person_y': '-0.55',
        'person_yaw': '3.14159',
        'target_label': 'person',
        'conf': '0.4',
        'follow_distance': '0.8',
        'max_linear': '0.18',
        'debug_image': '',
        'log_csv': 'follower_log.csv',
    }
    declared = [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
    lc = {k: LaunchConfiguration(k) for k in args}

    def as_float(name):
        return ParameterValue(lc[name], value_type=float)

    def as_str(name):
        return ParameterValue(lc[name], value_type=str)

    tb3_world = os.path.join(get_package_share_directory('turtlebot3_gazebo'),
                             'launch', 'turtlebot3_world.launch.py')

    return LaunchDescription(declared + [
        SetEnvironmentVariable('TURTLEBOT3_MODEL', 'waffle'),   # the one with a camera
        IncludeLaunchDescription(PythonLaunchDescriptionSource(tb3_world),
                                 condition=IfCondition(lc['start_gazebo'])),
        Node(
            package='gazebo_ros', executable='spawn_entity.py', name='spawn_person',
            output='screen',
            arguments=['-entity', 'person', '-file', lc['person_model'],
                       '-x', lc['person_x'], '-y', lc['person_y'], '-z', '0.0',
                       '-Y', lc['person_yaw'], '-timeout', '120']),
        Node(
            package='yolo_detector', executable='detector', name='yolo_detector',
            output='screen',
            parameters=[{
                'image_topic': '/camera/image_raw',
                'rate_hz': 5.0,
                'conf': as_float('conf'),
                'debug_image': as_str('debug_image'),
            }]),
        Node(
            package='yolo_detector', executable='follower', name='target_follower',
            output='screen',
            parameters=[{
                'drive': True,
                'cmd_topic': '/cmd_vel',
                'fov_deg': WAFFLE_CAMERA_FOV_DEG,
                'target_label': as_str('target_label'),
                'follow_distance': as_float('follow_distance'),
                'max_linear': as_float('max_linear'),
                'log_csv': as_str('log_csv'),
            }]),
    ])
