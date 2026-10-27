from glob import glob

from setuptools import find_packages, setup

package_name = 'yolo_detector'

setup(
    name=package_name,
    version='0.3.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rishik',
    maintainer_email='rishik25t4@gmail.com',
    description='YOLOv8 perception node, decision node, and closed-loop target follower (turtlesim and TurtleBot3/Gazebo) for ROS2',
    license='MIT',
    entry_points={
        'console_scripts': [
            'detector = yolo_detector.detector:main',
            'reactor = yolo_detector.reactor:main',
            'follower = yolo_detector.follower:main',
        ],
    },
)
