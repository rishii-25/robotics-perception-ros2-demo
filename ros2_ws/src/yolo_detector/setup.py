from setuptools import find_packages, setup

package_name = 'yolo_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rishik',
    maintainer_email='rishik25t4@gmail.com',
    description='YOLOv8 perception node publishing detections to a ROS2 topic, with a reactor node demonstrating perception-to-decision',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector = yolo_detector.detector:main',
            'reactor = yolo_detector.reactor:main',
        ],
    },
)
