from setuptools import find_packages, setup

package_name = 'minimal_demo'

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
    description='Minimal ROS2 publisher/subscriber demo for robotics track prep',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'talker = minimal_demo.talker:main',
            'listener = minimal_demo.listener:main',
        ],
    },
)
