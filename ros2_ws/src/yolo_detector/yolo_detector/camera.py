"""
Virtual pan camera -- lets turtlesim "see".

Turtlesim has no camera, so on its own the follower would be open-loop:
the turtle could turn, but turning wouldn't change what YOLO sees.

This class fixes that. It places a real photo in a 360-degree world around
the robot and returns the slice of that world the robot is currently facing,
based on the turtle's heading (theta from /turtle1/pose). When the turtle
turns, the view pans and the target moves across the frame -- so the control
command really does change the next sensor reading. That's what makes the
loop closed.

Pure numpy, no ROS imports, so it can be tested offline.
"""

import math

import numpy as np


def wrap_to_pi(angle):
    """Wrap an angle in radians to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class VirtualPanCamera:
    def __init__(self, scene_bgr, view_width=640, fov_deg=60.0,
                 scene_bearing_deg=-70.0, background=114):
        """
        scene_bgr          : the photo to place in the world (H x W x 3, BGR)
        view_width         : width in pixels of the camera's output frame
        fov_deg            : horizontal field of view of the virtual camera
        scene_bearing_deg  : where the photo's centre sits relative to the
                             robot's start heading (negative = to the right,
                             like turtlesim's counter-clockwise-positive theta).
                             The default of -70 deg starts the photo out of
                             view, so the robot has to search for it first.
        """
        self.view_width = int(view_width)
        self.fov_rad = math.radians(fov_deg)
        self.px_per_rad = self.view_width / self.fov_rad
        self.scene_bearing = math.radians(scene_bearing_deg)

        h, w = scene_bgr.shape[:2]
        # Pad enough that any heading (+/- pi away) still gives an in-bounds crop.
        self.pad = int(math.ceil(math.pi * self.px_per_rad + self.view_width / 2)) + 2
        self.canvas = np.full((h, w + 2 * self.pad, 3), background, dtype=np.uint8)
        self.canvas[:, self.pad:self.pad + w] = scene_bgr
        self.scene_center_x = self.pad + w / 2.0

    def view(self, robot_theta):
        """Return the frame the robot sees when facing robot_theta (radians)."""
        # Angle from where the robot is facing to the centre of the scene.
        # Turning left (theta up) moves the scene right in the view, and
        # the view window moves left in the canvas.
        delta = wrap_to_pi(self.scene_bearing - robot_theta)
        x_center = self.scene_center_x + delta * self.px_per_rad
        x0 = int(round(x_center - self.view_width / 2.0))
        x0 = max(0, min(x0, self.canvas.shape[1] - self.view_width))
        return self.canvas[:, x0:x0 + self.view_width]


def image_msg_to_bgr(height, width, step, encoding, data):
    """Convert a sensor_msgs/Image's fields to an OpenCV BGR array.

    Done by hand instead of with cv_bridge: cv_bridge from apt is built against
    the system numpy and often breaks when pip has installed a newer one
    (ultralytics does).
    """
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(encoding)
    if channels is None:
        raise ValueError(f'unsupported image encoding: {encoding}')
    try:
        raw = np.frombuffer(data, dtype=np.uint8)   # bytes / array.array: no copy
    except TypeError:
        raw = np.asarray(data, dtype=np.uint8)      # plain list
    buf = raw.reshape(height, step)
    img = buf[:, :width * channels].reshape(height, width, channels)
    if encoding == 'rgb8':
        return np.ascontiguousarray(img[:, :, ::-1])
    if encoding == 'rgba8':
        return np.ascontiguousarray(img[:, :, 2::-1])
    if encoding == 'bgra8':
        return np.ascontiguousarray(img[:, :, :3])
    if encoding == 'mono8':
        return np.ascontiguousarray(np.repeat(img, 3, axis=2))
    return np.ascontiguousarray(img)
