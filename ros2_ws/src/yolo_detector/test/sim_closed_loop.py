"""
Offline closed-loop simulation -- no ROS needed.

Runs the same VirtualPanCamera, HeadingController and Metrics the ROS nodes
use, with real YOLOv8 inference, and integrates the robot heading like
turtlesim does (theta += w * dt). Two occlusions are injected to exercise
detection-loss handling: a short one (should COAST and recover) and a long
one (should SEARCH and re-acquire).

    python test/sim_closed_loop.py            # from the package root
"""

import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import ASSETS  # noqa: E402

from yolo_detector.camera import VirtualPanCamera, wrap_to_pi  # noqa: E402
from yolo_detector.control import ControllerConfig, HeadingController, Metrics  # noqa: E402

DT = 0.1               # control period (10 Hz), same as the follower node
DETECT_EVERY = int(os.environ.get('DETECT_EVERY', 2))   # 2 -> detector at 5 Hz, 5 -> 2 Hz
SIM_SECONDS = 16.0
OCCLUSIONS = [(8.0, 8.8), (11.0, 13.0)]   # (start, end) in sim seconds


def detect(model, frame):
    r = model(frame, conf=0.4, verbose=False)[0]
    dets = []
    for c, p, xywh in zip(r.boxes.cls, r.boxes.conf, r.boxes.xywh):
        cx, cy, w, h = (float(v) for v in xywh)
        dets.append({'label': r.names[int(c)], 'confidence': float(p),
                     'cx': cx, 'cy': cy, 'w': w, 'h': h})
    return dets


def main(cfg=None, verbose=True):
    scene = cv2.imread(str(ASSETS / 'bus.jpg'))
    cam = VirtualPanCamera(scene, view_width=640, fov_deg=60.0, scene_bearing_deg=-70.0)
    model = YOLO('yolov8n.pt')
    ctrl = HeadingController(cfg or ControllerConfig())
    bearings = []   # world bearing of the tracked target, to detect identity switches
    metrics = Metrics(fov_deg=60.0)

    theta = 0.0
    pending = None   # (deliver_at_sim_time, payload)
    log = []
    for k in range(int(SIM_SECONDS / DT)):
        t = k * DT
        if k % DETECT_EVERY == 0:
            frame = cam.view(theta)
            if any(a <= t < b for a, b in OCCLUSIONS):
                frame = frame.copy()
                frame[:] = 114   # simulated occlusion: camera sees nothing
            t0 = time.perf_counter()
            dets = detect(model, frame)
            infer = time.perf_counter() - t0
            pending = (t + infer, {'stamp': t, 'frame_w': frame.shape[1],
                                   'frame_h': frame.shape[0], 'detections': dets})
        if pending and pending[0] <= t + 1e-9:
            payload = pending[1]
            found = ctrl.update_detections(payload, t)
            if found:
                # bearing of target in the world = heading + angular offset in the frame
                bearings.append((t, math.degrees(theta) - ctrl.err * 60.0))
            metrics.on_detection(t, payload['stamp'], found, ctrl.err)
            pending = None

        cmd, state = ctrl.compute(t)
        if verbose and state != metrics.prev_state:
            print(f't={t:5.1f}s  -> {state:9s} err={ctrl.err:+.3f} theta={math.degrees(theta):+7.1f}deg')
        metrics.on_tick(t, state)
        log.append((t, state, ctrl.err, cmd, theta))
        theta = wrap_to_pi(theta + cmd * DT)

    # Same person => bearing stays put. A jump between consecutive sightings
    # (outside the occlusion gaps) means association switched to someone else.
    switches = sum(1 for (t0, b0), (t1, b1) in zip(bearings, bearings[1:])
                   if t1 - t0 < 0.5 and abs(b1 - b0) > 4.0)
    summary = metrics.summary()
    summary['identity_switches'] = switches
    if verbose:
        print(json.dumps(summary, indent=2))
    final = [abs(e) for (t, s, e, c, th) in log if t > SIM_SECONDS - 1.5 and s == "TRACKING"]
    print('final |err| (last 1.5 s):', round(max(final), 3) if final else None,
          '| final heading:', round(math.degrees(theta), 1), 'deg')
    return summary


if __name__ == '__main__':
    main()
