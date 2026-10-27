"""
2D replica of the TurtleBot3 world for testing the follower without Gazebo.

  * 3x3 grid of pillars (1.1 m apart) inside hexagonal walls
  * 360-beam LiDAR ray-cast against pillars, walls and the person (5 Hz, noisy)
  * camera with the Waffle's 62.2 deg FOV; the person is lost when pillars
    block the line of sight; detections at 5 Hz with 0.1 s latency and misses
  * unicycle robot integrated at 20 Hz, driven by the REAL HeadingController,
    MotionPlanner and Metrics used in the ROS node

Runs scripted scenarios plus N random trials and reports task-completion
rate, collisions and timing.

    python test/sim_tb3_world.py            # scenarios + 30 random trials
    python test/sim_tb3_world.py 100        # scenarios + 100 random trials
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yolo_detector.control import ControllerConfig, HeadingController  # noqa: E402
from yolo_detector.navigation import MotionConfig, MotionPlanner, Scan  # noqa: E402

PILLARS = [(x, y) for x in (-1.1, 0.0, 1.1) for y in (-1.1, 0.0, 1.1)]
PILLAR_R = 0.15
HEX_R = 2.9                      # wall circumradius (vertices on +/-x)
PERSON_R = 0.15                  # what the LiDAR hits (legs)
PERSON_HALF_W = 0.22             # what the camera sees (body width / 2)
ROBOT_R = 0.21                   # Waffle footprint, approximated as a circle
FOV = math.radians(62.2)
IMG_W = 1920
DT = 0.05

HEX = [(HEX_R * math.cos(k * math.pi / 3), HEX_R * math.sin(k * math.pi / 3)) for k in range(6)]
WALLS = [(HEX[k], HEX[(k + 1) % 6]) for k in range(6)]


def ray_circle(ox, oy, dx, dy, cx, cy, r):
    fx, fy = ox - cx, oy - cy
    b = fx * dx + fy * dy
    c = fx * fx + fy * fy - r * r
    disc = b * b - c
    if disc < 0:
        return None
    t = -b - math.sqrt(disc)
    return t if t > 0 else None


def ray_segment(ox, oy, dx, dy, a, b):
    ex, ey = b[0] - a[0], b[1] - a[1]
    den = dx * ey - dy * ex
    if abs(den) < 1e-12:
        return None
    t = ((a[0] - ox) * ey - (a[1] - oy) * ex) / den
    u = ((a[0] - ox) * dy - (a[1] - oy) * dx) / den
    return t if t > 0 and 0 <= u <= 1 else None


def dist_to_segment(px, py, a, b):
    ex, ey = b[0] - a[0], b[1] - a[1]
    t = clamp(((px - a[0]) * ex + (py - a[1]) * ey) / (ex * ex + ey * ey), 0, 1)
    return math.hypot(px - (a[0] + t * ex), py - (a[1] + t * ey))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def clearance(x, y, person):
    """Distance from a point to the nearest obstacle surface."""
    d = min(math.hypot(x - px, y - py) - PILLAR_R for px, py in PILLARS)
    d = min(d, min(dist_to_segment(x, y, a, b) for a, b in WALLS))
    if person is not None:
        d = min(d, math.hypot(x - person[0], y - person[1]) - PERSON_R)
    return d


def lidar(x, y, th, person, rng):
    sx, sy = x - 0.064 * math.cos(th), y - 0.064 * math.sin(th)
    ranges = []
    for i in range(360):
        a = th + math.radians(i)
        dx, dy = math.cos(a), math.sin(a)
        best = math.inf
        for px, py in PILLARS:
            t = ray_circle(sx, sy, dx, dy, px, py, PILLAR_R)
            if t is not None and t < best:
                best = t
        t = ray_circle(sx, sy, dx, dy, person[0], person[1], PERSON_R)
        if t is not None and t < best:
            best = t
        for a_, b_ in WALLS:
            t = ray_segment(sx, sy, dx, dy, a_, b_)
            if t is not None and t < best:
                best = t
        ranges.append(best + rng.gauss(0, 0.01) if best <= 3.5 else math.inf)
    return Scan(angle_min=0.0, angle_increment=math.radians(1.0), ranges=ranges)


def camera(x, y, th, person, rng, miss_p=0.05):
    """Return a detector payload; the person is seen if >= 2 of 5 body points are visible."""
    # Waffle camera: 64 mm ahead, 65 mm right of centre.
    cx_ = x + 0.064 * math.cos(th) + 0.065 * math.sin(th)
    cy_ = y + 0.064 * math.sin(th) - 0.065 * math.cos(th)
    vx, vy = person[0] - cx_, person[1] - cy_
    dist = math.hypot(vx, vy)
    nx, ny = -vy / dist, vx / dist                 # body points across the person's width
    visible = []
    for k in (-1, -0.5, 0, 0.5, 1):
        bx, by = person[0] + k * PERSON_HALF_W * nx, person[1] + k * PERSON_HALF_W * ny
        bearing = math.atan2(by - cy_, bx - cx_) - th
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        if abs(bearing) > FOV / 2:
            continue
        d = math.hypot(bx - cx_, by - cy_)
        dx, dy = (bx - cx_) / d, (by - cy_) / d
        blocked = any((t := ray_circle(cx_, cy_, dx, dy, px, py, PILLAR_R)) is not None and t < d
                      for px, py in PILLARS)
        if not blocked:
            visible.append(bearing)
    dets = []
    if len(visible) >= 2 and dist < 6.0 and rng.random() > miss_p:
        b = sum(visible) / len(visible)
        cx = (0.5 - b / FOV) * IMG_W + rng.gauss(0, 6)
        # Box width shrinks when part of the body is hidden, like a real detector's.
        f = IMG_W / (2 * math.tan(FOV / 2))
        w = f * 2 * PERSON_HALF_W * (len(visible) / 5) / dist * rng.uniform(0.85, 1.15)
        dets.append({'label': 'person', 'confidence': 0.85, 'cx': cx, 'cy': 540.0,
                     'w': min(w, IMG_W), 'h': 1080.0})
    return dets


def run(start, person, seconds=60.0, seed=0, verbose=False, motion_cfg=None):
    rng = random.Random(seed)
    x, y, th = start
    ctrl = HeadingController(ControllerConfig(fov_deg=math.degrees(FOV)))
    nav = MotionPlanner(motion_cfg or MotionConfig())
    scan = None
    pending = None
    collisions, in_collision = 0, False
    min_clear = math.inf
    reached_since, t_done = None, None
    lin = ang = 0.0
    modes = {}
    for k in range(int(seconds / DT)):
        t = k * DT
        if k % 4 == 0:                                   # 5 Hz LiDAR
            scan = lidar(x, y, th, person, rng)
        if k % 4 == 0:                                   # 5 Hz camera, 0.1 s latency
            pending = (t + 0.1, {'stamp': t, 'frame_w': IMG_W, 'frame_h': 1080,
                                 'detections': camera(x, y, th, person, rng)})
        if pending and t >= pending[0] - 1e-9:
            ctrl.update_detections(pending[1], t)
            pending = None
        if k % 2 == 0:                                   # 10 Hz control
            ang_track, state = ctrl.compute(t)
            pred_err = (ctrl.predicted_cx() - 0.5) if ctrl.last_cx is not None else 0.0
            lin, ang = nav.plan(state, ang_track, pred_err, scan, t, ctrl.last_w_norm,
                                pose=(x, y, th))    # like /odom in Gazebo
            lin = clamp(lin, -0.26, 0.26)
            ang = clamp(ang, -1.82, 1.82)
            ctrl.set_applied_command(ang)
            m = nav.last.get('mode')
            modes[m] = modes.get(m, 0) + 1

        nx = x + lin * math.cos(th) * DT
        ny = y + lin * math.sin(th) * DT
        th = (th + ang * DT + math.pi) % (2 * math.pi) - math.pi
        c = clearance(nx, ny, person) - ROBOT_R
        if c < 0:                                        # contact: count it, don't move into it
            if not in_collision:
                collisions += 1
            in_collision = True
        else:
            in_collision = False
            x, y = nx, ny
        min_clear = min(min_clear, clearance(x, y, person) - ROBOT_R)

        # Task complete: within follow distance (+/- 0.25 m) and facing the person, for 1 s.
        gap = math.hypot(person[0] - x, person[1] - y) - PERSON_R - ROBOT_R
        bearing = (math.atan2(person[1] - y, person[0] - x) - th + math.pi) % (2 * math.pi) - math.pi
        ok = abs(gap + ROBOT_R - 0.8) < 0.25 and abs(bearing) < math.radians(10)
        if ok:
            reached_since = t if reached_since is None else reached_since
            if t - reached_since >= 1.0 and t_done is None:
                t_done = t
                if not verbose:
                    break
        else:
            reached_since = None
    return {'done': t_done is not None, 't_done': t_done, 'collisions': collisions,
            'min_clearance': round(min_clear, 3), 'final': (round(x, 2), round(y, 2)),
            'modes': modes}


def random_free_point(rng, margin):
    while True:
        x, y = rng.uniform(-2.6, 2.6), rng.uniform(-2.6, 2.6)
        inside = all(dist_to_segment(x, y, a, b) > margin for a, b in WALLS) and \
            math.hypot(x, y) < HEX_R * 0.866
        if inside and clearance(x, y, None) > margin:
            return x, y


def random_trials(n, motion_cfg=None, seed=42):
    rng = random.Random(seed)
    results = []
    for i in range(n):
        s = random_free_point(rng, 0.45)
        while True:
            p = random_free_point(rng, 0.45)
            if math.hypot(p[0] - s[0], p[1] - s[1]) > 1.5:
                break
        results.append(run((s[0], s[1], rng.uniform(-math.pi, math.pi)), p, seed=i,
                           motion_cfg=motion_cfg))
    return results


def summarize(results):
    done = [r for r in results if r['done']]
    ts = sorted(r['t_done'] for r in done)
    return {'n': len(results), 'completion_pct': round(100 * len(done) / len(results), 1),
            'trials_with_collision': sum(1 for r in results if r['collisions']),
            'median_time_s': round(ts[len(ts) // 2], 1) if ts else None,
            'min_clearance_m': round(min(r['min_clearance'] for r in results), 3)}


def main():
    scenarios = {
        'straight corridor (TB3 spawn)': ((-2.0, -0.55, 0.0), (1.6, -0.55)),
        'pillars in the way':            ((-2.0, 0.05, 0.0), (1.7, 0.3)),
        'person behind, must search':    ((-2.0, -0.5, math.pi / 2), (0.55, -1.65)),
        'diagonal through the grid':     ((-1.5, -1.5, 0.8), (1.6, 1.5)),
    }
    print('Scripted scenarios:')
    for name, (start, person) in scenarios.items():
        r = run(start, person)
        print(f"  {name:32s} done={r['done']!s:5s} t={r['t_done']} collisions={r['collisions']} "
              f"min_clear={r['min_clearance']} modes={r['modes']}")

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    rng = random.Random(42)
    results = []
    for i in range(n):
        s = random_free_point(rng, 0.45)
        while True:
            p = random_free_point(rng, 0.45)
            if math.hypot(p[0] - s[0], p[1] - s[1]) > 1.5:
                break
        results.append(run((s[0], s[1], rng.uniform(-math.pi, math.pi)), p, seed=i))
    done = [r for r in results if r['done']]
    print(f'\nRandom trials: {n}')
    print(f'  task completion rate : {100 * len(done) / n:.0f}%')
    print(f"  trials with collision: {sum(1 for r in results if r['collisions'])}")
    if done:
        ts = sorted(r['t_done'] for r in done)
        print(f'  time to complete     : median {ts[len(ts) // 2]:.1f} s, max {ts[-1]:.1f} s')
    print(f"  min clearance overall: {min(r['min_clearance'] for r in results):.3f} m")
    fails = [(i, r) for i, r in enumerate(results) if not r['done'] or r['collisions']]
    for i, r in fails[:8]:
        print(f'  FAIL trial {i}: {r}')
    return results


if __name__ == '__main__':
    main()
