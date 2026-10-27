"""
Camera + LiDAR motion layer for a real (or Gazebo) mobile robot -- no ROS imports.

The heading controller (control.py) says where the target is from the camera.
This module turns that into (linear, angular) velocity using the 360-degree
LiDAR, so the robot reaches the person without hitting anything:

  * Sensor fusion. The LiDAR scan is grouped into objects (clusters of
    neighbouring returns; long ones are walls). The camera detection is
    matched to the object whose centre lines up with it as seen from the
    camera's own position (13 cm from the LiDAR on a Waffle). When two
    objects line up, e.g. a pillar right in front of the person, the camera's
    rough range from box width and the remembered position break the tie.
  * Target memory. The person's position is stored in the odometry frame each
    time a trusted LiDAR range is available. When they leave the 62-degree
    camera view (e.g. while going around a pillar), the robot keeps heading
    for the remembered spot, which brings them back into view.
  * Heading selection. Every tick, candidate headings all around the robot are
    checked for a clear corridor as wide as the robot. The robot steers toward
    the clear heading closest to the goal, with a small penalty for switching
    sides so it commits to one way around an obstacle. Going around a pillar
    is a smooth curve rather than a stop-turn-stop stutter.
  * Safety. Never drive forward with anything in the robot's own corridor
    closer than stop_distance, whatever the camera says.
  * Exploration. With no memory, after a full 360-degree search with nothing
    found, drive toward the most open heading for a short leg, then search again.
"""

import math
from dataclasses import dataclass


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Scan:
    """The parts of sensor_msgs/LaserScan we need (angles in radians, CCW+)."""
    angle_min: float
    angle_increment: float
    ranges: list
    range_min: float = 0.12
    range_max: float = 3.5

    def points(self):
        """Yield (bearing, range) for valid returns; 'nothing hit' is skipped."""
        for i, r in enumerate(self.ranges):
            if r != r or r < self.range_min or r > self.range_max:   # NaN / too short / inf
                continue
            yield self.angle_min + i * self.angle_increment, r

    def sector_min(self, center, half_width):
        """Closest return within +/- half_width of center, or None if nothing there."""
        best = None
        for a, r in self.points():
            if abs(wrap(a - center)) <= half_width and (best is None or r < best):
                best = r
        return best

    def clusters(self, jump=0.15, max_gap=2):
        """Group returns into objects. Returns dicts with bearing (centre), range
        (nearest point), width (m) and foreground (True if whatever is just
        beyond each end is farther away or empty -- a free-standing object,
        not a piece of wall partly hidden by something in front of it).
        A new object starts at a range jump of more than `jump` m or after
        more than `max_gap` empty beams."""
        n = len(self.ranges)
        valid = [not (r != r or r < self.range_min or r > self.range_max) for r in self.ranges]
        if not any(valid):
            return []
        # Start just after a break so an object straddling index 0 isn't split.
        start = 0
        for i in range(n):
            if not valid[i] or (valid[i - 1] and abs(self.ranges[i] - self.ranges[i - 1]) > jump):
                start = i
                break
        groups, cur, gap, prev_r = [], [], 0, None
        for k in range(n):
            i = (start + k) % n
            if not valid[i]:
                gap += 1
                if gap > max_gap and cur:
                    groups.append(cur)
                    cur, prev_r = [], None
                continue
            r = self.ranges[i]
            if cur and prev_r is not None and abs(r - prev_r) > jump:
                groups.append(cur)
                cur = []
            cur.append(i)
            prev_r, gap = r, 0
        if cur:
            groups.append(cur)
        out = []
        for g in groups:
            angles = [self.angle_min + i * self.angle_increment for i in g]
            ref = angles[0]
            mid = wrap(ref + sum(wrap(a - ref) for a in angles) / len(angles))
            r_min = min(self.ranges[i] for i in g)
            x0 = self.ranges[g[0]] * math.cos(angles[0])
            y0 = self.ranges[g[0]] * math.sin(angles[0])
            x1 = self.ranges[g[-1]] * math.cos(angles[-1])
            y1 = self.ranges[g[-1]] * math.sin(angles[-1])
            def beyond(i, end_r):
                nb = self.ranges[i % n]
                return not valid[i % n] or nb > end_r
            foreground = (beyond(g[0] - 1, self.ranges[g[0]]) and
                          beyond(g[-1] + 1, self.ranges[g[-1]]))
            out.append({'bearing': mid, 'range': r_min, 'width': math.hypot(x1 - x0, y1 - y0),
                        'n': len(g), 'foreground': foreground})
        return out


@dataclass
class Robot:
    """TurtleBot3 Waffle: 281 x 306 mm footprint; LDS 64 mm behind centre;
    camera 64 mm ahead and 65 mm right of centre (from its URDF)."""
    half_length: float = 0.140
    half_width: float = 0.153
    scanner_x: float = -0.064
    camera_x: float = 0.064
    camera_y: float = -0.065


@dataclass
class MotionConfig:
    max_linear: float = 0.18          # m/s (Waffle tops out at 0.26)
    detour_speed: float = 0.12        # m/s while steering around something
    follow_distance: float = 0.8      # m from the LiDAR: stop this far from the target
    approach_ramp: float = 0.6        # m: ease off over this distance before stopping
    corridor_margin: float = 0.08     # m added to each side of the robot's width
    stop_distance: float = 0.12       # m from the front edge: never drive closer
    slow_distance: float = 0.50       # m from the front edge: start slowing
    lookahead: float = 0.8            # m of clear corridor a detour heading needs
    heading_step_deg: float = 10.0    # spacing of candidate headings
    turn_gain: float = 2.0            # rad/s per rad toward the chosen heading
    switch_penalty: float = 0.35      # rad-equivalent cost of switching detour side
    align_tolerance: float = 0.25     # |err| (fraction of frame) where forward speed hits 0
    camera_fov_deg: float = 62.2      # Waffle camera horizontal FOV
    target_window_deg: float = 4.0    # LiDAR window around the camera bearing
    person_radius: float = 0.15       # what the LiDAR sees of a person (legs)
    person_width_m: float = 0.45      # for the camera's rough range estimate
    max_object_width: float = 0.6     # m: wider LiDAR clusters are walls, not people
    occluder_ratio: float = 0.4       # object closer than ratio * camera range = in front of
                                      # the person, not the person (0 disables)
    memory_tiebreak: float = 1.0      # matching penalty for disagreeing with memory
    memory_jump: float = 0.6          # m: a bigger jump needs 3 agreeing readings
    explore_after_rad: float = 2 * math.pi   # search this much rotation before exploring
    explore_distance: float = 0.8     # m per exploration leg
    memory_timeout: float = 20.0      # s to trust a remembered target position
    give_up_after: float = 4.0        # s at the remembered spot without seeing the target


class MotionPlanner:
    def __init__(self, cfg: MotionConfig, robot: Robot = Robot()):
        self.cfg = cfg
        self.robot = robot
        self.pose = (0.0, 0.0, 0.0)   # odometry frame; from /odom, else dead-reckoned
        self.mem = None                # (x, y, time) of the target in the odometry frame
        self.at_mem_since = None
        self.mem_strikes = 0           # consecutive readings disagreeing with memory
        self.arrived = False           # at follow distance (with hysteresis)
        self.detour_side = 0.0         # sign of the last detour relative to the goal
        self.search_rot = 0.0          # rad rotated while searching
        self.explore = None            # {'bearing': rad (robot frame), 'left': m}
        self.last_t = None
        self.prev = (0.0, 0.0)
        self.last = {}

    # --- geometry -------------------------------------------------------------
    def _points_xy(self, scan):
        """LiDAR returns in the robot-centre frame."""
        sx = self.robot.scanner_x
        return [(r * math.cos(a) + sx, r * math.sin(a)) for a, r in scan.points()]

    def _free_toward(self, pts, heading, range_max):
        """Clear distance ahead of the front edge if the robot faced `heading`."""
        hw = self.robot.half_width + self.cfg.corridor_margin
        ch, sh = math.cos(heading), math.sin(heading)
        best = range_max
        for x0, y0 in pts:
            px = x0 * ch + y0 * sh
            if px > 0 and abs(-x0 * sh + y0 * ch) < hw:
                best = min(best, px - self.robot.half_length)
        return best

    def corridor_free(self, scan, heading=0.0):
        return self._free_toward(self._points_xy(scan), heading, scan.range_max)

    def fused_target_range(self, scan, pred_err, width_norm):
        """(range, bearing, source) of the target from camera bearing + LiDAR objects.

        range: from the LiDAR to the nearest point of the matched object.
        bearing: from the robot centre.
        """
        c, rb = self.cfg, self.robot
        half_fov = math.radians(c.camera_fov_deg) / 2
        b_cam = -pred_err * 2 * half_fov
        d_cam = None
        if width_norm and width_norm > 0.02:
            d_cam = c.person_width_m / (2 * math.tan(half_fov) * width_norm)
        d_mem = self._mem_range()

        best = None
        for obj in scan.clusters():
            if obj['width'] > c.max_object_width or not obj['foreground']:
                continue                                  # wall, or wall partly hidden
            # Object centre in the robot frame, then its bearing from the CAMERA.
            r_mid = obj['range'] + c.person_radius
            px = rb.scanner_x + r_mid * math.cos(obj['bearing'])
            py = r_mid * math.sin(obj['bearing'])
            b_from_cam = math.atan2(py - rb.camera_y, px - rb.camera_x)
            err = abs(wrap(b_from_cam - b_cam))
            tol = max(math.radians(6), math.atan2(0.25, max(r_mid, 0.3)))
            if err > tol:
                continue
            if d_cam is not None and r_mid < c.occluder_ratio * d_cam:
                # A person this close would look at least twice as big: this
                # object is standing in front of them (e.g. a pillar).
                continue
            score = err / tol
            if d_cam is not None:
                score += 0.5 * abs(math.log(max(obj['range'], 0.1) / d_cam))
            if d_mem is not None and abs(r_mid - d_mem) > 0.5:
                score += c.memory_tiebreak
            if best is None or score < best[0]:
                best = (score, obj['range'], math.atan2(py, px))

        if best is None:
            # Nothing on the LiDAR lines up: aim along the camera ray, camera range.
            d = d_cam if d_cam is not None else scan.range_max
            px, py = rb.camera_x + d * math.cos(b_cam), rb.camera_y + d * math.sin(b_cam)
            return d, math.atan2(py, px), 'camera'
        return best[1], best[2], 'lidar'

    def _mem_range(self):
        if self.mem is None:
            return None
        x, y, _ = self.pose
        return math.hypot(self.mem[0] - x, self.mem[1] - y)

    # --- main entry -----------------------------------------------------------
    def plan(self, state, ang_track, pred_err, scan, now, width_norm=None, pose=None):
        """Return (linear, angular) for this control tick.

        state      : controller state (TRACKING / COASTING / SEARCHING / WAITING)
        ang_track  : the heading controller's angular command
        pred_err   : predicted normalized horizontal error of the target
        scan       : Scan, or None before the first LiDAR message
        width_norm : tracked box width / frame width (rough range cue)
        pose       : (x, y, yaw) from odometry; dead-reckoned from commands if None
        """
        dt = 0.0 if self.last_t is None else clamp(now - self.last_t, 0.0, 0.5)
        self.last_t = now
        plin, pang = self.prev
        if pose is not None:
            self.pose = pose
        else:
            x, y, th = self.pose
            self.pose = (x + plin * math.cos(th) * dt, y + plin * math.sin(th) * dt,
                         wrap(th + pang * dt))
        if self.explore is not None:
            self.explore['bearing'] = wrap(self.explore['bearing'] - pang * dt)
            self.explore['left'] -= plin * dt
        if self.last.get('mode') in ('SEARCH', 'EXPLORE'):
            self.search_rot += abs(pang) * dt
        else:
            self.search_rot, self.explore = 0.0, None

        lin, ang, info = self._plan(state, ang_track, pred_err, scan, now, width_norm)
        self.prev = (lin, ang)
        self.last = info
        return lin, ang

    def _plan(self, state, ang_track, pred_err, scan, now, width_norm):
        c = self.cfg
        if scan is None:
            return 0.0, ang_track, {'mode': 'NO_SCAN'}     # turning in place is safe
        pts = self._points_xy(scan)
        free = self._free_toward(pts, 0.0, scan.range_max)
        v_clear = c.max_linear * clamp((free - c.stop_distance)
                                       / (c.slow_distance - c.stop_distance), 0.0, 1.0)
        fov = math.radians(c.camera_fov_deg)

        # 1. Where is the goal?
        if state == 'TRACKING':
            d_goal, bearing, source = self.fused_target_range(scan, pred_err, width_norm)
            if source == 'lidar':
                self._remember(d_goal, bearing, now)
            goal_err = pred_err
        else:
            if self.mem is not None and now - self.mem[2] > c.memory_timeout:
                self.mem = None
            if self.mem is None:
                return self._search(state, ang_track, pts, scan, free, v_clear)
            x, y, th = self.pose
            dx, dy = self.mem[0] - x, self.mem[1] - y
            d_goal, bearing, source = math.hypot(dx, dy), wrap(math.atan2(dy, dx) - th), 'memory'
            goal_err = -bearing / fov
            if d_goal <= c.follow_distance + 0.2 and abs(bearing) < 0.3:
                # At the remembered spot, facing it, still not seen: they moved.
                self.at_mem_since = self.at_mem_since or now
                if now - self.at_mem_since > c.give_up_after:
                    self.mem, self.at_mem_since = None, None
                    return self._search(state, ang_track, pts, scan, free, v_clear)

        # 2. Arrived: stand still and keep facing the person. (Near the goal the
        #    person themselves end the corridor, so re-planning there would just
        #    flicker between "clear" and "blocked" on sensor noise.)
        # Hysteresis: arrive at follow + 0.1 m, only leave again past follow + 0.25 m.
        self.arrived = d_goal <= c.follow_distance + (0.25 if self.arrived else 0.1)
        if self.arrived:
            self.detour_side = 0.0
            ang = ang_track if state == 'TRACKING' else clamp(c.turn_gain * bearing, -1.2, 1.2)
            return 0.0, ang, {'mode': 'HOLD', 'free': free, 'd_target': d_goal,
                              'heading': bearing, 'via': source}

        # 3. Pick a heading: the clear one closest to the goal.
        # Toward the goal itself, the person ends the corridor, so only the
        # part in front of them has to be clear.
        front = self.robot.half_length - self.robot.scanner_x
        need_goal = clamp(d_goal - front - c.person_radius - 0.1, 0.0, c.lookahead)
        best = None
        step = math.radians(c.heading_step_deg)
        candidates = [bearing] + [k * step for k in range(-int(math.pi / step), int(math.pi / step))]
        for h in candidates:
            off = wrap(h - bearing)
            direct = abs(off) < step / 2
            need = need_goal if direct else c.lookahead
            if self._free_toward(pts, h, scan.range_max) < need:
                continue
            cost = abs(off) + 0.2 * abs(h)
            if not direct and self.detour_side and math.copysign(1.0, off) != self.detour_side:
                cost += c.switch_penalty
            if best is None or cost < best[0]:
                best = (cost, h, direct)

        if best is None:
            # Boxed in: turn in place toward the goal, the safest thing to do.
            self.detour_side = 0.0
            return 0.0, clamp(c.turn_gain * bearing, -1.2, 1.2), {
                'mode': 'BLOCKED', 'free': free, 'd_target': d_goal, 'via': source}

        _, h, direct = best
        if direct:
            self.detour_side = 0.0
            ang = ang_track if state == 'TRACKING' else clamp(c.turn_gain * bearing, -1.2, 1.2)
            align = clamp(1.0 - abs(goal_err) / c.align_tolerance, 0.0, 1.0)
            v_target = c.max_linear * clamp((d_goal - c.follow_distance) / c.approach_ramp, 0, 1)
            lin = min(v_target, v_clear) * align
            mode = 'APPROACH' if lin > 1e-3 else 'HOLD'
        else:
            self.detour_side = math.copysign(1.0, wrap(h - bearing))
            ang = clamp(c.turn_gain * h, -1.2, 1.2)
            lin = min(c.detour_speed, v_clear) * clamp(1.0 - abs(h) / 0.5, 0.0, 1.0)
            mode = 'DETOUR'

        if free < c.stop_distance:
            lin = 0.0                                   # hard safety stop
        return lin, ang, {'mode': mode, 'free': free, 'd_target': d_goal,
                          'heading': h, 'via': source}

    def _remember(self, d, bearing, now):
        x, y, th = self.pose
        # The LiDAR hits the front of the legs; aim for the person's centre.
        d += self.cfg.person_radius
        gx, gy = x + d * math.cos(th + bearing), y + d * math.sin(th + bearing)
        if self.mem is not None and now - self.mem[2] < self.cfg.memory_timeout:
            if math.hypot(gx - self.mem[0], gy - self.mem[1]) > self.cfg.memory_jump:
                # Outlier gate: one stray reading (wall behind the person, a
                # pillar beside them) mustn't overwrite a good memory.
                self.mem_strikes += 1
                if self.mem_strikes < 3:
                    return
            else:
                gx, gy = 0.7 * self.mem[0] + 0.3 * gx, 0.7 * self.mem[1] + 0.3 * gy
        self.mem_strikes = 0
        self.mem = (gx, gy, now)
        self.at_mem_since = None

    def _search(self, state, ang_track, pts, scan, free, v_clear):
        c = self.cfg
        if state != 'SEARCHING':                     # WAITING / COASTING with nothing remembered
            return 0.0, ang_track, {'mode': 'ROTATE_ONLY', 'free': free}
        if self.explore is None and self.search_rot >= c.explore_after_rad:
            step = math.radians(30)
            options = [(self._free_toward(pts, k * step, scan.range_max), -abs(k), k * step)
                       for k in range(-6, 6)]
            _, _, heading = max(options)
            self.explore = {'bearing': heading, 'left': c.explore_distance}
        if self.explore is not None:
            e = self.explore
            if e['left'] <= 0 or (abs(e['bearing']) < 0.3 and free < c.slow_distance):
                self.explore, self.search_rot = None, 0.0     # leg done: search again
                return 0.0, ang_track, {'mode': 'SEARCH', 'free': free}
            ang = clamp(c.turn_gain * e['bearing'], -1.0, 1.0)
            lin = min(c.detour_speed, v_clear) if abs(e['bearing']) < 0.3 else 0.0
            return lin, ang, {'mode': 'EXPLORE', 'free': free}
        return 0.0, ang_track, {'mode': 'SEARCH', 'free': free}


# --- collision check from the LiDAR (for metrics) ----------------------------

def body_extent(bearing, robot=Robot()):
    """Distance from the LiDAR to the footprint edge along a bearing."""
    c, s = math.cos(bearing), math.sin(bearing)
    if c > 1e-9:
        tx = (robot.half_length - robot.scanner_x) / c
    elif c < -1e-9:
        tx = (-robot.half_length - robot.scanner_x) / c
    else:
        tx = math.inf
    ty = robot.half_width / abs(s) if abs(s) > 1e-9 else math.inf
    return min(tx, ty)


def touching(scan, margin=0.03, robot=Robot()):
    """True if any LiDAR return is within `margin` of the robot's outline.

    The back edge is 76 mm from the LDS, under its 120 mm minimum range, so
    contact directly behind can't be seen.
    """
    return any(r < body_extent(a, robot) + margin for a, r in scan.points())


class DriveMetrics:
    """Evaluation numbers for a drive run: collisions, clearance, task completion,
    following-distance error, distance travelled, time per mode."""

    def __init__(self, follow_distance, tolerance=0.15, hold_for=1.0):
        self.follow = follow_distance
        self.tol = tolerance
        self.hold_for = hold_for
        self.collisions = 0
        self.touching = False
        self.min_clearance = math.inf
        self.reached_since = None
        self.t_complete = None
        self.follow_errs = []
        self.distance = 0.0
        self.last_pose = None
        self.mode_ticks = {}

    def on_scan(self, scan):
        clear = min((r - body_extent(a) for a, r in scan.points()), default=math.inf)
        self.min_clearance = min(self.min_clearance, clear)
        now_touching = touching(scan)
        if now_touching and not self.touching:
            self.collisions += 1
        self.touching = now_touching

    def on_tick(self, now, info, goal_err, pose):
        mode = info.get('mode', '?')
        self.mode_ticks[mode] = self.mode_ticks.get(mode, 0) + 1
        if pose is not None and self.last_pose is not None:
            self.distance += math.hypot(pose[0] - self.last_pose[0], pose[1] - self.last_pose[1])
        self.last_pose = pose
        d = info.get('d_target')
        at_goal = (info.get('via') == 'lidar' and d is not None
                   and abs(d - self.follow) < self.tol and abs(goal_err) < 0.1)
        if at_goal:
            self.reached_since = self.reached_since or now
            if self.t_complete is None and now - self.reached_since >= self.hold_for:
                self.t_complete = now
        else:
            self.reached_since = None
        if self.t_complete is not None and info.get('via') == 'lidar' and d is not None:
            self.follow_errs.append(abs(d - self.follow))

    def summary(self, start):
        total = sum(self.mode_ticks.values()) or 1
        out = {
            'task_completed': self.t_complete is not None,
            'time_to_complete_s': (round(self.t_complete - start, 2)
                                   if self.t_complete is not None and start is not None else None),
            'following_distance_error_mean_m': (round(sum(self.follow_errs) / len(self.follow_errs), 3)
                                                if self.follow_errs else None),
            'collisions': self.collisions,
            'min_clearance_m': round(self.min_clearance, 3) if self.min_clearance < math.inf else None,
            'distance_travelled_m': round(self.distance, 2),
            'time_in_mode_pct': {k: round(100.0 * v / total, 1)
                                 for k, v in sorted(self.mode_ticks.items())},
        }
        return out
