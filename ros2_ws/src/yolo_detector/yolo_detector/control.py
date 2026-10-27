"""
Heading controller + metrics for the target follower -- no ROS imports,
so the same code runs in the ROS node and in the offline simulation test.

Control law (proportional, with ego-motion prediction):
    err   = target_cx / frame_width - 0.5      # -0.5 (far left) .. +0.5 (far right)
    pred  = err + (rotation since that frame) / FOV
    w_cmd = -kp * pred                         # turn toward the target (CCW is +)

The prediction matters when the detector is slow: at 2 frames/s the robot keeps
turning for ~0.5 s on a stale measurement and would overshoot. Shifting the
error by how far we've already turned since the frame removes that lag.

States, to handle detection loss. They're driven by what the detector reports,
not by how long it's been, so a slow detector isn't mistaken for a lost target:
    WAITING    no detector message yet (e.g. YOLO still loading): stay still
    TRACKING   the latest frame contained the target: apply P-control
    COASTING   a frame arrived WITHOUT the target (or no frame for stale_timeout):
               decay the last command smoothly instead of jerking to a stop
    SEARCHING  target not seen for lost_timeout: rotate slowly toward the side
               it was last seen on, until it's re-acquired
"""

import math
from dataclasses import dataclass

WAITING = 'WAITING'
TRACKING = 'TRACKING'
COASTING = 'COASTING'
SEARCHING = 'SEARCHING'


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass
class ControllerConfig:
    target_label: str = 'person'
    kp: float = 2.0             # rad/s per unit of normalized error
    max_angular: float = 1.2    # rad/s, command saturation
    deadband: float = 0.03      # |err| below this -> don't turn (stops dithering)
    stale_timeout: float = 1.0  # s with no detector message at all before we stop trusting it
    lost_timeout: float = 1.5   # s since the target was last seen before we start searching
    search_speed: float = 0.6   # rad/s while searching
    coast_decay: float = 3.0    # 1/s, exponential decay of the command while coasting
    reassoc_gate: float = 0.2   # max normalized distance from the predicted position
                                # to treat a box as the same target
    fov_deg: float = 60.0       # camera horizontal FOV, for ego-motion compensation


class HeadingController:
    def __init__(self, cfg: ControllerConfig):
        self.cfg = cfg
        self.err = 0.0
        self.last_seen = None      # time of last detection of the target
        self.last_cx = None        # normalized x of the target we're locked onto
        self.last_track_cmd = 0.0  # command at the end of TRACKING, used for coasting
        self.search_dir = -1.0     # -1 = turn right (clockwise), +1 = turn left
        self.state = SEARCHING
        self.rot_since_seen = 0.0  # rad the robot has turned since the target was last seen
        self.prev_cmd = 0.0
        self.last_tick = None
        self.missed = 0            # consecutive detector frames without the target
        self.detector_online = False
        self.last_w_norm = None    # tracked box width / frame width (rough range cue)
        self.coast_since = None    # when COASTING started

    def predicted_cx(self):
        """Where the locked target should appear now, given our own rotation.

        Ego-motion compensation: turning left by d radians shifts the whole
        scene right in the image by d / FOV of the frame width. Without this,
        a fast turn moves everyone ~0.15 frame-widths between detections and
        association can jump to the neighbouring person.
        """
        return self.last_cx + self.rot_since_seen / math.radians(self.cfg.fov_deg)

    def update_detections(self, payload, now):
        """Feed one detector message. Returns True if the target was in it."""
        self.detector_online = True
        frame_w = float(payload.get('frame_w') or 1.0)
        cands = [d for d in payload.get('detections', [])
                 if d.get('label') == self.cfg.target_label and 'cx' in d]
        if not cands:
            self.missed += 1
            return False

        best = None
        recently_locked = (self.last_cx is not None and self.last_seen is not None
                           and now - self.last_seen < self.cfg.lost_timeout)
        if recently_locked:
            # Data association: stay on the same person if several are visible,
            # matching against where we PREDICT they are after our own turning.
            pred = self.predicted_cx()
            nearest = min(cands, key=lambda d: abs(d['cx'] / frame_w - pred))
            if abs(nearest['cx'] / frame_w - pred) <= self.cfg.reassoc_gate:
                best = nearest
        if best is None:
            best = max(cands, key=lambda d: d.get('confidence', 0.0))

        cx_norm = best['cx'] / frame_w
        self.last_w_norm = best['w'] / frame_w if best.get('w') else None
        self.err = cx_norm - 0.5
        self.last_cx = cx_norm
        self.last_seen = now
        self.rot_since_seen = 0.0
        self.missed = 0
        if abs(self.err) > self.cfg.deadband:
            # If we lose it later, search toward the side it was drifting to.
            self.search_dir = -1.0 if self.err > 0 else 1.0
        return True

    def set_applied_command(self, angular):
        """Tell the controller what turn rate was actually sent to the robot.

        When a motion planner overrides the turn (e.g. to steer around an
        obstacle), the ego-motion prediction must integrate the real rotation,
        not the one this controller asked for.
        """
        self.prev_cmd = angular

    def compute(self, now):
        """Return (angular_cmd, state) for this control tick."""
        c = self.cfg
        if self.last_tick is not None:
            # Integrate the command we've been applying since the last tick.
            self.rot_since_seen += self.prev_cmd * (now - self.last_tick)
        self.last_tick = now
        age = None if self.last_seen is None else now - self.last_seen

        if not self.detector_online:
            self.state = WAITING
            cmd = 0.0
        elif age is None or age > c.lost_timeout:
            self.state = SEARCHING
            cmd = self.search_dir * c.search_speed
        elif self.missed == 0 and age <= c.stale_timeout:
            self.state = TRACKING
            pred_err = self.predicted_cx() - 0.5
            cmd = 0.0 if abs(pred_err) < c.deadband else -c.kp * pred_err
            cmd = clamp(cmd, -c.max_angular, c.max_angular)
            self.last_track_cmd = cmd
        else:
            if self.state != COASTING:
                self.coast_since = now
            self.state = COASTING
            cmd = self.last_track_cmd * math.exp(-c.coast_decay * (now - self.coast_since))
        self.prev_cmd = cmd
        return cmd, self.state


class Metrics:
    """Collects the numbers for the performance evaluation."""

    def __init__(self, fov_deg=None, settle_band=0.05):
        # The clock starts at the first detector message, so YOLO's model
        # loading time isn't counted as "time to acquire".
        self.start = None
        self.fov_deg = fov_deg
        self.settle_band = settle_band
        self.msgs = 0
        self.msgs_with_target = 0
        self.latencies_ms = []
        self.t_acquire = None
        self.t_centered = None
        self.steady_errs = []
        self.loss_events = 0     # TRACKING/COASTING -> SEARCHING
        self.recoveries = 0      # COASTING -> TRACKING (brief loss handled)
        self.prev_state = None
        self.end = None

    def on_detection(self, now, stamp, found, err):
        if self.start is None:
            self.start = now
        self.msgs += 1
        if stamp is not None:
            self.latencies_ms.append((now - stamp) * 1000.0)
        if not found:
            return
        self.msgs_with_target += 1
        if self.t_acquire is None:
            self.t_acquire = now
        if self.t_centered is None and abs(err) < self.settle_band:
            self.t_centered = now
        if self.t_centered is not None:
            self.steady_errs.append(abs(err))

    def on_tick(self, now, state):
        if self.start is None:      # detector not up yet: nothing to evaluate
            self.prev_state = state
            return
        if self.prev_state in (TRACKING, COASTING) and state == SEARCHING:
            self.loss_events += 1
        if self.prev_state == COASTING and state == TRACKING:
            self.recoveries += 1
        self.prev_state = state
        self.end = now

    def summary(self):
        def pct(xs, p):
            if not xs:
                return None
            s = sorted(xs)
            return s[min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1))))]

        if self.start is None:
            return {'detector_messages': 0, 'note': 'no detector messages received'}
        # Drop the first sample: it includes YOLO's one-time model warm-up.
        lat = self.latencies_ms[1:]
        duration = (self.end or self.start) - self.start
        mean_err = sum(self.steady_errs) / len(self.steady_errs) if self.steady_errs else None
        out = {
            'run_duration_s': round(duration, 2),
            'detector_messages': self.msgs,
            'detector_rate_hz': round(self.msgs / duration, 2) if duration > 0 else None,
            'target_detection_rate_pct': (round(100.0 * self.msgs_with_target / self.msgs, 1)
                                          if self.msgs else None),
            'perception_latency_ms_median': round(pct(lat, 50), 1) if lat else None,
            'perception_latency_ms_p95': round(pct(lat, 95), 1) if lat else None,
            'time_to_acquire_s': (round(self.t_acquire - self.start, 2)
                                  if self.t_acquire is not None else None),
            'time_to_center_s': (round(self.t_centered - self.start, 2)
                                 if self.t_centered is not None else None),
            'steady_state_error_mean_norm': round(mean_err, 4) if mean_err is not None else None,
            'loss_events': self.loss_events,
            'brief_losses_recovered': self.recoveries,
        }
        if mean_err is not None and self.fov_deg:
            out['steady_state_error_mean_deg'] = round(mean_err * self.fov_deg, 2)
        return out
