"""
Track-relative features (centerline / raceline lookup) for the energy-management
RL wrapper.
"""

import os
import numpy as np


def _wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def load_waypoints(csv_path: str, delim: str = ';', skiprows: int = 3):
    """Load a waypoint CSV. Returns a dict of arrays keyed by column."""
    csv_path = os.path.abspath(csv_path)
    arr = np.loadtxt(csv_path, delimiter=delim, skiprows=skiprows)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(
            f"Expected at least 5 columns (s, x, y, psi, kappa) in {csv_path}, "
            f"got shape {arr.shape}"
        )
    out = {
        's': arr[:, 0].astype(np.float64),
        'x': arr[:, 1].astype(np.float64),
        'y': arr[:, 2].astype(np.float64),
        'psi': arr[:, 3].astype(np.float64),
        'kappa': arr[:, 4].astype(np.float64),
    }
    if arr.shape[1] >= 6:
        out['v_ref'] = arr[:, 5].astype(np.float64)
    if arr.shape[1] >= 7:
        out['a_ref'] = arr[:, 6].astype(np.float64)
    return out


class TrackFeatures:
    """
    Holds a closed centerline and cheaply answers:
      - nearest point / progress given (x, y)
      - signed lateral deviation from the centerline
      - heading error vs centerline tangent
      - curvature at arbitrary arc-length offsets ahead
    """

    def __init__(self, csv_path: str, delim: str = ';', skiprows: int = 3):
        wp = load_waypoints(csv_path, delim=delim, skiprows=skiprows)
        self.s = wp['s']
        self.x = wp['x']
        self.y = wp['y']
        self.psi = wp['psi']
        self.kappa = wp['kappa']
        self.v_ref = wp.get('v_ref')
        self.xy = np.stack([self.x, self.y], axis=1)
        self.s_total = float(self.s[-1] + max(1e-6, self.s[1] - self.s[0]))

    def _nearest_segment(self, x: float, y: float):
        """Return (nearest_i, t_on_segment, nearest_xy, signed_lat_dev, s_at_nearest)."""
        pt = np.array([x, y], dtype=np.float64)
        diffs = self.xy[1:] - self.xy[:-1]
        l2 = np.sum(diffs * diffs, axis=1)
        # guard against zero-length segments
        l2 = np.maximum(l2, 1e-12)
        rel = pt - self.xy[:-1]
        t = np.einsum('ij,ij->i', rel, diffs) / l2
        t = np.clip(t, 0.0, 1.0)
        proj = self.xy[:-1] + t[:, None] * diffs
        d2 = np.sum((proj - pt) ** 2, axis=1)
        i = int(np.argmin(d2))
        near_xy = proj[i]

        # interpolate psi, kappa, s at nearest point
        psi_near = self.psi[i] + t[i] * _wrap_to_pi(self.psi[i + 1] - self.psi[i])
        s_near = self.s[i] + t[i] * (self.s[i + 1] - self.s[i])

        # signed lateral deviation: positive if car is left of tangent
        tangent = np.array([np.cos(psi_near), np.sin(psi_near)])
        err = pt - near_xy
        signed = tangent[0] * err[1] - tangent[1] * err[0]

        return i, t[i], near_xy, signed, s_near, psi_near

    def _kappa_at_s(self, s_target: float) -> float:
        s_wrapped = s_target % self.s_total
        i = int(np.searchsorted(self.s, s_wrapped, side='right') - 1)
        i = max(0, min(len(self.s) - 2, i))
        seg = max(1e-9, self.s[i + 1] - self.s[i])
        t = (s_wrapped - self.s[i]) / seg
        return float(self.kappa[i] + t * (self.kappa[i + 1] - self.kappa[i]))

    def features(self, x: float, y: float, yaw: float, lookaheads=(2.0, 5.0, 10.0, 20.0, 40.0)):
        """Return a dict of all track-relative features for the RL observation."""
        _, _, _, lat_dev, s_near, psi_near = self._nearest_segment(x, y)
        heading_err = _wrap_to_pi(yaw - psi_near)
        progress = (s_near % self.s_total) / self.s_total
        curv = tuple(self._kappa_at_s(s_near + d) for d in lookaheads)
        return {
            'lateral_dev': float(lat_dev),
            'heading_err': float(heading_err),
            'progress': float(progress),
            's': float(s_near),
            'curvature_lookaheads': np.array(curv, dtype=np.float64),
        }

    def progress_delta(self, s_prev: float, s_now: float) -> float:
        """Arc-length progress between two nearest-point s values, wrap-safe."""
        d = (s_now - s_prev) % self.s_total
        # if the car went backward a tiny bit, avoid a huge positive wrap
        if d > 0.5 * self.s_total:
            d -= self.s_total
        return float(d)

    def pose_at_s(self, s_target: float):
        """Return (x, y, yaw) on centerline at arc-length s, wrapping mod s_total."""
        s_wrapped = float(s_target) % self.s_total
        i = int(np.searchsorted(self.s, s_wrapped, side='right') - 1)
        i = max(0, min(len(self.s) - 2, i))
        seg = max(1e-9, self.s[i + 1] - self.s[i])
        t = (s_wrapped - self.s[i]) / seg
        x = float(self.x[i] + t * (self.x[i + 1] - self.x[i]))
        y = float(self.y[i] + t * (self.y[i + 1] - self.y[i]))
        dpsi = _wrap_to_pi(float(self.psi[i + 1] - self.psi[i]))
        yaw = float(self.psi[i] + t * dpsi)
        return x, y, yaw
