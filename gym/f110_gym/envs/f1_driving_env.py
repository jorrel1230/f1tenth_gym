import copy
import math
import os
from typing import Optional

import gym as _oldgym
import gymnasium
import numpy as np
from gymnasium import spaces

from f110_gym.envs.base_classes import Integrator
from f110_gym.envs.driving_task import ActionAdapter, DrivingTaskConfig, RewardModel
from f110_gym.envs.energy_model import EnergyManagedCar, EnergyParams
from f110_gym.envs.track_features import TrackFeatures


_EXAMPLES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples')
)
_RACETRACKS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'racetracks')
)

MAP_CONFIGS = {
    'example_map': {
        'map_path': os.path.join(_EXAMPLES_DIR, 'example_map'),
        'map_ext': '.png',
        'start_pose': (0.7, 0.0, 1.37079632679),
        'waypoint_csv': os.path.join(_EXAMPLES_DIR, 'example_waypoints.csv'),
        'wpt_delim': ';',
        'wpt_skiprows': 3,
    },
    # 200m x 200m bordered arena for free-driving F1-scale physics.
    'empty_arena': {
        'map_path': os.path.join(_EXAMPLES_DIR, 'empty_arena'),
        'map_ext': '.png',
        'start_pose': (0.0, 0.0, 0.0),
        'waypoint_csv': None,
    },
    # 10x-scaled example_map: same track shape, F1-fit dimensions
    # (resolution 0.0625 -> 0.625 m/px, origin scaled 10x). No waypoint CSV
    # ships for this map; path features will be zero unless one is provided.
    'example_map_f1': {
        'map_path': os.path.join(_EXAMPLES_DIR, 'example_map_f1'),
        'map_ext': '.png',
        'start_pose': (7.0, 0.0, 1.37079632679),
        'waypoint_csv': None,
    },
    # F1 1:10-scaled racetracks (TUM/Heilmeier dataset). Start poses use the
    # centerline first row (always (0,0)) with heading from first→second
    # centerline tangent.
    'Monza': {
        'map_path': os.path.join(_RACETRACKS_DIR, 'Monza', 'Monza_map'),
        'map_ext': '.png',
        'start_pose': (0.0, 0.0, 1.4729),
        'waypoint_csv': os.path.join(_RACETRACKS_DIR, 'Monza', 'Monza_raceline.csv'),
        'wpt_delim': ';',
        'wpt_skiprows': 3,
    },
    'Spa': {
        'map_path': os.path.join(_RACETRACKS_DIR, 'Spa', 'Spa_map'),
        'map_ext': '.png',
        'start_pose': (0.0, 0.0, 2.1327),
        'waypoint_csv': os.path.join(_RACETRACKS_DIR, 'Spa', 'Spa_raceline.csv'),
        'wpt_delim': ';',
        'wpt_skiprows': 3,
    },
    'Sakhir': {
        'map_path': os.path.join(_RACETRACKS_DIR, 'Sakhir', 'Sakhir_map'),
        'map_ext': '.png',
        'start_pose': (0.0, 0.0, 1.5252),
        'waypoint_csv': os.path.join(_RACETRACKS_DIR, 'Sakhir', 'Sakhir_raceline.csv'),
        'wpt_delim': ';',
        'wpt_skiprows': 3,
    },
}


LIDAR_NUM_BINS = 108
LIDAR_MAX_RANGE_M = 30.0


class F1DrivingEnv(gymnasium.Env):
    metadata = {'render_modes': ['human', 'human_fast']}

    @staticmethod
    def _apply_seed_to_inner(inner, seed: int) -> None:
        seed = int(seed)
        inner.seed = seed
        inner.sim.seed = seed
        for agent in inner.sim.agents:
            agent.seed = seed

    def __init__(
        self,
        map_name='example_map',
        task_config=None,
        decision_interval_steps=1,
        max_episode_steps=3000,
        integrator=Integrator.RK4,
        seed=12345,
        render_mode=None,
        max_episode_seconds: Optional[float] = 120.0,
        waypoint_csv: Optional[str] = None,
        downforce_k: float = 0.0,
        stall_decision_steps: int = 0,
        stall_min_progress_m: float = 0.01,
    ):
        super().__init__()
        self.cfg = task_config or DrivingTaskConfig()
        self.decision_interval_steps = int(decision_interval_steps)
        if self.decision_interval_steps != 1:
            raise ValueError('F1DrivingEnv is intentionally fixed at decision_interval_steps=1')

        cfg = MAP_CONFIGS[map_name]
        self._start_pose = np.array(cfg['start_pose'], dtype=np.float64)

        self.inner = _oldgym.make(
            'f110_gym:f110-v0',
            map=cfg['map_path'],
            map_ext=cfg['map_ext'],
            num_agents=1,
            timestep=0.01,
            integrator=integrator,
            seed=seed,
        ).unwrapped
        self._seed = int(seed)
        self.ego_idx = 0
        self.dt = self.inner.timestep
        self.render_mode = render_mode

        self.base_params = copy.deepcopy(self.inner.params)
        self._p_a_max_base = self.base_params['a_max']
        self.speed_cap_mps = float(self.cfg.speed_cap_mps)

        # Fake aero grip: mu_eff = mu_base * (1 + k * v^2).
        self.downforce_k = float(downforce_k)
        self._mu_base = float(self.base_params.get('mu', 1.0))

        self.action_adapter = ActionAdapter(self.base_params, self.cfg)
        self.reward_model = RewardModel(self.cfg)

        ep_kwargs = dict(
            m=self.base_params['m'],
            a_max_base=self._p_a_max_base,
            v_switch=self.base_params['v_switch'],
        )
        if not self.cfg.energy_enabled:
            ep_kwargs.update(P_mgu_k=0.0, E_batt=1e9)
        self.energy = EnergyManagedCar(EnergyParams(**ep_kwargs))

        # Full-FOV LiDAR min-pooling: 1080 raw beams -> LIDAR_NUM_BINS bins
        full_num_beams = int(self.inner.sim.agents[self.ego_idx].num_beams)
        full_fov = float(self.inner.sim.agents[self.ego_idx].fov)
        angle_incr = full_fov / max(full_num_beams - 1, 1)
        full_angles = -full_fov / 2.0 + np.arange(full_num_beams) * angle_incr
        self._beam_angles = full_angles
        beams_per_bin = max(1, full_num_beams // LIDAR_NUM_BINS)
        self._bin_assignment = np.minimum(
            np.arange(full_num_beams) // beams_per_bin,
            LIDAR_NUM_BINS - 1,
        )
        # Bin-center angles, evenly spaced across full FOV. Used by the GIF
        # renderer to draw radial rays at the right bearings.
        bin_edges = np.linspace(-full_fov / 2.0, full_fov / 2.0, LIDAR_NUM_BINS + 1)
        self.lidar_bin_angles = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # ----- track features -----
        wpt_csv = waypoint_csv if waypoint_csv is not None else cfg.get('waypoint_csv')
        if wpt_csv is not None:
            self.track = TrackFeatures(
                wpt_csv,
                delim=cfg.get('wpt_delim', ';'),
                skiprows=cfg.get('wpt_skiprows', 3),
            )
            self._lap_length = float(self.track.s_total)
        else:
            self.track = None
            self._lap_length = float('inf')

        # obs: [108 lidar] + [v_x, yaw_rate, heading_err, lateral_err, progress]
        # +1 SOC slot if energy_enabled.
        self._scalar_dim = 5 + (1 if self.cfg.energy_enabled else 0)
        self.OBS_DIM = LIDAR_NUM_BINS + self._scalar_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBS_DIM,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.max_episode_steps = int(max_episode_steps)
        self.max_episode_seconds = (
            None if max_episode_seconds is None else float(max_episode_seconds)
        )
        self.stall_decision_steps = int(stall_decision_steps)
        self.stall_min_progress_m = float(stall_min_progress_m)

        self._step_count = 0
        self._current_time = 0.0
        self._last_v = 0.0
        self._lap_counts_prev = 0
        self._s_prev = 0.0
        self._accum_progress = 0.0
        self._last_raw_scan = None
        self._stall_count = 0
        # Per-episode reward decomp + action/velocity stats. Reset on reset(),
        # accumulated each step, surfaced on terminal info for TB tracing.
        self._ep_reward_breakdown = {
            'progress': 0.0, 'step_pen': 0.0, 'lateral_pen': 0.0,
            'forward_bonus': 0.0, 'collision_pen': 0.0,
            'completion_bonus': 0.0, 'stall_pen': 0.0, 'regen_pen': 0.0,
        }
        self._ep_action_stats = {
            'steer_sum': 0.0, 'steer_abs_sum': 0.0,
            'v_set_sum': 0.0, 'v_set_max': 0.0,
            'deploy_sum': 0.0, 'deploy_pos_sum': 0.0, 'deploy_neg_sum': 0.0,
            'v_x_sum': 0.0, 'v_x_max': 0.0,
            # A. Action saturation counters (raw action ∈ [-1, 1]).
            'steer_sat_count': 0, 'v_set_sat_count': 0,
            'deploy_sat_pos_count': 0, 'deploy_sat_neg_count': 0,
            # B. Wall proximity (min over LiDAR bins, in meters).
            'lidar_min_sum': 0.0, 'lidar_min_min': 1e9,
            # C. Lateral deviation magnitude.
            'lat_dev_abs_sum': 0.0, 'lat_dev_abs_max': 0.0,
            # D. Stall counter peak.
            'stall_count_max': 0,
            # F. Crash location (set on terminal step if collision).
            'crash_progress': -1.0,
            'n': 0,
        }

    # ------------- helpers -------------

    def _a_drag(self, v: float) -> float:
        Cd_A = self.base_params['Cd_A']
        rho_air = self.base_params['rho_air']
        m = self.base_params['m']
        return -0.5 * rho_air * Cd_A * v * abs(v) / m

    def _apply_energy_to_params(self, deploy: float) -> None:
        new_a_max = self.energy.a_max_effective(deploy)
        self.base_params['a_max'] = float(new_a_max)
        self.inner.update_params(self.base_params, index=self.ego_idx)

    def _restore_base_a_max(self) -> None:
        self.base_params['a_max'] = self._p_a_max_base
        self.inner.update_params(self.base_params, index=self.ego_idx)

    def _lidar_bins(self, scan: np.ndarray) -> np.ndarray:
        """Min-pool full-FOV raw scan into LIDAR_NUM_BINS bins. Clip 30m, normalize /30."""
        beams = np.asarray(scan, dtype=np.float64)
        beams = np.where(np.isnan(beams), LIDAR_MAX_RANGE_M, beams)
        beams = np.where(np.isposinf(beams), LIDAR_MAX_RANGE_M, beams)
        beams = np.where(np.isneginf(beams), 0.0, beams)
        beams = np.clip(beams, 0.0, LIDAR_MAX_RANGE_M)
        bins = np.full(LIDAR_NUM_BINS, LIDAR_MAX_RANGE_M, dtype=np.float64)
        for bin_idx, distance in zip(self._bin_assignment, beams):
            if distance < bins[bin_idx]:
                bins[bin_idx] = distance
        return (bins / LIDAR_MAX_RANGE_M).astype(np.float32)

    def _path_features(self, x: float, y: float, yaw: float):
        """Return (s, heading_err, lateral_dev, progress). Zeros if no track."""
        if self.track is None:
            return 0.0, 0.0, 0.0, 0.0
        feats = self.track.features(x, y, yaw, lookaheads=())
        return (
            float(feats['s']),
            float(feats['heading_err']),
            float(feats['lateral_dev']),
            float(feats['progress']),
        )

    def _build_obs(self, inner_obs, path_feats) -> np.ndarray:
        scan = inner_obs['scans'][self.ego_idx]
        # Cache raw scan for LiDAR-based features.
        self._last_raw_scan = np.asarray(scan, dtype=np.float64).copy()
        lidar = self._lidar_bins(scan)

        v = float(inner_obs['linear_vels_x'][self.ego_idx])
        yaw_rate = float(inner_obs['ang_vels_z'][self.ego_idx])
        _s, heading_err, lat_dev, progress = path_feats

        scalars = [v, yaw_rate, heading_err, lat_dev, progress]
        if self.cfg.energy_enabled:
            scalars.append(float(self.energy.soc))

        obs = np.empty(self.OBS_DIM, dtype=np.float32)
        obs[:LIDAR_NUM_BINS] = lidar
        obs[LIDAR_NUM_BINS:] = np.asarray(scalars, dtype=np.float32)
        return obs

    # ------------- gym API -------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            super().reset(seed=seed)
            self._seed = int(seed)
            self._apply_seed_to_inner(self.inner, self._seed)

        pose = self._start_pose
        if options is not None and 'pose' in options:
            pose = np.asarray(options['pose'], dtype=np.float64)
        poses = pose.reshape(1, 3)

        inner_obs, _r, _d, _i = self.inner.reset(poses)
        self.energy.reset()
        self._restore_base_a_max()

        self._step_count = 0
        self._current_time = 0.0
        self._last_v = float(inner_obs['linear_vels_x'][self.ego_idx])
        self._lap_counts_prev = int(inner_obs['lap_counts'][self.ego_idx])

        x = float(inner_obs['poses_x'][self.ego_idx])
        y = float(inner_obs['poses_y'][self.ego_idx])
        yaw = float(inner_obs['poses_theta'][self.ego_idx])
        path_feats = self._path_features(x, y, yaw)
        self._s_prev = path_feats[0]
        self._accum_progress = 0.0
        self._stall_count = 0
        for k in self._ep_reward_breakdown:
            self._ep_reward_breakdown[k] = 0.0
        sa = self._ep_action_stats
        for k in ('steer_sum', 'steer_abs_sum', 'v_set_sum', 'v_set_max',
                  'deploy_sum', 'deploy_pos_sum', 'deploy_neg_sum',
                  'v_x_sum', 'v_x_max',
                  'lidar_min_sum', 'lat_dev_abs_sum', 'lat_dev_abs_max'):
            sa[k] = 0.0
        for k in ('steer_sat_count', 'v_set_sat_count',
                  'deploy_sat_pos_count', 'deploy_sat_neg_count',
                  'stall_count_max', 'n'):
            sa[k] = 0
        sa['lidar_min_min'] = 1e9
        sa['crash_progress'] = -1.0

        obs = self._build_obs(inner_obs, path_feats)
        info = {
            'lap_count': int(self._lap_counts_prev),
            'soc': float(self.energy.soc),
            'progress': float(path_feats[3]),
            'accum_progress_m': 0.0,
        }
        return obs, info

    def step(self, action):
        # Raw action retained for saturation telemetry. Decoded action is what
        # downstream physics sees (already clipped to [-1, 1] inside decode).
        raw_action = np.asarray(action, dtype=np.float64).flatten()
        if raw_action.size < 3:
            raw_action = np.pad(raw_action, (0, 3 - raw_action.size))
        steer, v_set, deploy = self.action_adapter.decode(action)
        inner_car = self.inner.sim.agents[self.ego_idx]

        # Fake-downforce mu update before re-pushing params. _apply_energy_to_params
        # below calls inner.update_params with the same blob so this rides along.
        if self.downforce_k > 0.0:
            v_now = float(inner_car.state[3])
            self.base_params['mu'] = self._mu_base * (1.0 + self.downforce_k * v_now * v_now)
        self._apply_energy_to_params(deploy)
        # ERS deploy lifts v_set ceiling above speed_cap so commanded deploy
        # has a kinetic effect (otherwise PID never demands the boosted
        # a_max at cap). Hard-clamped to physical v_max so it can't escape
        # tire / drag limits.
        d_eff = self.energy.effective_deploy(deploy)
        if d_eff > 0.0 and self.cfg.deploy_speed_boost_mps > 0.0:
            v_set = min(
                v_set + d_eff * self.cfg.deploy_speed_boost_mps,
                float(self.base_params['v_max']),
            )
        # F1 powertrain interlock: cut engine torque demand while regen
        # active by clamping v_set to current velocity.
        if deploy < 0.0:
            current_v = float(inner_car.state[3])
            # Skip clamp below v_floor: regen harvet zero there
            # so the engine-cut interlock would
            # deadlock the car at v=0 if the policy holds regen.
            if current_v > self.energy.params.v_floor:
                v_set = min(v_set, current_v)
        inner_obs, _r, _done, _info = self.inner.step(np.array([[steer, v_set]], dtype=np.float64))
        self._step_count += 1
        self._current_time += self.dt
        v_new = float(inner_obs['linear_vels_x'][self.ego_idx])
        a_actual = (v_new - self._last_v) / self.dt
        a_drag = self._a_drag(v_new)
        self.energy.step(deploy, a_actual, v_new, self.dt, a_drag=a_drag)

        self._last_v = v_new

        is_collision = bool(inner_obs['collisions'][self.ego_idx])
        lap_counts = int(inner_obs['lap_counts'][self.ego_idx])
        self._lap_counts_prev = lap_counts

        # Path features and Δs against centerline.
        x = float(inner_obs['poses_x'][self.ego_idx])
        y = float(inner_obs['poses_y'][self.ego_idx])
        yaw = float(inner_obs['poses_theta'][self.ego_idx])
        path_feats = self._path_features(x, y, yaw)
        s_now, heading_err, lat_dev, progress = path_feats

        if self.track is not None:
            delta_s = self.track.progress_delta(self._s_prev, s_now)
            self._s_prev = s_now
            # Forward progress only ratchets the lap counter.
            if delta_s > 0.0:
                self._accum_progress += delta_s
        else:
            delta_s = 0.0

        lap_completed = (
            self.track is not None and self._accum_progress >= self._lap_length
        )
        terminated = is_collision or lap_completed

        truncated = False
        if (
            self.max_episode_seconds is not None
            and self._current_time >= self.max_episode_seconds
        ):
            truncated = True

        # Stall-truncation: increment when forward arc-length ratchet does not
        # advance by at least `stall_min_progress_m`; reset on real progress.
        # Truncates (not terminates) so SAC bootstraps from V(s_T) instead of
        # treating it as a hard terminal. Note 0 disables.
        stalled_truncate = False
        if self.stall_decision_steps > 0 and not (terminated or truncated):
            if delta_s > self.stall_min_progress_m:
                self._stall_count = 0
            else:
                self._stall_count += 1
                if self._stall_count >= self.stall_decision_steps:
                    truncated = True
                    stalled_truncate = True

        reward = self.reward_model.step_reward(
            delta_s=delta_s,
            v_forward=v_new,
            lateral_dev=lat_dev,
            collided=is_collision,
            completed=lap_completed,
            dt=self.dt,
            stalled=stalled_truncate,
            deploy=deploy,
        )

        # Reward decomposition (mirrors RewardModel exactly so the sum matches).
        cfg = self.cfg
        ds_clipped = max(min(float(delta_s), cfg.delta_s_clip_pos), cfg.delta_s_clip_neg)
        rb = self._ep_reward_breakdown
        rb['progress'] += cfg.progress_reward_weight * ds_clipped
        rb['forward_bonus'] += cfg.forward_reward_scale * max(0.0, float(v_new)) * float(self.dt)
        rb['lateral_pen'] += -cfg.lateral_penalty_weight * abs(float(lat_dev))
        rb['step_pen'] += -cfg.step_penalty
        if is_collision:
            rb['collision_pen'] += -float(cfg.collision_penalty)
        if lap_completed:
            rb['completion_bonus'] += float(cfg.completion_bonus)
        if stalled_truncate:
            rb['stall_pen'] += -float(cfg.stall_truncation_penalty)
        if cfg.regen_penalty_weight > 0.0 and deploy < 0.0:
            rb['regen_pen'] += -float(cfg.regen_penalty_weight) * abs(float(deploy))

        # Action / velocity stats per-episode.
        sa = self._ep_action_stats
        sa['steer_sum'] += float(steer)
        sa['steer_abs_sum'] += abs(float(steer))
        sa['v_set_sum'] += float(v_set)
        if v_set > sa['v_set_max']:
            sa['v_set_max'] = float(v_set)
        sa['deploy_sum'] += float(deploy)
        if deploy > 0.0:
            sa['deploy_pos_sum'] += float(deploy)
        elif deploy < 0.0:
            sa['deploy_neg_sum'] += float(deploy)
        sa['v_x_sum'] += float(v_new)
        if v_new > sa['v_x_max']:
            sa['v_x_max'] = float(v_new)
        sa['n'] = int(sa['n']) + 1

        # A. Action saturation (raw input ∈ [-1, 1]).
        if abs(raw_action[0]) > 0.95:
            sa['steer_sat_count'] = int(sa['steer_sat_count']) + 1
        if raw_action[1] > 0.95:
            sa['v_set_sat_count'] = int(sa['v_set_sat_count']) + 1
        if raw_action[2] > 0.95:
            sa['deploy_sat_pos_count'] = int(sa['deploy_sat_pos_count']) + 1
        elif raw_action[2] < -0.95:
            sa['deploy_sat_neg_count'] = int(sa['deploy_sat_neg_count']) + 1

        # B. Wall proximity from raw scan (cached in _build_obs as meters).
        if self._last_raw_scan is not None and self._last_raw_scan.size:
            min_bin = float(np.min(self._last_raw_scan))
            sa['lidar_min_sum'] += min_bin
            if min_bin < sa['lidar_min_min']:
                sa['lidar_min_min'] = min_bin

        # C. Lateral deviation magnitude.
        lat_abs = abs(float(lat_dev))
        sa['lat_dev_abs_sum'] += lat_abs
        if lat_abs > sa['lat_dev_abs_max']:
            sa['lat_dev_abs_max'] = lat_abs

        # D. Stall counter peak.
        if int(self._stall_count) > int(sa['stall_count_max']):
            sa['stall_count_max'] = int(self._stall_count)

        # F. Crash location: capture progress fraction at terminal collision step.
        if is_collision and sa['crash_progress'] < 0.0:
            sa['crash_progress'] = float(progress)

        self._restore_base_a_max()

        obs = self._build_obs(inner_obs, path_feats)
        # Capture the physical (non-normalized) LiDAR bin distances so the GIF
        # can draw true-scale rays. obs[:LIDAR_NUM_BINS] is normalized by
        # LIDAR_MAX_RANGE_M so we invert that here once instead of in the renderer.
        lidar_bins_m = obs[:LIDAR_NUM_BINS].astype(np.float32) * LIDAR_MAX_RANGE_M
        info = {
            'lap_count': int(self._lap_counts_prev),
            'lap_time': float(self._current_time),
            'collision': bool(is_collision),
            'steer_command': float(steer),
            'v_set_command': float(v_set),
            'deploy_command': float(deploy),
            'soc': float(self.energy.soc),
            'steer_angle_actual': float(inner_car.state[2]),
            'lidar_bins_m': lidar_bins_m,
            'progress': float(progress),
            'accum_progress_m': float(self._accum_progress),
            'delta_s': float(delta_s),
            'lap_completed': bool(lap_completed),
            'stall_count': int(self._stall_count),
            'mu_effective': float(self.base_params.get('mu', self._mu_base)),
            'truncated': bool(truncated),
            'truncated_by_stall': bool(stalled_truncate),
        }
        for k in ('deploy_effective', 'P_in', 'P_out', 'dSOC'):
            info[k] = float(self.energy.last_step.get(k, 0.0))

        # On terminal step only, surface per-episode breakdown + stats so the
        # SAC callback can pull them once and log to TB without polluting every
        # step's info dict.
        if terminated or truncated:
            sa = self._ep_action_stats
            n = max(1, int(sa['n']))
            info['reward_breakdown'] = dict(self._ep_reward_breakdown)
            info['ep_stats'] = {
                'steer_mean': float(sa['steer_sum']) / n,
                'steer_abs_mean': float(sa['steer_abs_sum']) / n,
                'v_set_mean': float(sa['v_set_sum']) / n,
                'v_set_max': float(sa['v_set_max']),
                'deploy_mean': float(sa['deploy_sum']) / n,
                'deploy_pos_mean': float(sa['deploy_pos_sum']) / n,
                'deploy_neg_mean': float(sa['deploy_neg_sum']) / n,
                'v_x_mean': float(sa['v_x_sum']) / n,
                'v_x_max': float(sa['v_x_max']),
                'ep_len': int(n),
                # A. Action saturation rates [0, 1].
                'steer_sat_rate': float(sa['steer_sat_count']) / n,
                'v_set_sat_rate': float(sa['v_set_sat_count']) / n,
                'deploy_sat_pos_rate': float(sa['deploy_sat_pos_count']) / n,
                'deploy_sat_neg_rate': float(sa['deploy_sat_neg_count']) / n,
                # B. Wall proximity (meters).
                'lidar_min_mean': float(sa['lidar_min_sum']) / n,
                'lidar_min_min': float(sa['lidar_min_min']) if sa['lidar_min_min'] < 1e8 else -1.0,
                # C. Lateral deviation magnitude.
                'lat_dev_abs_mean': float(sa['lat_dev_abs_sum']) / n,
                'lat_dev_abs_max': float(sa['lat_dev_abs_max']),
                # D. Stall progression peak.
                'stall_count_max': int(sa['stall_count_max']),
                # F. Where the crash happened (-1 if no collision this ep).
                'crash_progress': float(sa['crash_progress']),
            }
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        mode = self.render_mode or 'human_fast'
        self.inner.render(mode=mode)

    def close(self):
        pass
