from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from f110_gym.envs.driving_task import DrivingTaskConfig
from f110_gym.envs.f1_driving_env import F1DrivingEnv, LIDAR_NUM_BINS


# Obs layout: [108 lidar] + [v, yaw_rate, head_err, lat_err, progress] (+ soc if energy_enabled)
_OBS_DIM_NO_ENERGY = LIDAR_NUM_BINS + 5
_OBS_DIM_WITH_ENERGY = LIDAR_NUM_BINS + 6


def detect_energy_enabled_from_vecnorm(vecnorm_path: str) -> bool | None:
    """Energy mode is encoded by obs dim:
    113 = no energy, 114 = energy_enabled. Returns None if unparseable.
    """
    if not vecnorm_path or not os.path.exists(vecnorm_path):
        return None
    try:
        with open(vecnorm_path, 'rb') as f:
            vn = pickle.load(f)
        rms = getattr(vn, 'obs_rms', None)
        if rms is None:
            return None
        var = np.asarray(rms.var)
        if var.ndim != 1:
            return None
        if var.shape[0] == _OBS_DIM_WITH_ENERGY:
            return True
        if var.shape[0] == _OBS_DIM_NO_ENERGY:
            return False
        return None
    except Exception:
        return None

try:
    from training.gif_utils import write_driving_gif
except ModuleNotFoundError:
    from gif_utils import write_driving_gif

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _distance_along_path(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.diff(x, prepend=x[:1])
    dy = np.diff(y, prepend=y[:1])
    return np.cumsum(np.hypot(dx, dy))


def plot_speed(traces, out_path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    d = traces['distance_m']
    ax.plot(d, traces['v'], lw=1.3, label='actual v', color='#1f77b4')
    ax.plot(d, traces['v_set_cmd'], lw=1.0, alpha=0.6, label='commanded v_set',
            color='#ff7f0e')
    ax.set_xlabel('Distance [m]'); ax.set_ylabel('Speed [m/s]')
    ax.grid(alpha=0.3); ax.legend(loc='upper right')
    ax.set_title('Speed profile')
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_soc_deploy(traces, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    d = traces['distance_m']
    ax1.plot(d, traces['soc'], lw=1.5, color='#1f77b4')
    ax1.set_ylabel('SOC'); ax1.set_ylim(-0.05, 1.05); ax1.grid(alpha=0.3)
    ax1.set_title('State of charge')
    ax2.plot(d, traces['deploy'], lw=1.3, color='#d62728', label='deploy')
    ax2.axhline(0.0, color='k', lw=0.5)
    ax2.set_ylabel('deploy [-1, 1]'); ax2.set_xlabel('Distance [m]')
    ax2.set_ylim(-1.1, 1.1); ax2.grid(alpha=0.3); ax2.legend(loc='lower right')
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_energy_power(traces, out_path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    d = traces['distance_m']
    ax1.plot(d, traces['P_out'], lw=1.2, color='#d62728', label='P_out deploy')
    ax1.plot(d, traces['P_in'], lw=1.2, color='#2ca02c', label='P_in regen')
    ax1.set_ylabel('Power [W]'); ax1.grid(alpha=0.3); ax1.legend(loc='upper right')
    ax1.set_title('Energy power diagnostics')
    ax2.plot(d, traces['dSOC'], lw=1.2, color='#9467bd', label='dSOC')
    ax2.axhline(0.0, color='k', lw=0.5)
    ax2.set_xlabel('Distance [m]'); ax2.set_ylabel('SOC step delta')
    ax2.grid(alpha=0.3); ax2.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_track(traces, out_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(traces['x'], traces['y'], c=traces['v'],
                    cmap='viridis', s=14, edgecolors='none')
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label('speed [m/s]')
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
    ax.grid(alpha=0.3)
    ax.set_title('Trajectory colored by speed')
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_steer(traces, out_path):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    d = traces['distance_m']
    ax.plot(d, traces['steer_cmd'], lw=1.0, alpha=0.6, label='commanded',
            color='#ff7f0e')
    ax.plot(d, traces['steer_actual'], lw=1.3, label='actual', color='#1f77b4')
    ax.axhline(0.0, color='k', lw=0.5)
    ax.set_xlabel('Distance [m]'); ax.set_ylabel('Steering [rad]')
    ax.grid(alpha=0.3); ax.legend(loc='upper right')
    ax.set_title('Steering trace')
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def dump_episode_artifacts(traces: dict, out_dir, ep_idx: int) -> dict:
    """Write traces.npz and per-episode plots. Return summary dict."""
    ep_dir = out_dir / f'ep_{ep_idx}'
    ep_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(traces.get('x', []), dtype=np.float32)
    y = np.asarray(traces.get('y', []), dtype=np.float32)
    if x.size == 0:
        return {'ep_dir': str(ep_dir), 'distance_m': 0.0}
    traces['distance_m'] = _distance_along_path(x, y)
    n = x.size
    # Backfill any optional traces missing on minimal-trace producers.
    for key in ('v', 'v_set_cmd', 'steer_cmd', 'steer_actual', 'soc', 'deploy',
                'deploy_effective', 'P_in', 'P_out', 'dSOC'):
        if key not in traces:
            traces[key] = np.zeros(n, dtype=np.float32)
    np.savez(ep_dir / 'traces.npz', **{k: np.asarray(v) for k, v in traces.items()
                                       if not isinstance(v, dict)})
    try:
        plot_speed(traces, ep_dir / 'plot_speed.png')
        plot_soc_deploy(traces, ep_dir / 'plot_soc_deploy.png')
        plot_energy_power(traces, ep_dir / 'plot_energy_power.png')
        plot_track(traces, ep_dir / 'plot_track.png')
        plot_steer(traces, ep_dir / 'plot_steer.png')
    except Exception as e: 
        print(f'[eval_driving] plot warning ep_{ep_idx}: {e}')
    return {
        'ep_dir': str(ep_dir),
        'distance_m': float(traces['distance_m'][-1]),
        'v_max_mps': float(np.max(traces['v'])),
        'v_mean_mps': float(np.mean(traces['v'])),
        'soc_min': float(np.min(traces['soc'])),
        'soc_final': float(traces['soc'][-1]),
        'steps': int(len(traces['t'])),
    }


class ZeroPolicy:
    def __init__(self, env: F1DrivingEnv):
        self.action_dim = env.action_space.shape[0]

    def reset(self):
        pass

    def act(self, obs, env_state=None) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)


def rollout(env: F1DrivingEnv, policy, max_steps: int, capture_traces: bool = False,
            render: bool = False, seed: int = 0):
    obs, _info = env.reset(seed=seed)
    if hasattr(policy, 'reset'):
        policy.reset()
    if render:
        env.render()
    total_reward = 0.0
    collided = False
    lap_completed = False
    lap_time = None
    traces = None
    if capture_traces:
        traces = {key: [] for key in (
            't', 'x', 'y', 'yaw', 'v', 'soc', 'deploy', 'lap_count',
            'reward_cum', 'v_set_cmd', 'steer_cmd', 'steer_actual',
            'deploy_effective', 'P_in', 'P_out', 'dSOC', 'lidar_bins_m',
        )}
        # Bin-center angles are constant for the whole rollout; attached once.
        traces['lidar_bin_angles'] = np.asarray(env.lidar_bin_angles, dtype=np.float32)

    for _ in range(max_steps):
        inner_car = env.inner.sim.agents[env.ego_idx]
        x = float(inner_car.state[0])
        y = float(inner_car.state[1])
        yaw = float(inner_car.state[4])

        action = policy.act(obs, env_state={'x': x, 'y': y, 'yaw': yaw})
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if render:
            env.render()

        if traces is not None:
            traces['t'].append(float(env._current_time))
            traces['x'].append(x)
            traces['y'].append(y)
            traces['yaw'].append(yaw)
            traces['v'].append(float(obs[LIDAR_NUM_BINS]))
            traces['soc'].append(float(info.get('soc', 1.0)))
            traces['deploy'].append(float(info.get('deploy_command', 0.0)))
            traces['deploy_effective'].append(float(info.get('deploy_effective', 0.0)))
            traces['P_in'].append(float(info.get('P_in', 0.0)))
            traces['P_out'].append(float(info.get('P_out', 0.0)))
            traces['dSOC'].append(float(info.get('dSOC', 0.0)))
            traces['lap_count'].append(int(info.get('lap_count', env._lap_counts_prev)))
            traces['reward_cum'].append(float(total_reward))
            traces['v_set_cmd'].append(float(info.get('v_set_command', 0.0)))
            traces['steer_cmd'].append(float(info.get('steer_command', 0.0)))
            traces['steer_actual'].append(float(info.get('steer_angle_actual', 0.0)))
            bins_m = info.get('lidar_bins_m')
            if bins_m is None:
                bins_m = np.zeros_like(env.lidar_bin_angles)
            traces['lidar_bins_m'].append(np.asarray(bins_m, dtype=np.float32))

        if terminated:
            collided = bool(info.get('collision'))
            lap_completed = not collided
            lap_time = float(info.get('lap_time', env._current_time))
            break
        if truncated:
            break

    metrics = {
        'lap_completed': lap_completed,
        'lap_time_s': lap_time,
        'collision': collided,
        'total_reward': float(total_reward),
    }
    if traces is not None:
        traces = {k: np.asarray(v, dtype=np.float32) for k, v in traces.items()}
    return metrics, traces


class SB3Policy:
    def __init__(self, model, vecnorm=None, deterministic: bool = True):
        self.model = model
        self.vecnorm = vecnorm
        self.deterministic = deterministic

    def reset(self):
        pass

    def act(self, obs, env_state=None) -> np.ndarray:
        o = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        if self.vecnorm is not None:
            o = self.vecnorm.normalize_obs(o)
        action, _ = self.model.predict(o, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32).reshape(-1)


def _make_env_for_shape(map_name: str, speed_cap_mps: float, energy_enabled: bool):
    cfg = DrivingTaskConfig(speed_cap_mps=speed_cap_mps, energy_enabled=energy_enabled)
    return F1DrivingEnv(map_name=map_name, task_config=cfg)


def build_policy(args, env: F1DrivingEnv):
    model = SAC.load(args.checkpoint, device='cpu')
    vecnorm = None
    if args.vecnorm and os.path.exists(args.vecnorm):
        dummy = DummyVecEnv([lambda: _make_env_for_shape(
            args.map, args.speed_cap_mps, args.energy_enabled)])
        vecnorm = VecNormalize.load(args.vecnorm, dummy)
        vecnorm.training = False
    return SB3Policy(model, vecnorm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, required=True)
    ap.add_argument('--vecnorm', type=str, default=None)
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--max-steps', type=int, default=3000)
    ap.add_argument('--out', type=str, required=True)
    ap.add_argument('--map', type=str, default='example_map')
    ap.add_argument('--speed-cap-mps', type=float, default=6.0)
    ap.add_argument('--deploy-speed-boost-mps', type=float, default=3.0)
    ap.add_argument('--downforce-k', type=float, default=0.0)
    ap.add_argument('--stall-decision-steps', type=int, default=0)
    ap.add_argument('--stall-min-progress-m', type=float, default=0.01)
    energy_group = ap.add_mutually_exclusive_group()
    energy_group.add_argument(
        '--energy-enabled', dest='energy_enabled', action='store_true', default=None,
        help='Force energy-managed mode on at eval time.'
    )
    energy_group.add_argument(
        '--no-energy-enabled', dest='energy_enabled', action='store_false',
        help='Force energy-managed mode off at eval time (overrides auto-detect).'
    )
    ap.add_argument('--save-gif', type=str, default=None)
    ap.add_argument('--gif-fps', type=int, default=10)
    ap.add_argument('--gif-stride', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0,
                    help='Reset seed. Training\'s clean_eval uses --seed+1000 '
                         '(default 1000). Match it to reproduce a recorded fastest_lap, '
                         'or vary across episodes to probe policy robustness.')
    ap.add_argument('--render', type=str, default=None,
                    choices=[None, 'human', 'human_fast'],
                    help='Open a live pyglet window during rollout.')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.energy_enabled is None:
        detected = detect_energy_enabled_from_vecnorm(args.vecnorm) if args.vecnorm else None
        if detected is None:
            args.energy_enabled = False
            print('[eval_driving] energy_enabled not specified and could not be inferred; defaulting to False')
        else:
            args.energy_enabled = detected
            print(f'[eval_driving] auto-detected energy_enabled={detected} from {args.vecnorm}')
    else:
        print(f'[eval_driving] energy_enabled={args.energy_enabled} (user-specified)')

    cfg = DrivingTaskConfig(
        speed_cap_mps=args.speed_cap_mps,
        deploy_speed_boost_mps=args.deploy_speed_boost_mps,
        energy_enabled=args.energy_enabled,
    )
    env = F1DrivingEnv(
        map_name=args.map,
        task_config=cfg,
        max_episode_steps=args.max_steps,
        render_mode=args.render,
        downforce_k=args.downforce_k,
        stall_decision_steps=args.stall_decision_steps,
        stall_min_progress_m=args.stall_min_progress_m,
    )
    policy = build_policy(args, env)

    per_ep = []
    gif_traces = None
    for ep in range(args.episodes):
        metrics, traces = rollout(env, policy, max_steps=args.max_steps,
                                  capture_traces=True, render=bool(args.render),
                                  seed=int(args.seed) + ep)
        artifacts = dump_episode_artifacts(traces, out_dir, ep)
        metrics.update(artifacts)
        per_ep.append(metrics)
        print(f"\n=== episode {ep} ===")
        print(f"  lap={'Y' if metrics['lap_completed'] else 'N'} "
              f"collision={'Y' if metrics['collision'] else 'N'} "
              f"lap_time_s={metrics['lap_time_s']} "
              f"distance_m={metrics.get('distance_m', 0):.1f} "
              f"v_max={metrics.get('v_max_mps', 0):.2f} "
              f"v_mean={metrics.get('v_mean_mps', 0):.2f} "
              f"soc_min={metrics.get('soc_min', 1):.3f} "
              f"reward={metrics['total_reward']:.2f}")
        print(f"  wrote: {artifacts['ep_dir']}/")
        if ep == 0:
            gif_traces = traces

    laps = [m['lap_time_s'] for m in per_ep if m['lap_time_s'] is not None]
    report = {
        'policy': args.checkpoint,
        'episodes': args.episodes,
        'lap_completion_rate': float(sum(m['lap_completed'] for m in per_ep) / args.episodes),
        'collision_rate': float(sum(m['collision'] for m in per_ep) / args.episodes),
        'lap_time_mean_s': float(np.mean(laps)) if laps else None,
        'lap_time_std_s': float(np.std(laps)) if laps else None,
        'per_episode': per_ep,
    }
    with open(out_dir / 'eval_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_dir / 'eval_report.json'}")

    if args.save_gif and gif_traces is not None and len(gif_traces['t']) > 0:
        write_driving_gif(
            traces=gif_traces,
            map_name=args.map,
            out=args.save_gif,
            fps=args.gif_fps,
            stride=args.gif_stride,
            title='Driving policy trajectory',
        )
        print(f"Wrote {args.save_gif}")


if __name__ == '__main__':
    main()
