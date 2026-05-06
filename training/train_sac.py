"""
SAC trainer for F1DrivingEnv

Usage:
    python training/train_sac.py --map example_map --total-steps 1000000 \
        --decision-interval 1 --tag sac_tagging

Artifacts:
    runs/<tag>/best.zip                latest best-by-reward checkpoint
    runs/<tag>/ckpt_*.zip              periodic checkpoints
    runs/<tag>/vecnorm.pkl             VecNormalize stats (required to eval)
    runs/<tag>/tb/                     TensorBoard logs
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import gymnasium
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from f110_gym.envs.driving_task import DrivingTaskConfig
from f110_gym.envs.f1_driving_env import F1DrivingEnv

class StripArrayInfoWrapper(gymnasium.Wrapper):
    """Drop array-valued info keys before SB3's Monitor sees them (Monitor can't serialize them)."""
    _ARRAY_KEYS = ('soc_trace', 'deploy_effective_trace',
                   't_trace', 'x_trace', 'y_trace', 'v_trace', 'a_trace')

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if 'energy_deployed_j' not in info:
            energy_deployed_j = compute_energy_deployed_j(info, getattr(self.env, 'unwrapped', self.env))
            if energy_deployed_j is not None:
                info['energy_deployed_j'] = energy_deployed_j
        for k in self._ARRAY_KEYS:
            info.pop(k, None)
        return obs, reward, terminated, truncated, info


def make_env(map_name: str, decision_interval: int, max_episode_steps: int,
             seed: int, log_dir: str, rank: int = 0,
             speed_cap_mps: float = 8.0,
             deploy_speed_boost_mps: float = 3.0,
             energy_enabled: bool = False,
             downforce_k: float = 0.0,
             stall_decision_steps: int = 0,
             stall_min_progress_m: float = 0.01,
             progress_reward_weight: float = 1.0,
             step_penalty: float = 0.01,
             collision_penalty: float = 5.0,
             completion_bonus: float = 3.0,
             stall_truncation_penalty: float = 0.0,
             regen_penalty_weight: float = 0.0,
             forward_reward_scale: float = 0.0):
    if int(decision_interval) != 1:
        raise ValueError('F1DrivingEnv is fixed at decision_interval=1')

    def _init():
        cfg = DrivingTaskConfig(
            speed_cap_mps=speed_cap_mps,
            deploy_speed_boost_mps=deploy_speed_boost_mps,
            energy_enabled=energy_enabled,
            progress_reward_weight=progress_reward_weight,
            step_penalty=step_penalty,
            collision_penalty=collision_penalty,
            completion_bonus=completion_bonus,
            stall_truncation_penalty=stall_truncation_penalty,
            regen_penalty_weight=regen_penalty_weight,
            forward_reward_scale=forward_reward_scale,
        )
        # max_episode_steps <= 0 disables time-based truncation entirely
        # (episode runs until lap-completion / collision / stall). Stall detector
        # (stall_decision_steps) is the recommended cutoff in that mode.
        if int(max_episode_steps) > 0:
            mes_arg = int(max_episode_steps)
            mes_sec_arg = mes_arg * 0.01
        else:
            mes_arg = 10**9
            mes_sec_arg = None
        env = F1DrivingEnv(
            map_name=map_name,
            task_config=cfg,
            max_episode_steps=mes_arg,
            max_episode_seconds=mes_sec_arg,
            seed=seed + rank,
            downforce_k=downforce_k,
            stall_decision_steps=stall_decision_steps,
            stall_min_progress_m=stall_min_progress_m,
        )
        env = StripArrayInfoWrapper(env)
        env = Monitor(env, filename=os.path.join(log_dir, f'monitor_{rank}.csv'))
        return env
    return _init


# --- custom metrics callback ---

def _safe_mean(values):
    vals = [float(v) for v in values if v is not None]
    return float(np.round(np.mean(vals), 6)) if vals else None


def summarize_clean_eval_metrics(episodes: list[dict]) -> dict:
    n = max(1, len(episodes))
    steer = [np.asarray(ep.get('steer_cmd', []), dtype=np.float32) for ep in episodes]
    speed = [np.asarray(ep.get('v_set_cmd', []), dtype=np.float32) for ep in episodes]
    steer_abs = [float(np.mean(np.abs(x))) for x in steer if x.size]
    steer_delta = [float(np.mean(np.abs(np.diff(x)))) for x in steer if x.size > 1]
    speed_delta = [float(np.mean(np.abs(np.diff(x)))) for x in speed if x.size > 1]
    return {
        'completion_rate': float(sum(bool(ep.get('lap_completed', False)) for ep in episodes) / n),
        'collision_rate': float(sum(bool(ep.get('collision', False)) for ep in episodes) / n),
        'lap_time_mean': _safe_mean(ep.get('lap_time_s') for ep in episodes if ep.get('lap_completed', False)),
        'reward_mean': _safe_mean(ep.get('total_reward') for ep in episodes),
        'progress_mean': _safe_mean(ep.get('progress') for ep in episodes),
        'mean_abs_steer': _safe_mean(steer_abs),
        'mean_abs_steer_delta': _safe_mean(steer_delta),
        'mean_abs_speed_delta': _safe_mean(speed_delta),
        'soc_min_mean': _safe_mean(ep.get('soc_min') for ep in episodes),
        'energy_deployed_mean_j': _safe_mean(ep.get('energy_deployed_j') for ep in episodes),
    }


def compute_energy_deployed_j(info: dict, env=None):
    if info.get('energy_deployed_j') is not None:
        return float(info['energy_deployed_j'])
    deploy_trace = np.asarray(info.get('deploy_effective_trace', []), dtype=np.float32)
    if not deploy_trace.size or env is None:
        return None
    energy = getattr(env, 'energy', None)
    params = getattr(energy, 'params', None)
    power = getattr(params, 'P_mgu_k', None)
    dt = getattr(env, 'dt', None)
    if power is None or dt is None:
        return None
    return float(np.sum(np.clip(deploy_trace, 0.0, 1.0)) * float(power) * float(dt))


class SaveVecNormalizeOnBestCallback(BaseCallback):
    """Dump VecNormalize stats next to best_model.zip whenever EvalCallback
    triggers a new best. Without this, mid-run viz of best_model.zip is
    useless because the policy expects normalized obs but no stats file
    exists until train_env.save() at end-of-training."""

    def __init__(self, run_dir: Path):
        super().__init__()
        self._run_dir = Path(run_dir)

    def _on_step(self) -> bool:
        vn = self.model.get_vec_normalize_env()
        if vn is not None:
            vn.save(str(self._run_dir / 'vecnorm.pkl'))
        return True


class CleanEvalMetricsCallback(BaseCallback):
    """Run deterministic eval episodes and log aggregate clean metrics.
    """

    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int = 1, max_steps: int = 10_000,
                 fastest_lap_dir: Optional[Path] = None):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = max(1, int(eval_freq))
        self.n_eval_episodes = max(1, int(n_eval_episodes))
        self.max_steps = max(1, int(max_steps))
        self._fastest_lap_dir = Path(fastest_lap_dir) if fastest_lap_dir is not None else None
        self._fastest_lap_s: Optional[float] = None

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # NOTE: critiaal cuz sync VecNormalize stats from train_env into eval_env
        # before rolling out. Otherwise eval normalizes raw obs with stale
        # init stats while policy was trained on drifted-stat normalization
        # garbage actions, crashes every eval episode.
        self._sync_vecnormalize_stats()

        episodes = [self._rollout_episode() for _ in range(self.n_eval_episodes)]
        for key, value in summarize_clean_eval_metrics(episodes).items():
            if value is not None:
                self.logger.record(f'clean_eval/{key}', value)

        # Fastest-lap checkpoint: pick the min lap_time_s among completed laps
        # in this eval batch, compare to running min, save model+vecnorm if new
        # record. Logged so TB shows the lap-time floor over training.
        if self._fastest_lap_dir is not None:
            completed = [ep for ep in episodes
                         if ep.get('lap_completed') and ep.get('lap_time_s') is not None]
            if completed:
                this_best = float(min(float(ep['lap_time_s']) for ep in completed))
                if self._fastest_lap_s is None or this_best < self._fastest_lap_s:
                    self._fastest_lap_s = this_best
                    self._save_fastest()
                self.logger.record('clean_eval/fastest_lap_s', float(self._fastest_lap_s))
        return True

    def _sync_vecnormalize_stats(self):
        train_vn = self.model.get_vec_normalize_env()
        if train_vn is None:
            return
        # Walk through eval_env wrapper stack to find the VecNormalize.
        eval_vn = self.eval_env
        while eval_vn is not None and not hasattr(eval_vn, 'obs_rms'):
            eval_vn = getattr(eval_vn, 'venv', None)
        if eval_vn is None:
            return

        if hasattr(train_vn, 'obs_rms') and hasattr(eval_vn, 'obs_rms'):
            eval_vn.obs_rms = train_vn.obs_rms
        if hasattr(train_vn, 'ret_rms') and hasattr(eval_vn, 'ret_rms'):
            eval_vn.ret_rms = train_vn.ret_rms

    def _save_fastest(self):
        d = self._fastest_lap_dir
        d.mkdir(parents=True, exist_ok=True)
        self.model.save(str(d / 'fastest_lap.zip'))
        vn = self.model.get_vec_normalize_env()
        if vn is not None:
            vn.save(str(d / 'fastest_lap_vecnorm.pkl'))
        try:
            (d / 'fastest_lap_time_s.txt').write_text(f'{self._fastest_lap_s:.6f}\n')
        except OSError:
            pass

    def _rollout_episode(self) -> dict:
        obs = self.eval_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        total_reward = 0.0
        steer_cmd = []
        v_set_cmd = []
        last_info = {}
        energy_deployed_j_total = 0.0
        saw_energy_deployed_j = False
        env = self._first_unwrapped_env()

        for _ in range(self.max_steps):
            action, _ = self.model.predict(obs, deterministic=True)
            steer, v_set = self._decode_action(action)
            if steer is not None:
                steer_cmd.append(steer)
            if v_set is not None:
                v_set_cmd.append(v_set)

            step_out = self.eval_env.step(action)
            if len(step_out) == 4:
                obs, rewards, dones, infos = step_out
            else:
                obs, rewards, terminated, truncated, infos = step_out
                dones = np.asarray(terminated) | np.asarray(truncated)
            reward = rewards[0] if np.ndim(rewards) else rewards
            done = bool(dones[0] if np.ndim(dones) else dones)
            info = infos[0] if isinstance(infos, (list, tuple)) else infos
            last_info = info.get('terminal_info', info.get('final_info', info))
            energy_deployed_j = compute_energy_deployed_j(last_info, env)
            if energy_deployed_j is not None:
                energy_deployed_j_total += energy_deployed_j
                saw_energy_deployed_j = True
            total_reward += float(reward)
            if done:
                break

        soc_trace = np.asarray(last_info.get('soc_trace', []), dtype=np.float32)
        return {
            'lap_completed': bool(last_info.get('lap_completed', False)),
            'collision': bool(last_info.get('collision', False)),
            'lap_time_s': last_info.get('lap_time_s', last_info.get('lap_time')),
            'total_reward': total_reward,
            'progress': last_info.get('progress'),
            'steer_cmd': np.asarray(steer_cmd, dtype=np.float32),
            'v_set_cmd': np.asarray(v_set_cmd, dtype=np.float32),
            'soc_min': float(np.min(soc_trace)) if soc_trace.size else last_info.get('soc'),
            'energy_deployed_j': energy_deployed_j_total if saw_energy_deployed_j else None,
        }

    def _decode_action(self, action):
        env = self._first_unwrapped_env()
        raw = np.asarray(action, dtype=np.float32)
        if raw.ndim > 1:
            raw = raw[0]
        if env is not None and hasattr(env, '_decode_action'):
            decoded = env._decode_action(raw)
            return float(decoded[0]), float(decoded[1])
        if raw.size >= 2:
            return float(raw[0]), float(raw[1])
        return None, None

    def _first_unwrapped_env(self):
        env = self.eval_env
        while hasattr(env, 'venv'):
            env = env.venv
        if hasattr(env, 'envs') and env.envs:
            env = env.envs[0]
        return getattr(env, 'unwrapped', env)


class EpisodeMetricsCallback(BaseCallback):
    """
    Logs per-episode lap_time, soc_final, energy_deployed, collision flag from env info.
    SAC uses 1 env so we just peek at infos[0].
    """

    def __init__(self):
        super().__init__()
        self._buffer = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [])
        for env_i, done in enumerate(dones):
            if not done:
                continue
            info = infos[env_i]

            term_info = info.get('terminal_info', info)
            laptime = term_info.get('lap_time')
            soc = term_info.get('soc')
            collided = bool(term_info.get('collision', False))
            lap_done = bool(term_info.get('lap_completed', False))
            truncated = bool(term_info.get('truncated', False))
            self._buffer.append((laptime, soc, collided))

            if lap_done and laptime is not None:
                self.logger.record('episode/lap_time_s', float(laptime))
            if soc is not None:
                self.logger.record('episode/soc_final', float(soc))
            self.logger.record('episode/collision', float(collided))
            self.logger.record('episode/lap_completed', float(lap_done))
            self.logger.record('episode/truncated', float(truncated))
            self.logger.record('episode/truncated_by_stall',
                               float(bool(term_info.get('truncated_by_stall', False))))
            
            # Reward decomposition: each component is a sum-over-episode so we
            # can verify which terms dominate the policy's chosen behavior.
            rb = term_info.get('reward_breakdown') or {}
            for k, v in rb.items():
                try:
                    self.logger.record(f'reward_decomp/{k}', float(v))
                except (TypeError, ValueError):
                    pass
            # Action / velocity stats per-episode (means + maxes).
            es = term_info.get('ep_stats') or {}
            for k, v in es.items():
                try:
                    self.logger.record(f'ep_stats/{k}', float(v))
                except (TypeError, ValueError):
                    pass
        return True


def _parse_ent(s: str):
    """SB3 SAC accepts 'auto', 'auto_<float>', or a float for ent_coef."""
    if s.startswith('auto'):
        return s
    return float(s)


def _parse_target_entropy(s: str):
    if s == 'auto':
        return 'auto'
    return float(s)


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', type=str, default='example_map')
    ap.add_argument('--total-steps', type=int, default=1_000_000)
    ap.add_argument('--decision-interval', type=int, default=1)
    ap.add_argument('--max-episode-steps', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', type=str, default='sac')
    ap.add_argument('--learning-rate', type=float, default=3e-4)
    ap.add_argument('--learning-starts', type=int, default=5_000)
    ap.add_argument('--n-envs', type=int, default=1)
    ap.add_argument('--speed-cap-mps', type=float, default=8.0)
    ap.add_argument('--deploy-speed-boost-mps', type=float, default=3.0)
    ap.add_argument('--downforce-k', type=float, default=0.02)
    ap.add_argument('--stall-decision-steps', type=int, default=0)
    ap.add_argument('--stall-min-progress-m', type=float, default=0.01)
    ap.add_argument('--stall-truncation-penalty', type=float, default=0.0)
    ap.add_argument('--ent-coef', type=str, default='auto')
    ap.add_argument('--target-entropy', type=str, default='auto')
    ap.add_argument('--energy-enabled', dest='energy_enabled', action='store_true')
    ap.add_argument('--no-energy-enabled', dest='energy_enabled', action='store_false')
    ap.set_defaults(energy_enabled=False)
    ap.add_argument('--progress-w', type=float, default=1.0)
    ap.add_argument('--step-penalty', type=float, default=0.01)
    ap.add_argument('--collision-penalty', type=float, default=5.0)
    ap.add_argument('--completion-bonus', type=float, default=3.0)
    ap.add_argument('--regen-penalty', type=float, default=0.0)
    ap.add_argument('--forward-reward-scale', type=float, default=0.0)
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    run_dir = Path('runs') / args.tag
    tb_dir = run_dir / 'tb'
    ckpt_dir = run_dir / 'ckpt'
    eval_dir = run_dir / 'eval'
    for d in (run_dir, tb_dir, ckpt_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    env_kwargs = dict(
        speed_cap_mps=args.speed_cap_mps,
        deploy_speed_boost_mps=args.deploy_speed_boost_mps,
        energy_enabled=args.energy_enabled,
        downforce_k=args.downforce_k,
        stall_decision_steps=args.stall_decision_steps,
        stall_min_progress_m=args.stall_min_progress_m,
        progress_reward_weight=args.progress_w,
        step_penalty=args.step_penalty,
        collision_penalty=args.collision_penalty,
        completion_bonus=args.completion_bonus,
        stall_truncation_penalty=args.stall_truncation_penalty,
        regen_penalty_weight=args.regen_penalty,
        forward_reward_scale=args.forward_reward_scale,
    )

    env_fns = [make_env(args.map, args.decision_interval, args.max_episode_steps,
                        args.seed, str(run_dir), rank=i, **env_kwargs)
               for i in range(args.n_envs)]
    if args.n_envs > 1:
        train_env = SubprocVecEnv(env_fns, start_method='fork')
    else:
        train_env = DummyVecEnv(env_fns)

    eval_env = DummyVecEnv([
        make_env(args.map, args.decision_interval, args.max_episode_steps,
                 args.seed + 1000, str(eval_dir), rank=0, **env_kwargs)
    ])

    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
                            training=False)

    eval_env.obs_rms = train_env.obs_rms
    eval_env.ret_rms = train_env.ret_rms

    model = SAC(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=300_000,
        learning_starts=args.learning_starts,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef=_parse_ent(args.ent_coef),
        target_entropy=_parse_target_entropy(args.target_entropy),
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=str(tb_dir),
        seed=args.seed,
        device='auto',
        verbose=1,
    )

    # SB3 callbacks count vec-env _on_step calls (one per n_envs env-steps), so
    # divide the user-specified env-step thresholds by n_envs.
    eval_freq_eff = max(1, 25_000 // args.n_envs)
    ckpt_freq_eff = max(1, 25_000 // args.n_envs)

    ckpt_cb = CheckpointCallback(
        save_freq=ckpt_freq_eff,
        save_path=str(ckpt_dir),
        name_prefix='ckpt',
        save_vecnormalize=True,
    )
    eval_cb = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(eval_dir),
        eval_freq=eval_freq_eff,
        n_eval_episodes=3,
        deterministic=True,
        render=False,
        callback_on_new_best=SaveVecNormalizeOnBestCallback(run_dir),
    )
    metrics_cb = EpisodeMetricsCallback()
    clean_metrics_cb = CleanEvalMetricsCallback(
        eval_env, eval_freq=eval_freq_eff,
        fastest_lap_dir=run_dir,
    )

    model.learn(
        total_timesteps=args.total_steps,
        callback=[ckpt_cb, eval_cb, clean_metrics_cb, metrics_cb],
        progress_bar=False,
        tb_log_name='sac',
    )

    model.save(run_dir / 'final')
    train_env.save(run_dir / 'vecnorm.pkl')
    print(f"Saved final model to {run_dir / 'final.zip'}")
    print(f"TensorBoard:  tensorboard --logdir {tb_dir}")


if __name__ == '__main__':
    main()
