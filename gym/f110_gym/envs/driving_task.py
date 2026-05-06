from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DrivingTaskConfig:
    speed_cap_mps: float = 6.0
    energy_enabled: bool = False
    # MGU-K deploy lifts the v_set ceiling above speed_cap by up to this
    # many m/s (scaled by deploy command in [0, 1]). Real F1 ERS gives a
    # short top-speed bump on commanded deploy; without lift the PID
    # never demands the boosted a_max so deploy has no kinetic effect at
    # cap. Set to 0.0 to disable.
    deploy_speed_boost_mps: float = 3.0
    # Minimum commanded v_set after action decode. Default 0 = action can
    # demand a full stop. Set > 0 (e.g. 0.5) to floor the speed channel so
    # the actor cannot collapse to a "stand still" deterministic mean: the
    # policy can still slow via brake/regen but nevr commands zero throttle.
    v_set_min_mps: float = 0.0

    # Per-step arc-length progress (clipped) is the dominant signal. Velocity
    # bonus stays opt-in for legacy energy runs. step_penalty is absolute per
    # env step, not scaled by dt.
    progress_reward_weight: float = 1.0
    forward_reward_scale: float = 0.0
    step_penalty: float = 0.01
    lateral_penalty_weight: float = 0.0
    collision_penalty: float = 5.0
    completion_bonus: float = 3.0
    # Penalty applied when episode is truncated by the stall detector
    # (creep/no-progress). Distinct from collision: stall-truncate sets
    # truncated=True (not terminated), so SAC bootstraps from V(s_T)
    # instead of treating it as a true terminal. Penalty discourages the
    # creep basin without poisoning value estimates as a hard terminal.
    stall_truncation_penalty: float = 0.0
    # Per-step penalty on regen command magnitude (action[2] < 0). Shaped to
    # discourage full-regen-spam basins where the policy parks itself in
    # max-regen, gets engine-cut clamped, then stalls. Cost = w * |deploy|
    # when deploy < 0. Default 0 = no shaping.
    regen_penalty_weight: float = 0.0
    # Δs clip protects against centerline-projection jumps near lap wrap.
    delta_s_clip_pos: float = 5.0
    delta_s_clip_neg: float = -1.0

    @classmethod
    def stage0(cls):
        return cls(speed_cap_mps=6.0, energy_enabled=False, step_penalty=0.01)

    @classmethod
    def stage1(cls):
        return cls(speed_cap_mps=5.0, energy_enabled=False, step_penalty=0.0)

    @classmethod
    def stage2(cls):
        return cls(speed_cap_mps=8.0, energy_enabled=True, step_penalty=0.0)


class ActionAdapter:
    def __init__(self, base_params: dict, cfg: DrivingTaskConfig):
        self._s_min = float(base_params['s_min'])
        self._s_max = float(base_params['s_max'])
        self._cfg = cfg

    def decode(self, action) -> tuple[float, float, float]:
        a = np.asarray(action, dtype=np.float64).flatten()
        a = np.clip(a, -1.0, 1.0)
        steer = self._s_min + 0.5 * (a[0] + 1.0) * (self._s_max - self._s_min)
        # Speed channel maps [-1, 1] -> [v_set_min, speed_cap]. No reverse.
        v_min = float(self._cfg.v_set_min_mps)
        v_cap = float(self._cfg.speed_cap_mps)
        v_set = v_min + 0.5 * (a[1] + 1.0) * (v_cap - v_min)
        deploy = float(a[2]) if self._cfg.energy_enabled else 0.0
        return float(steer), float(v_set), float(deploy)


class RewardModel:
    def __init__(self, cfg: DrivingTaskConfig):
        self._cfg = cfg

    def step_reward(
        self,
        *,
        delta_s: float,
        v_forward: float,
        lateral_dev: float,
        collided: bool,
        completed: bool,
        dt: float,
        stalled: bool = False,
        deploy: float = 0.0,
    ) -> float:
        cfg = self._cfg
        ds = float(delta_s)
        if ds > cfg.delta_s_clip_pos:
            ds = cfg.delta_s_clip_pos
        elif ds < cfg.delta_s_clip_neg:
            ds = cfg.delta_s_clip_neg
        reward = cfg.progress_reward_weight * ds
        reward += cfg.forward_reward_scale * max(0.0, float(v_forward)) * float(dt)
        reward -= cfg.lateral_penalty_weight * abs(float(lateral_dev))
        reward -= cfg.step_penalty
        if collided:
            reward -= float(cfg.collision_penalty)
        if completed:
            reward += float(cfg.completion_bonus)
        if stalled:
            reward -= float(cfg.stall_truncation_penalty)
        if cfg.regen_penalty_weight > 0.0 and deploy < 0.0:
            reward -= float(cfg.regen_penalty_weight) * abs(float(deploy))
        return float(reward)
