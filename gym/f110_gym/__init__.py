from gym.envs.registration import register as gym_register

try:
	from gymnasium.envs.registration import register as gymnasium_register
except ModuleNotFoundError:  # pragma: no cover - optional RL dependency
	gymnasium_register = None


gym_register(
	id='f110-v0',
	entry_point='f110_gym.envs:F110Env',
	)

gym_register(
	id='f110-driving-v0',
	entry_point='f110_gym.envs:F1DrivingEnv',
	)

if gymnasium_register is not None:
	gymnasium_register(
		id='f110-driving-v0',
		entry_point='f110_gym.envs:F1DrivingEnv',
		)
