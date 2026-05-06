# F1 2026 Energy-Management RL

Reinforcement-learning environment + training scripts for studying time-discrete
energy-deployment / regen strategy under F1 2026 regulations (no MGU-H, ~3×
MGU-K power, same battery capacity).

Fork of [`f1tenth_gym`](https://github.com/f1tenth/f1tenth_gym), stripped to a
single-track (ST) dynamics core wrapped by `F1DrivingEnv`: a Gymnasium env
with continuous LiDAR + path-feature observations, three-axis action
(steer / v_set / deploy), and an `EnergyManagedCar` battery model with
SOC-gated deploy and regen-on-brake harvest.

## Project context

COS 435 / ECE 433 final project (Princeton, Spring 2026). Karit Matanachai,
Evan Lin, Jorrel Rajan.

## Install

```bash
python -m venv .venv-rl
source .venv-rl/bin/activate
pip install -e '.[rl]'
```

## Train

```bash
python -u training/train_sac.py \
    --tag my_run \
    --map example_map \
    --total-steps 1500000 \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 --stall-truncation-penalty 5 \
    --energy-enabled \
    --ent-coef auto_1.0 --target-entropy -1.5 \
    --collision-penalty 15 --completion-bonus 100 \
    --progress-w 1.0 --step-penalty 0.01 \
    --learning-rate 3e-4 --learning-starts 5000 --n-envs 4 --seed 0 \
    2>&1 | tee runs/my_run.log
```

Pass `--no-energy-enabled` for the no-SOC baseline.

Artifacts written to `runs/<tag>/`:

| File                                          | Contents                                                        |
| --------------------------------------------- | --------------------------------------------------------------- |
| `best_model.zip`                              | EvalCallback's best-by-reward checkpoint                        |
| `vecnorm.pkl`                                 | VecNormalize obs stats: required to eval                        |
| `fastest_lap.zip` + `fastest_lap_vecnorm.pkl` | Fastest deterministic-eval lap seen during training             |
| `final.zip`                                   | End-of-training checkpoint                                      |
| `tb/`                                         | TensorBoard logs (lap_time, reward decomposition, action stats) |
| `ckpt/ckpt_*.zip`                             | Periodic checkpoints (every 25k env-steps)                      |

## Evaluate

```bash
python training/eval_driving.py \
  --checkpoint runs/my_run/best_model.zip \
  --vecnorm runs/my_run/vecnorm.pkl \
  --episodes 1 --max-steps 12000 \
  --map example_map \
  --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
  --stall-decision-steps 500 --stall-min-progress-m 0.005 \
  --energy-enabled \
  --out runs/my_run/viz \
  --save-gif runs/my_run/viz/run.mp4 --gif-stride 8 --gif-fps 30 \
  --render human_fast
```

Writes per-episode plots, traces, `eval_report.json`, and an MP4/GIF if
`--save-gif` is set.

`--render human_fast` opens a pyglet window during rollout. Drop the flag for
headless runs.

## Maps

`commands.sh` runs the four primary tracks: `example_map`, `Monza`, `Spa`,
`Sakhir`. Other circuits available under `racetracks/` (Austin, Catalunya,
Hockenheim, Silverstone, Suzuka-equivalents, etc.): set via `--map`.

`example_map` is the small F1Tenth test track; the rest are F1 1:10-scaled
circuits from the TUM/Heilmeier dataset. For longer tracks (Spa) bump
`--max-steps` on eval (e.g. 16000).

## Recipes

`commands.sh` contains the full set of training + evaluation invocations used
for the project (`example_map` energy/no-energy, Monza v2/v3, Spa, Sakhir
v1/v2). Use it as the canonical reference for hyperparameters per experiment.

## Code layout

```
training/
  train_sac.py        SAC trainer (entry point)
  eval_driving.py     deterministic eval + GIF/MP4 writer
  gif_utils.py        rendering helper

gym/f110_gym/envs/
  f1_driving_env.py   Gymnasium env (LiDAR + scalars, energy coupling)
  driving_task.py     DrivingTaskConfig, ActionAdapter, RewardModel
  energy_model.py     EnergyParams + EnergyManagedCar (SOC, deploy, regen)
  track_features.py   centerline / raceline progress + lateral dev
  base_classes.py     ST-only inner sim (RaceCar + Simulator)
  dynamic_models.py   vehicle_dynamics_st + vehicle_dynamics_ks
  tire_pacejka.py     Pacejka 89 lateral tire model
  laser_models.py     2D LiDAR scan simulator
  collision_models.py vertex / TTC collision logic
  rendering.py        pyglet renderer
  f110_env.py         legacy F110 Gym env (wrapped by F1DrivingEnv)
```

## License

Inherits the original f1tenth_gym MIT license: see `LICENSE`.
