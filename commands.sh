#!/usr/bin/env bash
# F1 2026 energy-RL training + eval recipes.
#
# Each block is independent. Blocks are not chained — copy/paste the train
# call and its eval calls per experiment. A full train run is 1.5M steps
# (~hours on CPU); evals are seconds.
#
# Tags follow `sac_jorrel_<variant>` so artifacts land under
# `runs/<tag>/`. Eval consumes:
#   - runs/<tag>/best_model.zip      + vecnorm.pkl              → viz_best
#   - runs/<tag>/fastest_lap.zip     + fastest_lap_vecnorm.pkl  → viz_fastest

# ============================================================================
# example_map — energy on (A_scratch)
# ============================================================================
python -u training/train_sac.py \
    --tag sac_jorrel_di1_energy_A_scratch \
    --map example_map \
    --decision-interval 1 --max-episode-steps 0 \
    --total-steps 1500000 \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 --stall-truncation-penalty 5 \
    --energy-enabled \
    --ent-coef auto_1.0 --target-entropy -1.5 \
    --collision-penalty 15 --completion-bonus 100 \
    --progress-w 1.0 --step-penalty 0.01 \
    --learning-rate 3e-4 --learning-starts 5000 --n-envs 4 --seed 0 \
    2>&1 | tee runs/sac_jorrel_di1_energy_A_scratch.log

# eval best
python training/eval_driving.py \
    --checkpoint runs/sac_jorrel_di1_energy_A_scratch/best_model.zip \
    --vecnorm runs/sac_jorrel_di1_energy_A_scratch/vecnorm.pkl \
    --episodes 1 --max-steps 12000 \
    --map example_map \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 \
    --energy-enabled \
    --out runs/sac_jorrel_di1_energy_A_scratch/viz_best \
    --save-gif runs/sac_jorrel_di1_energy_A_scratch/viz_best/best.mp4 \
    --gif-stride 8 --gif-fps 30 --render human_fast

# eval fastest lap
python training/eval_driving.py \
    --checkpoint runs/sac_jorrel_di1_energy_A_scratch/fastest_lap.zip \
    --vecnorm runs/sac_jorrel_di1_energy_A_scratch/fastest_lap_vecnorm.pkl \
    --episodes 1 --max-steps 12000 \
    --map example_map \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 \
    --energy-enabled \
    --out runs/sac_jorrel_di1_energy_A_scratch/viz_fastest \
    --save-gif runs/sac_jorrel_di1_energy_A_scratch/viz_fastest/fastest.mp4 \
    --gif-stride 8 --gif-fps 30 --render human_fast


# ============================================================================
# example_map — no-energy baseline (A_scratch_noenergy)
# ============================================================================
python -u training/train_sac.py \
    --tag sac_jorrel_A_scratch_noenergy \
    --map example_map \
    --decision-interval 1 --max-episode-steps 0 \
    --total-steps 1500000 \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 --stall-truncation-penalty 5 \
    --no-energy-enabled \
    --ent-coef auto_1.0 --target-entropy -1.5 \
    --collision-penalty 15 --completion-bonus 100 \
    --progress-w 1.0 --step-penalty 0.01 \
    --learning-rate 3e-4 --learning-starts 5000 --n-envs 4 --seed 0 \
    2>&1 | tee runs/sac_jorrel_A_scratch_noenergy.log

# eval best
python training/eval_driving.py \
    --checkpoint runs/sac_jorrel_A_scratch_noenergy/best_model.zip \
    --vecnorm runs/sac_jorrel_A_scratch_noenergy/vecnorm.pkl \
    --episodes 1 --max-steps 12000 \
    --map example_map \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 \
    --no-energy-enabled \
    --out runs/sac_jorrel_A_scratch_noenergy/viz_best \
    --save-gif runs/sac_jorrel_A_scratch_noenergy/viz_best/best.mp4 \
    --gif-stride 8 --gif-fps 30 --render human_fast

# eval fastest lap (3 episodes)
python training/eval_driving.py \
    --checkpoint runs/sac_jorrel_A_scratch_noenergy/fastest_lap.zip \
    --vecnorm runs/sac_jorrel_A_scratch_noenergy/fastest_lap_vecnorm.pkl \
    --episodes 3 --max-steps 12000 \
    --map example_map \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 \
    --no-energy-enabled \
    --out runs/sac_jorrel_A_scratch_noenergy/viz_fastest \
    --save-gif runs/sac_jorrel_A_scratch_noenergy/viz_fastest/fastest.mp4 \
    --gif-stride 8 --gif-fps 30 --render human_fast

# ============================================================================
# Monza — energy on
# ============================================================================
python -u training/train_sac.py \
    --tag sac_jorrel_di1_energy_Monza \
    --map Monza \
    --decision-interval 1 --max-episode-steps 0 \
    --total-steps 1500000 \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 --stall-truncation-penalty 5 \
    --energy-enabled \
    --ent-coef auto_1.0 --target-entropy -1.5 \
    --collision-penalty 15 --completion-bonus 100 \
    --progress-w 1.0 --step-penalty 0.01 \
    --learning-rate 3e-4 --learning-starts 5000 --n-envs 4 --seed 0 \
    2>&1 | tee runs/sac_jorrel_di1_energy_Monza.log

# eval fastest lap
python training/eval_driving.py \
    --checkpoint runs/sac_jorrel_di1_energy_Monza/fastest_lap.zip \
    --vecnorm runs/sac_jorrel_di1_energy_Monza/fastest_lap_vecnorm.pkl \
    --episodes 1 --max-steps 12000 \
    --map Monza \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 \
    --energy-enabled \
    --out runs/sac_jorrel_di1_energy_Monza/viz_fastest \
    --save-gif runs/sac_jorrel_di1_energy_Monza/viz_fastest/fastest.mp4 \
    --gif-stride 8 --gif-fps 30 --render human_fast

# ============================================================================
# Spa — longer track (eval --max-steps 16000)
# ============================================================================
python -u training/train_sac.py \
    --tag sac_jorrel_di1_energy_Spa \
    --map Spa \
    --decision-interval 1 --max-episode-steps 0 \
    --total-steps 1500000 \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 --stall-truncation-penalty 5 \
    --energy-enabled \
    --ent-coef auto_1.0 --target-entropy -1.5 \
    --collision-penalty 15 --completion-bonus 100 \
    --progress-w 1.0 --step-penalty 0.01 \
    --learning-rate 3e-4 --learning-starts 5000 --n-envs 4 --seed 0 \
    2>&1 | tee runs/sac_jorrel_di1_energy_Spa.log

# eval best
python training/eval_driving.py \
    --checkpoint runs/sac_jorrel_di1_energy_Spa/best_model.zip \
    --vecnorm runs/sac_jorrel_di1_energy_Spa/vecnorm.pkl \
    --episodes 1 --max-steps 16000 \
    --map Spa \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 \
    --energy-enabled \
    --out runs/sac_jorrel_di1_energy_Spa/viz_best \
    --save-gif runs/sac_jorrel_di1_energy_Spa/viz_best/best.mp4 \
    --gif-stride 8 --gif-fps 30 --render human_fast

# eval fastest lap
python training/eval_driving.py \
    --checkpoint runs/sac_jorrel_di1_energy_Spa/fastest_lap.zip \
    --vecnorm runs/sac_jorrel_di1_energy_Spa/fastest_lap_vecnorm.pkl \
    --episodes 1 --max-steps 16000 \
    --map Spa \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 \
    --energy-enabled \
    --out runs/sac_jorrel_di1_energy_Spa/viz_fastest \
    --save-gif runs/sac_jorrel_di1_energy_Spa/viz_fastest/fastest.mp4 \
    --gif-stride 8 --gif-fps 30 --render human_fast


# ============================================================================
# Sakhir — energy on (v1)
# ============================================================================
python -u training/train_sac.py \
    --tag sac_jorrel_di1_energy_Sakhir \
    --map Sakhir \
    --decision-interval 1 --max-episode-steps 0 \
    --total-steps 1500000 \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 --stall-truncation-penalty 5 \
    --energy-enabled \
    --ent-coef auto_1.0 --target-entropy -1.5 \
    --collision-penalty 15 --completion-bonus 100 \
    --progress-w 1.0 --step-penalty 0.01 \
    --learning-rate 3e-4 --learning-starts 5000 --n-envs 4 --seed 0 \
    2>&1 | tee runs/sac_jorrel_di1_energy_Sakhir.log

# eval best
python training/eval_driving.py \
    --checkpoint runs/sac_jorrel_di1_energy_Sakhir/best_model.zip \
    --vecnorm runs/sac_jorrel_di1_energy_Sakhir/vecnorm.pkl \
    --episodes 1 --max-steps 12000 \
    --map Sakhir \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 \
    --energy-enabled \
    --out runs/sac_jorrel_di1_energy_Sakhir/viz_best \
    --save-gif runs/sac_jorrel_di1_energy_Sakhir/viz_best/best.mp4 \
    --gif-stride 8 --gif-fps 30 --render human_fast

# eval fastest lap
python training/eval_driving.py \
    --checkpoint runs/sac_jorrel_di1_energy_Sakhir/fastest_lap.zip \
    --vecnorm runs/sac_jorrel_di1_energy_Sakhir/fastest_lap_vecnorm.pkl \
    --episodes 1 --max-steps 12000 \
    --map Sakhir \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 \
    --energy-enabled \
    --out runs/sac_jorrel_di1_energy_Sakhir/viz_fastest \
    --save-gif runs/sac_jorrel_di1_energy_Sakhir/viz_fastest/fastest.mp4 \
    --gif-stride 8 --gif-fps 30 --render human_fast


# ============================================================================
# Sakhir v2 — adds --regen-penalty 0.05 + --forward-reward-scale 0.3
# ============================================================================
python -u training/train_sac.py \
    --tag sac_jorrel_di1_energy_Sakhir_v2 \
    --map Sakhir \
    --decision-interval 1 --max-episode-steps 0 \
    --total-steps 1500000 \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 --stall-truncation-penalty 5 \
    --energy-enabled \
    --ent-coef auto_1.0 --target-entropy -1.5 \
    --collision-penalty 15 --completion-bonus 100 \
    --progress-w 1.0 --step-penalty 0.01 \
    --regen-penalty 0.05 --forward-reward-scale 0.3 \
    --learning-rate 3e-4 --learning-starts 5000 --n-envs 4 --seed 0 \
    2>&1 | tee runs/sac_jorrel_di1_energy_Sakhir_v2.log

# eval best
python training/eval_driving.py \
    --checkpoint runs/sac_jorrel_di1_energy_Sakhir_v2/best_model.zip \
    --vecnorm runs/sac_jorrel_di1_energy_Sakhir_v2/vecnorm.pkl \
    --episodes 1 --max-steps 12000 \
    --map Sakhir \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 \
    --energy-enabled \
    --out runs/sac_jorrel_di1_energy_Sakhir_v2/viz_best \
    --save-gif runs/sac_jorrel_di1_energy_Sakhir_v2/viz_best/best.mp4 \
    --gif-stride 8 --gif-fps 30 --render human_fast

# eval fastest lap (--seed 1000 to match training's clean_eval rng)
python training/eval_driving.py \
    --checkpoint runs/sac_jorrel_di1_energy_Sakhir_v2/fastest_lap.zip \
    --vecnorm runs/sac_jorrel_di1_energy_Sakhir_v2/fastest_lap_vecnorm.pkl \
    --episodes 1 --max-steps 12000 \
    --map Sakhir --seed 1000 \
    --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 \
    --energy-enabled \
    --out runs/sac_jorrel_di1_energy_Sakhir_v2/viz_fastest \
    --save-gif runs/sac_jorrel_di1_energy_Sakhir_v2/viz_fastest/fastest.mp4 \
    --gif-stride 8 --gif-fps 30 --render human_fast
