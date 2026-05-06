python -u training/train_sac.py \
    --tag sac_jorrel_di1_energy_A_scratch \
    --map example_map \
    --decision-interval 1 \
    --max-episode-steps 0 \
    --total-steps 1500000 \
    --speed-cap-mps 8.0 \
    --deploy-speed-boost-mps 3.0 \
    --downforce-k 0.02 \
    --stall-decision-steps 500 \
    --stall-min-progress-m 0.005 \
    --stall-truncation-penalty 5 \
    --energy-enabled \
    --ent-coef auto_1.0 \
    --target-entropy -1.5 \
    --collision-penalty 15 \
    --completion-bonus 100 \
    --progress-w 1.0 \
    --step-penalty 0.01 \
    --learning-rate 3e-4 \
    --learning-starts 5000 \
    --n-envs 4 \
    --seed 0 2>&1 | tee runs/sac_jorrel_di1_energy_A_scratch.log
# A_scratch
python training/eval_driving.py \
  --checkpoint runs/sac_jorrel_di1_energy_A_scratch/best_model.zip \
  --vecnorm runs/sac_jorrel_di1_energy_A_scratch/vecnorm.pkl \
  --episodes 1 --max-steps 12000 \
  --map example_map \
  --speed-cap-mps 8.0 \
  --deploy-speed-boost-mps 3.0 \
  --downforce-k 0.02 \
  --stall-decision-steps 500 \
  --stall-min-progress-m 0.005 \
  --energy-enabled \
  --out runs/sac_jorrel_di1_energy_A_scratch/viz_midrun \
  --save-gif runs/sac_jorrel_di1_energy_A_scratch/viz_midrun/best.mp4 \
  --gif-stride 8 --gif-fps 30 \
  --render human_fast
# fastest lap
python training/eval_driving.py \
--checkpoint runs/sac_jorrel_di1_energy_A_scratch/fastest_lap.zip \
--vecnorm runs/sac_jorrel_di1_energy_A_scratch/fastest_lap_vecnorm.pkl \
--episodes 1 --max-steps 12000 \
--speed-cap-mps 8.0 \
--map example_map \
--speed-cap-mps 8.0 \
--deploy-speed-boost-mps 3.0 \
--downforce-k 0.02 \
--stall-decision-steps 500 \
--stall-min-progress-m 0.005 \
--energy-enabled \
--out runs/sac_jorrel_di1_energy_A_scratch/viz_fastest \
--save-gif runs/sac_jorrel_di1_energy_A_scratch/viz_fastest/fastest.mp4 \
--gif-stride 8 --gif-fps 30 \
--render human_fast

# no SOC
python -u training/train_sac.py \
  --tag sac_jorrel_A_scratch_noenergy \
  --map example_map \
  --decision-interval 1 \
  --max-episode-steps 0 \
  --total-steps 1500000 \
  --speed-cap-mps 8.0 \
  --deploy-speed-boost-mps 3.0 \
  --downforce-k 0.02 \
  --stall-decision-steps 500 \
  --stall-min-progress-m 0.005 \
  --stall-truncation-penalty 5 \
  --ent-coef auto_1.0 \
  --target-entropy -1.5 \
  --collision-penalty 15 \
  --completion-bonus 100 \
  --progress-w 1.0 \
  --step-penalty 0.01 \
  --learning-rate 3e-4 \
  --learning-starts 5000 \
  --n-envs 4 \
  --seed 0 2>&1 | tee runs/sac_jorrel_A_scratch_noenergy.log
python training/eval_driving.py \
--checkpoint runs/sac_jorrel_A_scratch_noenergy/best_model.zip \
--vecnorm runs/sac_jorrel_A_scratch_noenergy/vecnorm.pkl \
--episodes 1 --max-steps 12000 \
--map example_map \
--speed-cap-mps 8.0 \
--deploy-speed-boost-mps 3.0 \
--downforce-k 0.02 \
--stall-decision-steps 500 \
--stall-min-progress-m 0.005 \
--no-energy-enabled \
--out runs/sac_jorrel_A_scratch_noenergy/viz_best \
--save-gif runs/sac_jorrel_A_scratch_noenergy/viz_best/best.mp4 \
--gif-stride 8 --gif-fps 30 \
--render human_fast
# fastest
python training/eval_driving.py \
--checkpoint runs/sac_jorrel_A_scratch_noenergy/fastest_lap.zip \
--vecnorm runs/sac_jorrel_A_scratch_noenergy/fastest_lap_vecnorm.pkl \
--episodes 3 --max-steps 12000 \
--map example_map \
--speed-cap-mps 8.0 \
--deploy-speed-boost-mps 3.0 \
--downforce-k 0.02 \
--stall-decision-steps 500 \
--stall-min-progress-m 0.005 \
--no-energy-enabled \
--out runs/sac_jorrel_A_scratch_noenergy/viz_fastest \
--save-gif runs/sac_jorrel_A_scratch_noenergy/viz_fastest/fastest.mp4 \
--gif-stride 8 --gif-fps 30 \
--render human_fast


# no SOC v2
python -u training/train_sac.py \
  --tag sac_jorrel_A_scratch_noenergy_v2 \
  --map example_map \
  --decision-interval 1 \
  --max-episode-steps 0 \
  --total-steps 1500000 \
  --speed-cap-mps 8.0 \
  --deploy-speed-boost-mps 3.0 \
  --downforce-k 0.02 \
  --stall-decision-steps 500 \
  --stall-min-progress-m 0.005 \
  --stall-truncation-penalty 50 \
  --ent-coef auto_1.0 \
  --target-entropy -1.5 \
  --collision-penalty 15 \
  --completion-bonus 100 \
  --progress-w 1.0 \
  --step-penalty 0.01 \
  --learning-rate 3e-4 \
  --learning-starts 5000 \
  --n-envs 4 \
  --seed 0 2>&1 | tee runs/sac_jorrel_A_scratch_noenergy_v2.log
python training/eval_driving.py \
--checkpoint runs/sac_jorrel_A_scratch_noenergy_v2/best_model.zip \
--vecnorm runs/sac_jorrel_A_scratch_noenergy_v2/vecnorm.pkl \
--episodes 1 --max-steps 12000 \
--map example_map \
--speed-cap-mps 8.0 \
--deploy-speed-boost-mps 3.0 \
--downforce-k 0.02 \
--stall-decision-steps 500 \
--stall-min-progress-m 0.005 \
--no-energy-enabled \
--out runs/sac_jorrel_A_scratch_noenergy_v2/viz_best \
--save-gif runs/sac_jorrel_A_scratch_noenergy_v2/viz_best/best.mp4 \
--gif-stride 8 --gif-fps 30 \
--render human_fast


sac_jorrel_di1_energy_Monza/tb/sac_1
python -u training/train_sac.py \
    --tag sac_jorrel_di1_energy_Monza \
    --map Monza \
    --decision-interval 1 \
    --max-episode-steps 0 \
    --total-steps 1500000 \
    --speed-cap-mps 8.0 \
    --deploy-speed-boost-mps 3.0 \
    --downforce-k 0.02 \
    --stall-decision-steps 500 \
    --stall-min-progress-m 0.005 \
    --stall-truncation-penalty 5 \
    --energy-enabled \
    --ent-coef auto_1.0 \
    --target-entropy -1.5 \
    --collision-penalty 15 \
    --completion-bonus 100 \
    --progress-w 1.0 \
    --step-penalty 0.01 \
    --learning-rate 3e-4 \
    --learning-starts 5000 \
    --n-envs 4 \
    --seed 0 2>&1 | tee runs/sac_jorrel_di1_energy_Monza.log
python training/eval_driving.py \
  --checkpoint runs/sac_jorrel_di1_energy_Monza/fastest_lap.zip \
  --vecnorm runs/sac_jorrel_di1_energy_Monza/fastest_lap_vecnorm.pkl \
  --episodes 1 --max-steps 12000 \
  --map Monza \
  --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
  --stall-decision-steps 500 --stall-min-progress-m 0.005 \
  --energy-enabled \
  --out runs/sac_jorrel_di1_energy_Monza/viz_fastest \
  --save-gif runs/sac_jorrel_di1_energy_Monza/viz_fastest/best.mp4 \
  --gif-stride 8 --gif-fps 30 --render human_fast

# monza v2
python -u training/train_sac.py \
    --tag sac_jorrel_di1_energy_Monza_v2 \
    --map Monza --decision-interval 1 --max-episode-steps 0 \
    --total-steps 1500000 \
    --speed-cap-mps 14.0 --deploy-speed-boost-mps 5.0 --downforce-k 0.005 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 --stall-truncation-penalty 5 \
    --energy-enabled --ent-coef auto_1.0 --target-entropy -1.5 \
    --collision-penalty 15 --completion-bonus 100 \
    --progress-w 1.0 --step-penalty 0.01 \
    --learning-rate 3e-4 --learning-starts 5000 --n-envs 4 --seed 0 \
    2>&1 | tee runs/sac_jorrel_di1_energy_Monza_v2.log
python training/eval_driving.py \
  --checkpoint runs/sac_jorrel_di1_energy_Monza_v2/best_model.zip \
  --vecnorm runs/sac_jorrel_di1_energy_Monza_v2/vecnorm.pkl \
  --episodes 3 --max-steps 12000 \
  --map Monza \
  --speed-cap-mps 14.0 --deploy-speed-boost-mps 5.0 --downforce-k 0.005 \
  --energy-enabled \
  --stall-decision-steps 500 --stall-min-progress-m 0.005 \
  --out runs/sac_jorrel_di1_energy_Monza_v2/viz_best_model \
  --save-gif runs/sac_jorrel_di1_energy_Monza_v2/viz_best_model/best.mp4 \
  --gif-stride 8 --gif-fps 30 --render human_fast
# fastest
python training/eval_driving.py \
  --checkpoint runs/sac_jorrel_di1_energy_Monza_v2/fastest_lap.zip \
  --vecnorm runs/sac_jorrel_di1_energy_Monza_v2/fastest_lap_vecnorm.pkl \
  --episodes 3 --max-steps 12000 \
  --map Monza \
  --speed-cap-mps 14.0 --deploy-speed-boost-mps 5.0 --downforce-k 0.005 \
  --energy-enabled \
  --stall-decision-steps 500 --stall-min-progress-m 0.005 \
  --out runs/sac_jorrel_di1_energy_Monza_v2/viz_fastest \
  --save-gif runs/sac_jorrel_di1_energy_Monza_v2/viz_fastest/best.mp4 \
  --gif-stride 8 --gif-fps 30 --render human_fast

# monza v3
python -u training/train_sac.py \
    --tag sac_jorrel_di1_energy_Monza_v3 \
    --map Monza --decision-interval 1 --max-episode-steps 0 \
    --total-steps 1500000 \
    --speed-cap-mps 14.0 --deploy-speed-boost-mps 1.0 --downforce-k 0.005 \
    --stall-decision-steps 500 --stall-min-progress-m 0.005 --stall-truncation-penalty 5 \
    --energy-enabled \
    --ent-coef 0.2 \
    --collision-penalty 100 --completion-bonus 100 \
    --progress-w 1.0 --step-penalty 0.01 \
    --learning-rate 3e-4 --learning-starts 5000 --n-envs 4 --seed 0 \
    2>&1 | tee runs/sac_jorrel_di1_energy_Monza_v3.log
python training/eval_driving.py \
  --checkpoint runs/sac_jorrel_di1_energy_Monza_v3/best_model.zip \
  --vecnorm runs/sac_jorrel_di1_energy_Monza_v3/vecnorm.pkl \
  --episodes 1 --max-steps 12000 \
  --map Monza \
  --speed-cap-mps 14.0 --deploy-speed-boost-mps 1.0 --downforce-k 0.005 \
  --stall-decision-steps 500 --stall-min-progress-m 0.005 \
  --energy-enabled \
  --out runs/sac_jorrel_di1_energy_Monza_v3/viz_best \
  --save-gif runs/sac_jorrel_di1_energy_Monza_v3/viz_best/best.mp4 \
  --gif-stride 8 --gif-fps 30 --render human_fast



# Spa (longer track — eval --max-steps 16000)
python -u training/train_sac.py \
    --tag sac_jorrel_di1_energy_Spa \
    --map Spa \
    --decision-interval 1 \
    --max-episode-steps 0 \
    --total-steps 1500000 \
    --speed-cap-mps 8.0 \
    --deploy-speed-boost-mps 3.0 \
    --downforce-k 0.02 \
    --stall-decision-steps 500 \
    --stall-min-progress-m 0.005 \
    --stall-truncation-penalty 5 \
    --energy-enabled \
    --ent-coef auto_1.0 \
    --target-entropy -1.5 \
    --collision-penalty 15 \
    --completion-bonus 100 \
    --progress-w 1.0 \
    --step-penalty 0.01 \
    --learning-rate 3e-4 \
    --learning-starts 5000 \
    --n-envs 4 \
    --seed 0 2>&1 | tee runs/sac_jorrel_di1_energy_Spa.log

# eval best
python training/eval_driving.py \
  --checkpoint runs/sac_jorrel_di1_energy_Spa/best_model.zip \
  --vecnorm runs/sac_jorrel_di1_energy_Spa/vecnorm.pkl \
  --episodes 1 --max-steps 16000 \
  --map Spa \
  --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
  --stall-decision-steps 500 --stall-min-progress-m 0.005 \
  --energy-enabled \
  --out runs/sac_jorrel_di1_energy_Spa/viz_midrun \
  --save-gif runs/sac_jorrel_di1_energy_Spa/viz_midrun/best.mp4 \
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

# Sakhir
python -u training/train_sac.py \
    --tag sac_jorrel_di1_energy_Sakhir \
    --map Sakhir \
    --decision-interval 1 \
    --max-episode-steps 0 \
    --total-steps 1500000 \
    --speed-cap-mps 8.0 \
    --deploy-speed-boost-mps 3.0 \
    --downforce-k 0.02 \
    --stall-decision-steps 500 \
    --stall-min-progress-m 0.005 \
    --stall-truncation-penalty 5 \
    --energy-enabled \
    --ent-coef auto_1.0 \
    --target-entropy -1.5 \
    --collision-penalty 15 \
    --completion-bonus 100 \
    --progress-w 1.0 \
    --step-penalty 0.01 \
    --learning-rate 3e-4 \
    --learning-starts 5000 \
    --n-envs 4 \
    --seed 0 2>&1 | tee runs/sac_jorrel_di1_energy_Sakhir.log

# eval best
python training/eval_driving.py \
  --checkpoint runs/sac_jorrel_di1_energy_Sakhir/best_model.zip \
  --vecnorm runs/sac_jorrel_di1_energy_Sakhir/vecnorm.pkl \
  --episodes 1 --max-steps 12000 \
  --map Sakhir \
  --speed-cap-mps 8.0 --deploy-speed-boost-mps 3.0 --downforce-k 0.02 \
  --stall-decision-steps 500 --stall-min-progress-m 0.005 \
  --energy-enabled \
  --out runs/sac_jorrel_di1_energy_Sakhir/viz_midrun \
  --save-gif runs/sac_jorrel_di1_energy_Sakhir/viz_midrun/best.mp4 \
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

# sakhir v2
python -u training/train_sac.py \
    --tag sac_jorrel_di1_energy_Sakhir_v2 \
    --map Sakhir --decision-interval 1 --max-episode-steps 0 \
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

# eval fastest lap (only if any lap completed during training)
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

# TO USE

/Users/keithmatanachai/Desktop/f1tenth_gym/runs/sac_jorrel_di1_energy_A_scratch/viz_fastest
/Users/keithmatanachai/Desktop/f1tenth_gym/runs/sac_jorrel_di1_energy_A_scratch/tb/sac_3
sac_jorrel_di1_energy_A_scratch/tb/sac_3

runs/sac_jorrel_A_scratch_noenergy/final.zip
/Users/keithmatanachai/Desktop/f1tenth_gym/runs/sac_jorrel_A_scratch_noenergy/viz_fastest
sac_jorrel_A_scratch_noenergy/tb/sac_1	

/Users/keithmatanachai/Desktop/f1tenth_gym/runs/sac_jorrel_di1_energy_Monza
/Users/keithmatanachai/Desktop/f1tenth_gym/runs/sac_jorrel_di1_energy_Monza/viz_fastest

/Users/keithmatanachai/Desktop/f1tenth_gym/runs/sac_jorrel_di1_energy_Spa
/Users/keithmatanachai/Desktop/f1tenth_gym/runs/sac_jorrel_di1_energy_Spa/viz_fastest

sac_jorrel_di1_energy_Sakhir_v2/tb/sac_1
/Users/keithmatanachai/Desktop/f1tenth_gym/runs/sac_jorrel_di1_energy_Sakhir_v2/viz_fastest_seed1000