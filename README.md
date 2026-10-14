# Gym-Workout RL Recommender — full pipeline (not trained)

Implementation scaffold of *"Personalized Recommender Systems for Gym Workouts: A Reinforcement
Learning Approach"* (Rosema, Torkamaan et al.), using **FitRec (Endomondo)** as the user-data source.

```
 FitRec workouts ──► user profiles ──► simulator + skip model ──► Gymnasium env (EONS/EOS/FPNS/FPS)
 (HR, speed, sport)   (age, sex, BMI,      (hidden capacity,          │  T = 8 steps, terminal reward
                       goal, experience,    workload, muscles)        ▼
                       frequency, capacity)                     PPO training  ──► evaluation vs
                                                                      │           Greedy / PSO / Random
                                                                      ▼
                               user data (JSON) ──► policy ──► multi-week workout timeline (JSON/CSV)
```

## Input / output
- **Input (user data)**: age, sex, height, weight, goal (`strength|hypertrophy|endurance|weight_loss|general`),
  experience (`beginner|intermediate|advanced`), training frequency/week, optional capacity. See `examples/user.json`.
- **Output**: a timeline of sessions (week, day) — each with 8 ordered exercises and sets, reps, %1RM, working weight.

## Environments (Table 1)
| Env  | Action | Skips | Reward on |
|------|--------|-------|-----------|
| EONS | Discrete(217) | no | suggested session |
| EOS  | Discrete(217) | yes | completed session × completion |
| FPNS | MultiDiscrete(217, 5, 19, 21) | no | suggested session |
| FPS  | MultiDiscrete(217, 5, 19, 21) | yes | completed session × completion |

Reward: components (uniqueness, intra/inter diversity, routine alignment, fatigue, + prescription & persistence for FP),
each below threshold τ replaced by −0.5, weighted sum clipped to [−1, 1] (Eq. 2–4). All values in `configs/default.yaml`.

## Run
```bash
pip install -r requirements.txt
python scripts/01_prepare_data.py              # downloads FitRec, builds profiles + catalog
#   (or --synthetic to skip the download)
pytest -q                                      # smoke-test the 4 environments
python scripts/02_train_ppo.py --mode FPS --pool dynamic --timesteps 2000000
python scripts/03_evaluate.py  --mode FPS --model runs/FPS_dynamic/best/best_model.zip
python scripts/04_recommend_timeline.py --model runs/FPS_dynamic/best/best_model.zip \
       --profile examples/user.json --weeks 8 --out timeline.json
```
Before training, step 4 works without `--model` (uses the Greedy baseline) so you can test the whole flow.

## Important notes
- FitRec is cardio data — it has no age/height/weight or gym exercises. Those fields are sampled from
  population priors (`gymrl/data/fitrec.py`); replace them with real data if you have it.
- The exercise catalog is synthetic (217 items) unless you supply `data/processed/exercise_catalog.csv`
  (StrengthLog exercises + StrengthLevel 1RM ratios as in the paper). Column spec in `gymrl/data/catalog.py`.
- Reward weights/thresholds are placeholders: the paper puts the exact values in its supplementary material.
- The simulator is a research proxy, not clinically validated.
