"""Step 4 (inference): user data in -> multi-week workout timeline out.

Input  : a user profile JSON (age, sex, height_cm, weight_kg, goal, experience,
         frequency_per_week, capacity[optional]) — or --user-id from the processed FitRec profiles.
Output : timeline JSON/CSV: week, day, order, exercise, muscles, sets, reps, %1RM, working weight.

  python scripts/04_recommend_timeline.py --model runs/FPS_dynamic/best/best_model.zip \
      --profile examples/user.json --weeks 8 --out timeline.json
  (without --model the greedy baseline is used, useful to test the flow before training)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from gymrl.baselines import GreedyAgent
from gymrl.config import load_config
from gymrl.factory import load_profiles, make_env
from gymrl.users import make_user

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--model")
ap.add_argument("--profile", help="user JSON")
ap.add_argument("--user-id")
ap.add_argument("--weeks", type=int)
ap.add_argument("--simulate-skips", action="store_true", help="simulate user skips for what-if analysis")
ap.add_argument("--out", default="timeline.json")
args = ap.parse_args()

cfg = load_config(args.config)
cfg["env"]["mode"] = "FPS" if args.simulate_skips else "FPNS"   # full prescription for a usable plan
cfg["env"]["user_pool"] = "dynamic"
weeks = args.weeks or cfg["timeline"]["weeks"]

if args.profile:
    prof = json.loads(Path(args.profile).read_text())
    h = prof["height_cm"] / 100
    prof.setdefault("bmi", round(prof["weight_kg"] / h ** 2, 1))
    prof.setdefault("capacity", {"beginner": 0.45, "intermediate": 0.65, "advanced": 0.85}[prof["experience"]])
    prof.setdefault("user_id", "custom")
else:
    df = load_profiles(cfg)
    prof = df[df.user_id.astype(str) == str(args.user_id)].iloc[0].to_dict() if args.user_id else df.iloc[0].to_dict()

env = make_env(cfg, seed=cfg["seed"], profiles=pd.DataFrame([prof]))
user = make_user(prof, np.random.default_rng(cfg["seed"]))
env.set_user(user)

if args.model:
    from stable_baselines3 import PPO
    m = PPO.load(args.model); act = lambda o: m.predict(o, deterministic=True)[0]
else:
    print("[info] no --model given -> using Greedy baseline")
    g = GreedyAgent(env); act = g.act

per_week = int(round(prof["frequency_per_week"]))
day_slots = {1: [1], 2: [1, 4], 3: [1, 3, 5], 4: [1, 2, 4, 5], 5: [1, 2, 3, 5, 6], 6: [1, 2, 3, 4, 5, 6], 7: list(range(1, 8))}
rows, sessions = [], []
for w in range(1, weeks + 1):
    for d in day_slots[max(1, min(7, per_week))]:
        o, _ = env.reset(); done = False
        while not done:
            o, r, done, _, info = env.step(act(o))
        sess = {"week": w, "day": d, "reward": round(r, 3), "exercises": []}
        for i, s in enumerate(info["session"], 1):
            ex = env.catalog.loc[s["exercise"]]
            item = {"order": i, "exercise": ex["name"], "primary_muscle": ex["primary_muscle"],
                    "sets": s["sets"], "reps": s["reps"], "pct_1rm": round(s["load"] * 100),
                    "weight_kg": env.working_weight_kg(s), "skipped": s.get("skipped", False)}
            sess["exercises"].append(item); rows.append({"week": w, "day": d, **item})
        sessions.append(sess)

out = Path(args.out)
out.write_text(json.dumps({"user": prof, "weeks": weeks, "sessions": sessions}, indent=2, default=str))
pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
print(f"timeline: {len(sessions)} sessions -> {out} (+ .csv)")
print(pd.DataFrame(rows).head(16).to_string(index=False))
