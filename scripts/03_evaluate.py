"""Step 3: compare PPO vs Greedy / PSO / Random (mean return + 95% bootstrap CI, completion, components).

  python scripts/03_evaluate.py --mode FPS --model runs/FPS_dynamic/best/best_model.zip
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gymrl.baselines import GreedyAgent, PSOAgent, RandomAgent
from gymrl.config import load_config
from gymrl.factory import make_env

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--mode"); ap.add_argument("--pool")
ap.add_argument("--model", help="path to PPO .zip (optional)")
ap.add_argument("--episodes", type=int)
ap.add_argument("--agents", default="ppo,greedy,pso,random")
args = ap.parse_args()
cfg = load_config(args.config)
if args.mode: cfg["env"]["mode"] = args.mode
if args.pool: cfg["env"]["user_pool"] = args.pool
E = args.episodes or cfg["eval"]["episodes"]


def boot_ci(x, n=2000, rng=np.random.default_rng(0)):
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def run(name):
    env = make_env(cfg, seed=cfg["seed"] + 100)
    if name == "ppo":
        from stable_baselines3 import PPO
        if not args.model: return None
        m = PPO.load(args.model); act = lambda o: m.predict(o, deterministic=True)[0]
    else:
        agent = {"greedy": GreedyAgent, "pso": PSOAgent, "random": RandomAgent}[name](env); act = agent.act
    rets, comps = [], defaultdict(list)
    for ep in range(E if name != "pso" else min(E, 100)):
        o, _ = env.reset(); done = False
        while not done:
            o, r, done, _, info = env.step(act(o))
        rets.append(r)
        for k, v in info["components"].items(): comps[k].append(v)
    rets = np.array(rets)
    return {"mean_return": float(rets.mean()), "ci95": boot_ci(rets),
            **{f"mean_{k}": float(np.mean(v)) for k, v in comps.items()}}


res = {a: run(a) for a in args.agents.split(",")}
res = {k: v for k, v in res.items() if v}
print(json.dumps(res, indent=2))
out = Path(cfg["ppo"]["out_dir"]) / f"eval_{cfg['env']['mode']}_{cfg['env']['user_pool']}.json"
out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=2))
print("saved ->", out)
