"""Step 2: train PPO (stable-baselines3) on one environment.

  python scripts/02_train_ppo.py --mode FPS --pool dynamic
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from gymrl.config import load_config
from gymrl.data.catalog import load_catalog
from gymrl.factory import load_profiles, make_env

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/default.yaml")
ap.add_argument("--mode", choices=["EONS", "EOS", "FPNS", "FPS"])
ap.add_argument("--pool", choices=["static", "dynamic"])
ap.add_argument("--timesteps", type=int)
args = ap.parse_args()

cfg = load_config(args.config)
if args.mode: cfg["env"]["mode"] = args.mode
if args.pool: cfg["env"]["user_pool"] = args.pool
p = cfg["ppo"]
steps = args.timesteps or p["total_timesteps"]

profiles = load_profiles(cfg)
train = profiles.sample(frac=0.8, random_state=cfg["seed"])
val = profiles.drop(train.index)
catalog = load_catalog(cfg["data"]["catalog_path"], cfg["data"]["catalog_size"])

run = Path(p["out_dir"]) / f"{cfg['env']['mode']}_{cfg['env']['user_pool']}"
run.mkdir(parents=True, exist_ok=True)

venv = make_vec_env(lambda: make_env(cfg, profiles=train, catalog=catalog), n_envs=p["n_envs"], seed=cfg["seed"])
eval_env = make_vec_env(lambda: make_env(cfg, profiles=val, catalog=catalog), n_envs=1, seed=cfg["seed"] + 1)

model = PPO("MlpPolicy", venv, learning_rate=p["learning_rate"], n_steps=p["n_steps"],
            batch_size=p["batch_size"], n_epochs=p["n_epochs"], gamma=p["gamma"],
            gae_lambda=p["gae_lambda"], clip_range=p["clip_range"], ent_coef=p["ent_coef"],
            policy_kwargs={"net_arch": p["net_arch"]}, tensorboard_log=str(run / "tb"),
            seed=cfg["seed"], verbose=1)
model.learn(total_timesteps=steps, callback=[
    CheckpointCallback(save_freq=max(100_000 // p["n_envs"], 1), save_path=str(run / "ckpt")),
    EvalCallback(eval_env, best_model_save_path=str(run / "best"), eval_freq=max(50_000 // p["n_envs"], 1),
                 n_eval_episodes=200, deterministic=True),
])
model.save(run / "final_model")
print("saved ->", run / "final_model.zip")
