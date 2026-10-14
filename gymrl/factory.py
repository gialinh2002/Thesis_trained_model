from pathlib import Path

import pandas as pd

from gymrl.data.catalog import load_catalog
from gymrl.data.fitrec import synthetic_profiles
from gymrl.envs.gym_workout_env import GymWorkoutEnv


def load_profiles(cfg) -> pd.DataFrame:
    p = Path(cfg["data"]["profiles_out"])
    if p.exists():
        return pd.read_parquet(p)
    print(f"[warn] {p} not found -> using synthetic profiles. Run scripts/01_prepare_data.py first.")
    return synthetic_profiles(seed=cfg["seed"])


def make_env(cfg, seed=None, profiles=None, catalog=None):
    catalog = catalog if catalog is not None else load_catalog(cfg["data"]["catalog_path"], cfg["data"]["catalog_size"])
    profiles = profiles if profiles is not None else load_profiles(cfg)
    return GymWorkoutEnv(cfg, catalog, profiles, seed=seed)
