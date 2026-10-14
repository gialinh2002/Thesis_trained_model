"""Exercise catalog (paper uses N=217 StrengthLog exercises + StrengthLevel 1RM refs).

If you have a real catalog, supply a CSV with columns:
  name, primary_muscle, secondary_muscles ('|' separated), category (compound/isolation),
  equipment, difficulty (0-1), fatigue_cost (0-1), base_1rm_ratio (1RM / bodyweight for an
  intermediate lifter of that sex, StrengthLevel style)
Otherwise a reproducible synthetic catalog is generated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MUSCLES = ["chest", "back", "shoulders", "biceps", "triceps", "quads", "hamstrings",
           "glutes", "calves", "core", "forearms", "traps"]
EQUIPMENT = ["barbell", "dumbbell", "machine", "cable", "bodyweight", "kettlebell"]
MOVES = ["press", "row", "curl", "extension", "raise", "squat", "deadlift", "lunge", "fly",
         "pulldown", "pull-up", "hold", "thrust", "shrug", "crunch"]


def generate_catalog(n: int = 217, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        prim = MUSCLES[i % len(MUSCLES)]
        sec = rng.choice([m for m in MUSCLES if m != prim], size=rng.integers(0, 3), replace=False)
        compound = len(sec) >= 1 and rng.random() < 0.7
        eq = rng.choice(EQUIPMENT)
        rows.append({
            "exercise_id": i,
            "name": f"{eq} {prim} {rng.choice(MOVES)} #{i}",
            "primary_muscle": prim,
            "secondary_muscles": "|".join(sec),
            "category": "compound" if compound else "isolation",
            "equipment": eq,
            "difficulty": round(float(rng.beta(2, 3) + (0.2 if compound else 0)), 3),
            "fatigue_cost": round(float(np.clip(rng.normal(0.65 if compound else 0.35, 0.12), 0.05, 1)), 3),
            "base_1rm_ratio": round(float(np.clip(rng.normal(1.1 if compound else 0.35, 0.25), 0.05, 2.5)), 3),
        })
    return pd.DataFrame(rows)


def load_catalog(path: str | Path, n: int = 217) -> pd.DataFrame:
    p = Path(path)
    if p.exists():
        df = pd.read_csv(p)
        if "exercise_id" not in df:
            df.insert(0, "exercise_id", range(len(df)))
        df["secondary_muscles"] = df["secondary_muscles"].fillna("")
        return df
    df = generate_catalog(n)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df


def muscle_matrix(catalog: pd.DataFrame) -> np.ndarray:
    """(N, M) involvement: 1.0 primary, 0.5 secondary."""
    m = np.zeros((len(catalog), len(MUSCLES)), dtype=np.float32)
    for i, r in catalog.reset_index(drop=True).iterrows():
        m[i, MUSCLES.index(r.primary_muscle)] = 1.0
        for s in filter(None, str(r.secondary_muscles).split("|")):
            if s in MUSCLES:
                m[i, MUSCLES.index(s)] = max(m[i, MUSCLES.index(s)], 0.5)
    return m
