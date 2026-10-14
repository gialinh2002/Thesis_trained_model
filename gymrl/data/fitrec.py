"""FitRec (Endomondo) download + conversion into user profiles.

FitRec records are cardio workouts (heart rate, speed, altitude, sport). The paper's
simulator needs *gym* user profiles: age, sex, height, weight/BMI, goal, experience,
training frequency and capacity. We derive these from FitRec signals:

  experience  <- number of workouts + span of activity
  frequency   <- workouts per week
  capacity    <- normalised cardio fitness (resting-ish HR, HR reserve, speed)
  goal        <- dominant sport family (endurance / strength / weight-loss / general)
  sex         <- FitRec `gender`
  age/height/weight  <- not in FitRec -> sampled from population priors, conditioned on sex
                        (replace with your own data if available)
"""
from __future__ import annotations

import ast
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ENDURANCE = {"run", "bike", "mountain bike", "bike (transport)", "orienteering", "roller skiing",
             "cross-country skiing", "rowing", "indoor cycling", "treadmill running"}
STRENGTH = {"weight training", "fitness walking", "core stability training", "circuit training",
            "gymnastics", "crossfit"}
WEIGHT_LOSS = {"walk", "hiking", "elliptical", "aerobics", "step counter"}

GOALS = ["strength", "hypertrophy", "endurance", "weight_loss", "general"]
EXPERIENCE = ["beginner", "intermediate", "advanced"]


def download(url: str, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"[fitrec] already present: {out}")
        return out
    print(f"[fitrec] downloading {url} -> {out}")
    try:
        urllib.request.urlretrieve(url, out)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"Download failed ({e}). Download endomondoHR.json.gz manually from "
            "https://sites.google.com/eng.ucsd.edu/fitrec-project/home and place it at " + str(out)
        )
    return out


def _parse_line(line: str) -> dict:
    line = line.strip()
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)  # FitRec files are python-dict literals


def iter_workouts(path: str | Path, max_workouts: int | None = None):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_workouts and i >= max_workouts:
                break
            if line.strip():
                yield _parse_line(line)


def summarise_workout(w: dict) -> dict:
    hr = np.asarray(w.get("heart_rate") or [np.nan], dtype=float)
    sp = np.asarray(w.get("speed") or [np.nan], dtype=float)
    ts = np.asarray(w.get("timestamp") or [0], dtype=float)
    return {
        "userId": w.get("userId"),
        "sport": str(w.get("sport", "")).lower(),
        "gender": str(w.get("gender", "unknown")).lower(),
        "start": float(ts[0]) if ts.size else 0.0,
        "duration_min": float((ts[-1] - ts[0]) / 60.0) if ts.size > 1 else 0.0,
        "hr_mean": float(np.nanmean(hr)),
        "hr_max": float(np.nanmax(hr)),
        "hr_min": float(np.nanmin(hr)),
        "speed_mean": float(np.nanmean(sp)) if np.isfinite(sp).any() else 0.0,
    }


def _goal_from_sports(sports: pd.Series, rng) -> str:
    fam = sports.map(lambda s: "endurance" if s in ENDURANCE else "strength" if s in STRENGTH
                     else "weight_loss" if s in WEIGHT_LOSS else "general")
    top = fam.value_counts().idxmax()
    if top == "strength":
        return rng.choice(["strength", "hypertrophy"])
    return top


def build_profiles(raw_path: str | Path, max_workouts: int | None, min_workouts: int,
                   seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = [summarise_workout(w) for w in tqdm(iter_workouts(raw_path, max_workouts), desc="workouts")]
    df = pd.DataFrame(rows).dropna(subset=["userId"])
    grouped = df.groupby("userId")
    out = []
    for uid, g in tqdm(grouped, desc="users"):
        if len(g) < min_workouts:
            continue
        span_weeks = max((g.start.max() - g.start.min()) / (7 * 86400), 1.0)
        freq = float(np.clip(len(g) / span_weeks, 1, 7))
        n = len(g)
        exp = "beginner" if n < 20 else "intermediate" if n < 80 else "advanced"
        hr_reserve = (g.hr_max.median() - g.hr_min.median())
        sex = g.gender.mode().iat[0] if not g.gender.mode().empty else "unknown"
        male = sex == "male"
        height = rng.normal(178 if male else 165, 7)
        bmi = float(np.clip(rng.normal(25 if male else 24, 3.5), 17, 40))
        out.append({
            "user_id": uid,
            "sex": sex,
            "age": int(np.clip(rng.normal(35, 10), 18, 70)),
            "height_cm": round(float(height), 1),
            "weight_kg": round(bmi * (height / 100) ** 2, 1),
            "bmi": round(bmi, 1),
            "goal": _goal_from_sports(g.sport, rng),
            "experience": exp,
            "frequency_per_week": round(freq, 2),
            "n_workouts": n,
            "hr_reserve": float(hr_reserve),
            "speed_mean": float(g.speed_mean.median()),
            "duration_mean": float(g.duration_min.median()),
        })
    prof = pd.DataFrame(out)
    # capacity in [0.3, 1.0]: blend of cardio fitness, experience, frequency
    z = lambda s: (s - s.mean()) / (s.std() + 1e-8)
    exp_num = prof.experience.map({"beginner": 0, "intermediate": 1, "advanced": 2})
    raw = 0.4 * z(prof.hr_reserve) + 0.2 * z(prof.speed_mean) + 0.25 * z(exp_num) + 0.15 * z(prof.frequency_per_week)
    prof["capacity"] = (0.3 + 0.7 * (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)).round(3)
    return prof


def synthetic_profiles(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Fallback when FitRec is not available (also used by tests)."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["male", "female"], n)
    h = np.where(sex == "male", rng.normal(178, 7, n), rng.normal(165, 7, n))
    bmi = np.clip(rng.normal(24.5, 3.5, n), 17, 40)
    return pd.DataFrame({
        "user_id": np.arange(n), "sex": sex, "age": np.clip(rng.normal(35, 10, n), 18, 70).astype(int),
        "height_cm": h.round(1), "weight_kg": (bmi * (h / 100) ** 2).round(1), "bmi": bmi.round(1),
        "goal": rng.choice(GOALS, n), "experience": rng.choice(EXPERIENCE, n, p=[.4, .4, .2]),
        "frequency_per_week": rng.uniform(2, 6, n).round(2), "capacity": rng.uniform(0.3, 1.0, n).round(3),
    })
