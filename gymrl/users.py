"""Simulated users + skip model (hidden true capacity, online estimates from skip-only feedback)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gymrl.data.catalog import MUSCLES
from gymrl.data.fitrec import EXPERIENCE, GOALS

# Goal -> preferred routine (target rep range, sets, %1RM, muscle-split emphasis)
GOAL_RX = {
    "strength":    {"reps": (3, 6),   "sets": (3, 6), "load": (0.80, 0.95)},
    "hypertrophy": {"reps": (6, 12),  "sets": (3, 5), "load": (0.65, 0.80)},
    "endurance":   {"reps": (12, 20), "sets": (2, 4), "load": (0.40, 0.65)},
    "weight_loss": {"reps": (10, 15), "sets": (2, 4), "load": (0.50, 0.70)},
    "general":     {"reps": (8, 12),  "sets": (2, 4), "load": (0.60, 0.75)},
}
EXP_MULT = {"beginner": 0.75, "intermediate": 1.0, "advanced": 1.25}


@dataclass
class User:
    profile: dict
    true_capacity: float                     # hidden from the agent
    routine: np.ndarray                      # (M,) preferred muscle emphasis
    est_capacity: float = 0.6                # online estimate (observed)
    est_skip_bias: float = 0.0
    est_avoid: np.ndarray = field(default_factory=lambda: np.zeros(len(MUSCLES), np.float32))
    true_avoid: np.ndarray = field(default_factory=lambda: np.zeros(len(MUSCLES), np.float32))
    history: list = field(default_factory=list)   # list of past sessions (list of exercise ids)
    sessions_done: int = 0

    @property
    def goal(self) -> str:
        return self.profile["goal"]

    @property
    def experience(self) -> str:
        return self.profile["experience"]

    def features(self) -> np.ndarray:
        p = self.profile
        g = np.eye(len(GOALS))[GOALS.index(p["goal"])]
        e = np.eye(len(EXPERIENCE))[EXPERIENCE.index(p["experience"])]
        num = np.array([
            (p["age"] - 18) / 52, 1.0 if p["sex"] == "male" else 0.0, (p["bmi"] - 17) / 23,
            p["frequency_per_week"] / 7, self.est_capacity, np.tanh(self.est_skip_bias),
        ], np.float32)
        return np.concatenate([num, g, e, self.routine, self.est_avoid]).astype(np.float32)

    def advance(self, completion: float, workload: float, rng: np.random.Generator):
        """Dynamic pool: users adapt over sessions (progressive overload / detraining)."""
        self.sessions_done += 1
        gain = 0.01 * completion * EXP_MULT[self.experience] ** -1
        if workload > 1.3 * self.true_capacity:
            gain -= 0.01
        self.true_capacity = float(np.clip(self.true_capacity + gain + rng.normal(0, 0.003), 0.2, 1.2))
        if self.sessions_done % 12 == 0 and self.experience != "advanced":
            self.profile["experience"] = EXPERIENCE[EXPERIENCE.index(self.experience) + 1]


def make_user(profile: dict, rng: np.random.Generator) -> User:
    routine = rng.dirichlet(np.ones(len(MUSCLES)) * 0.8).astype(np.float32)
    avoid = (rng.random(len(MUSCLES)) < 0.1).astype(np.float32) * rng.uniform(0.3, 1.0, len(MUSCLES))
    cap = float(profile.get("capacity", 0.6))
    return User(profile=dict(profile), true_capacity=cap, routine=routine,
                est_capacity=float(np.clip(cap + rng.normal(0, 0.15), 0.2, 1.2)),
                true_avoid=avoid.astype(np.float32))


class SkipModel:
    """p(skip) = sigmoid(base + w1*workload/capacity + w2*strain + w3*mismatch + w4*recency + w5*pos)."""

    def __init__(self, cfg: dict):
        self.c = cfg

    def prob(self, user: User, workload: float, strain: float, mismatch: float,
             recency: float, position: float, avoid_hit: float) -> float:
        c = self.c
        z = (c["base"] + c["w_workload"] * (workload / max(user.true_capacity, 1e-3) - 1.0)
             + c["w_muscle_strain"] * strain + c["w_routine_mismatch"] * mismatch
             + c["w_recency"] * recency + c["w_position"] * position + 2.0 * avoid_hit)
        return float(1 / (1 + np.exp(-z)))

    def update(self, user: User, skipped: bool, muscles: np.ndarray, p_pred: float):
        lr = self.c["online_lr"]
        err = float(skipped) - p_pred
        user.est_skip_bias += lr * err
        user.est_capacity = float(np.clip(user.est_capacity - lr * 0.3 * err, 0.2, 1.2))
        user.est_avoid = np.clip(user.est_avoid + lr * err * muscles, 0, 1).astype(np.float32)
