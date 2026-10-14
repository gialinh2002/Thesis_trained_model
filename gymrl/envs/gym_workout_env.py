"""Gymnasium environments EONS / EOS / FPNS / FPS (Table 1 of the paper)."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from gymrl.data.catalog import MUSCLES, muscle_matrix
from gymrl.rewards import session_reward
from gymrl.users import SkipModel, User, make_user

MODES = {"EONS": (False, False), "EOS": (False, True), "FPNS": (True, False), "FPS": (True, True)}


class GymWorkoutEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, cfg: dict, catalog: pd.DataFrame, profiles: pd.DataFrame, seed: int | None = None):
        super().__init__()
        e = cfg["env"]
        self.cfg = cfg
        self.mode = e["mode"]
        self.full_rx, self.skips = MODES[self.mode]
        self.T = e["session_length"]
        self.dynamic = e["user_pool"] == "dynamic"
        self.hist_len = e["history_len"]
        self.catalog = catalog.reset_index(drop=True)
        self.N = len(catalog)
        self.M = len(MUSCLES)
        self.mus = muscle_matrix(self.catalog)
        self.fatigue = self.catalog["fatigue_cost"].to_numpy(np.float32)
        self.difficulty = self.catalog["difficulty"].to_numpy(np.float32)
        self.set_bins = np.array(e["set_bins"])
        self.rep_bins = np.array(e["rep_bins"])
        lb = e["load_bins"]
        self.load_bins = np.linspace(lb["low"], lb["high"], lb["n"])
        self.skip_model = SkipModel(cfg["skip_model"])
        self.rng = np.random.default_rng(seed)
        self.profiles = profiles.to_dict("records")
        self.users: dict[int, User] = {}

        if self.full_rx:
            self.action_space = spaces.MultiDiscrete([self.N, len(self.set_bins), len(self.rep_bins), len(self.load_bins)])
        else:
            self.action_space = spaces.Discrete(self.N)
        self.obs_dim = len(self._dummy_user().features()) + self.T + self.M * 2 + 4 + self.N
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.obs_dim,), np.float32)

    # ---------------- helpers ----------------
    def _dummy_user(self):
        return make_user(self.profiles[0], np.random.default_rng(0))

    def _pick_user(self) -> User:
        i = int(self.rng.integers(len(self.profiles)))
        if not self.dynamic or i not in self.users:
            self.users[i] = make_user(self.profiles[i], self.rng)
        return self.users[i]

    def set_user(self, user: User):
        """Use a specific user (for inference / timeline generation)."""
        self._forced_user = user

    def decode(self, action) -> dict:
        if self.full_rx:
            e, s, r, l = (int(x) for x in action)
            return {"exercise": e, "sets": int(self.set_bins[s]), "reps": int(self.rep_bins[r]),
                    "load": float(self.load_bins[l])}
        return {"exercise": int(action)}

    def working_weight_kg(self, step: dict) -> float:
        """load fraction x baseline 1RM (StrengthLevel-style ratio x bodyweight x experience)."""
        from gymrl.users import EXP_MULT
        r = self.catalog.loc[step["exercise"], "base_1rm_ratio"]
        one_rm = r * self.user.profile["weight_kg"] * EXP_MULT[self.user.experience] * (1.0 if self.user.profile["sex"] == "male" else 0.7)
        return round(float(step["load"] * one_rm) / 2.5) * 2.5

    def _step_workload(self, step: dict) -> float:
        base = self.fatigue[step["exercise"]]
        if self.full_rx:
            vol = step["sets"] * step["reps"] * step["load"] / 40.0   # ~1.0 for 4x10@1.0
            return float(base * (0.5 + vol))
        return float(base)

    def _obs(self) -> np.ndarray:
        pos = np.zeros(self.T, np.float32); pos[min(self.t, self.T - 1)] = 1
        sess_mus = self.mus[[s["exercise"] for s in self.done_steps]].sum(0) if self.done_steps else np.zeros(self.M)
        hist_mus = np.zeros(self.M, np.float32)
        for sess in self.user.history[-self.hist_len:]:
            hist_mus += self.mus[sess].sum(0)
        stats = np.array([self.cum_workload / self.T, len(self.done_steps) / self.T,
                          self.n_skips / self.T, float(self.last_skipped)], np.float32)
        used = np.zeros(self.N, np.float32)
        for s in self.proposed:
            used[s["exercise"]] = 1
        return np.concatenate([self.user.features(), pos, sess_mus / self.T, hist_mus / (self.T * self.hist_len),
                               stats, used]).astype(np.float32)

    # ---------------- gym API ----------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.user = getattr(self, "_forced_user", None) or self._pick_user()
        self.t = 0
        self.proposed, self.done_steps = [], []
        self.cum_workload = 0.0
        self.n_skips = 0
        self.last_skipped = False
        return self._obs(), {}

    def step(self, action):
        step = self.decode(action)
        step["workload"] = self._step_workload(step)
        self.proposed.append(step)
        skipped = False
        if self.skips:
            m = self.mus[step["exercise"]]
            prev = self.mus[self.proposed[-2]["exercise"]] if len(self.proposed) > 1 else np.zeros(self.M)
            recent = any(step["exercise"] in s for s in self.user.history[-2:])
            p = self.skip_model.prob(
                self.user, (self.cum_workload + step["workload"]) / (self.t + 1),
                strain=float((m * prev).sum() > 0.9), mismatch=float(1 - (m * self.user.routine).sum() / (m.sum() + 1e-8) * self.M / 2),
                recency=float(recent), position=self.t / self.T, avoid_hit=float((m * self.user.true_avoid).max()))
            skipped = bool(self.rng.random() < p)
            self.skip_model.update(self.user, skipped, m, p)
        step["skipped"] = skipped
        self.last_skipped = skipped
        if skipped:
            self.n_skips += 1
        else:
            self.done_steps.append(step)
            self.cum_workload += step["workload"]
        self.t += 1
        terminated = self.t >= self.T
        reward, info = 0.0, {"step": step}
        if terminated:
            scored = self.done_steps if self.skips else self.proposed
            reward, comps = session_reward(scored, self.T, self.user, self.mus, self.fatigue,
                                           self.cfg["reward"], self.full_rx, self.skips)
            info.update(components=comps, session=list(self.proposed))
            self.user.history.append([s["exercise"] for s in scored])
            self.user.history = self.user.history[-self.hist_len:]
            if self.dynamic:
                self.user.advance(len(self.done_steps) / self.T, self.cum_workload / self.T, self.rng)
        return self._obs(), float(reward), terminated, False, info
