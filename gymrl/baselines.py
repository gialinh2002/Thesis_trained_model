"""Baselines from the paper: Random, Greedy (one-step reward-aligned), PSO (whole-plan search)."""
from __future__ import annotations

import copy

import numpy as np

from gymrl.users import GOAL_RX


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def act(self, obs):
        return self.env.action_space.sample()


class GreedyAgent:
    """Picks the exercise that adds the most unused muscle coverage aligned with the routine,
    avoiding repeats. Uses a fixed mid prescription (no dosing intelligence) — as in the paper."""

    def __init__(self, env):
        self.env = env

    def act(self, obs):
        env = self.env
        used = {s["exercise"] for s in env.proposed}
        cover = env.mus[[s["exercise"] for s in env.proposed]].sum(0) if env.proposed else np.zeros(env.M)
        score = env.mus @ (env.user.routine - 0.3 * cover) - 0.2 * env.fatigue
        score[list(used)] = -1e9
        e = int(np.argmax(score))
        if env.full_rx:
            return np.array([e, len(env.set_bins) // 2, len(env.rep_bins) // 2, len(env.load_bins) // 2])
        return e


class PSOAgent:
    """Particle Swarm Optimisation over a full T-step plan, scored by the terminal reward on a
    cloned env (commits to the plan; cannot adapt online to skips)."""

    def __init__(self, env, particles=20, iters=15, seed=0):
        self.env, self.P, self.I = env, particles, iters
        self.rng = np.random.default_rng(seed)
        self.plan, self.k = None, 0

    def _dims(self):
        e = self.env
        return np.array(e.action_space.nvec if e.full_rx else [e.N]), e.T

    def _score(self, plan):
        sim = copy.deepcopy(self.env)
        sim.skips = False
        sim.reset(); sim.user = copy.deepcopy(self.env.user)
        r = 0.0
        for a in plan:
            _, r, *_ = sim.step(a if sim.full_rx else int(a[0]))
        return r

    def _optimise(self):
        nvec, T = self._dims()
        D = len(nvec)
        x = self.rng.random((self.P, T, D)); v = np.zeros_like(x)
        to_a = lambda p: np.minimum((p * nvec).astype(int), nvec - 1)
        pbest, pval = x.copy(), np.array([self._score(to_a(p)) for p in x])
        g = pbest[pval.argmax()].copy()
        for _ in range(self.I):
            r1, r2 = self.rng.random(x.shape), self.rng.random(x.shape)
            v = 0.7 * v + 1.5 * r1 * (pbest - x) + 1.5 * r2 * (g - x)
            x = np.clip(x + v, 0, 0.9999)
            val = np.array([self._score(to_a(p)) for p in x])
            better = val > pval
            pbest[better], pval[better] = x[better], val[better]
            g = pbest[pval.argmax()].copy()
        return to_a(g)

    def act(self, obs):
        if self.env.t == 0:
            self.plan = self._optimise()
        a = self.plan[self.env.t]
        return a if self.env.full_rx else int(a[0])
