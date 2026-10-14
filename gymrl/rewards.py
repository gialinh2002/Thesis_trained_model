"""Terminal session reward (Eq. 2-4 of the paper).

R_k~ = penalty if R_k < tau_k else R_k ;  R_total = clip(sum w_k R_k~, -1, 1)
Skip envs: R_final = (|S|/T) * R_total computed on the completed session.
"""
from __future__ import annotations

import numpy as np

from gymrl.users import GOAL_RX, User


def _in_range(x, lo, hi, slack):
    if lo <= x <= hi:
        return 1.0
    d = (lo - x) if x < lo else (x - hi)
    return float(max(0.0, 1.0 - d / slack))


def components(steps: list[dict], user: User, muscle_mat: np.ndarray, catalog_fatigue: np.ndarray,
               full_prescription: bool) -> dict[str, float]:
    if not steps:
        return {}
    ex = [s["exercise"] for s in steps]
    n = len(ex)
    mus = muscle_mat[ex]                                    # (n, M)
    out = {}
    out["uniqueness"] = len(set(ex)) / n
    prim = mus.argmax(1)
    out["intra_diversity"] = len(set(prim)) / min(n, muscle_mat.shape[1])
    if user.history:
        past = set(e for sess in user.history for e in sess)
        out["inter_diversity"] = 1.0 - len(set(ex) & past) / n
    else:
        out["inter_diversity"] = 1.0
    dist = mus.sum(0); dist = dist / (dist.sum() + 1e-8)
    out["routine_alignment"] = float(1.0 - 0.5 * np.abs(dist - user.routine).sum())
    # fatigue: penalise consecutive same-muscle work and total load beyond capacity
    consec = np.mean([float((mus[i] * mus[i - 1]).sum() > 0.9) for i in range(1, n)]) if n > 1 else 0.0
    load = sum(s.get("workload", catalog_fatigue[s["exercise"]]) for s in steps) / max(n, 1)
    over = max(0.0, load - user.true_capacity)
    out["fatigue"] = float(np.clip(1.0 - consec - 1.5 * over, 0, 1))
    if full_prescription:
        rx = GOAL_RX[user.goal]
        q = [(_in_range(s["reps"], *rx["reps"], 6) + _in_range(s["sets"], *rx["sets"], 2)
              + _in_range(s["load"], *rx["load"], 0.3)) / 3 for s in steps]
        out["prescription"] = float(np.mean(q))
        # persistence: plausible reps for the load (inverted Epley: reps = 30*(1/load - 1))
        pers = [_in_range(s["reps"], 0, max(1.0, 30 * (1 / max(s["load"], 1e-3) - 1)), 4) for s in steps]
        out["persistence"] = float(np.mean(pers))
    return out


def session_reward(steps_completed: list[dict], T: int, user: User, muscle_mat, catalog_fatigue,
                   cfg_reward: dict, full_prescription: bool, completion_scaling: bool):
    comps = components(steps_completed, user, muscle_mat, catalog_fatigue, full_prescription)
    if not comps:
        return (cfg_reward["penalty"] if not completion_scaling else 0.0), {}
    total = 0.0
    for k, v in comps.items():
        spec = cfg_reward["components"][k]
        total += spec["w"] * (cfg_reward["penalty"] if v < spec["tau"] else v)
    total = float(np.clip(total, -1, 1))
    c = len(steps_completed) / T
    final = c * total if completion_scaling else total
    return final, {**comps, "R_total": total, "completion": c}
