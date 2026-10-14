import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from gymrl.baselines import GreedyAgent, RandomAgent
from gymrl.config import load_config
from gymrl.data.catalog import generate_catalog
from gymrl.data.fitrec import synthetic_profiles
from gymrl.envs.gym_workout_env import GymWorkoutEnv

CFG = load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")


@pytest.mark.parametrize("mode", ["EONS", "EOS", "FPNS", "FPS"])
def test_env_episode(mode):
    cfg = {**CFG, "env": {**CFG["env"], "mode": mode}}
    env = GymWorkoutEnv(cfg, generate_catalog(217), synthetic_profiles(50), seed=0)
    assert env.action_space.shape == (() if mode.startswith("EO") else (4,))
    if mode.startswith("FP"):
        assert int(np.prod(env.action_space.nvec)) == 217 * 5 * 19 * 21  # 432,915
    o, _ = env.reset()
    assert o.shape == env.observation_space.shape
    for agent in (RandomAgent(env), GreedyAgent(env)):
        o, _ = env.reset(); rewards = []
        for t in range(env.T):
            o, r, done, _, info = env.step(agent.act(o)); rewards.append(r)
        assert done and all(x == 0 for x in rewards[:-1])          # terminal-only reward
        assert -1 <= rewards[-1] <= 1
