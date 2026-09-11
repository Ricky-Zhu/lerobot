"""Optional language metadata must not kill asynchronous rollout workers."""

import unittest
import warnings
from functools import partial

import gymnasium as gym
import numpy as np
import torch

from lerobot.scripts.lerobot_eval import rollout


class TaskEnv(gym.Env):
    observation_space = gym.spaces.Dict(
        {"agent_pos": gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)}
    )
    action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

    def __init__(self, task_attr):
        if task_attr is not None:
            setattr(self, task_attr, "Move the block")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return {"agent_pos": np.zeros(2, dtype=np.float32)}, {}

    def step(self, action):
        return {"agent_pos": np.zeros(2, dtype=np.float32)}, 0.0, False, False, {}


def make_task_env(task_attr):
    return gym.wrappers.TimeLimit(TaskEnv(task_attr), max_episode_steps=2)


def identity(value):
    return value


class CheckingPolicy(torch.nn.Module):
    def __init__(self, expected_task):
        super().__init__()
        self.expected_task = expected_task

    def reset(self):
        pass

    def select_action(self, observation):
        assert observation["task"] == [self.expected_task] * 2
        return torch.zeros(2, 2)


class OptionalTaskAttributesTest(unittest.TestCase):
    def test_rollout_keeps_workers_alive(self):
        for vector_cls in (gym.vector.SyncVectorEnv, gym.vector.AsyncVectorEnv):
            for task_attr in (None, "task", "task_description"):
                with self.subTest(vector=vector_cls.__name__, task_attr=task_attr):
                    kwargs = {"context": "forkserver"} if vector_cls is gym.vector.AsyncVectorEnv else {}
                    env = vector_cls([partial(make_task_env, task_attr)] * 2, **kwargs)
                    try:
                        policy = CheckingPolicy("" if task_attr is None else "Move the block")
                        with warnings.catch_warnings(record=True):
                            result = rollout(env, policy, identity, identity, identity, identity)
                        self.assertEqual(tuple(result["action"].shape), (2, 2, 2))
                        # A subsequent reset also exercises the worker pipes after the rollout.
                        env.reset()
                    finally:
                        env.close()


if __name__ == "__main__":
    unittest.main()
