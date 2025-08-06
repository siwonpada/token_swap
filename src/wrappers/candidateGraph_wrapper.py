import gymnasium as gym
from typing import Any, SupportsFloat
import numpy as np
from math import comb
import random

from gymnasium.utils import RecordConstructorArgs
from src.envs.tokenSwap_env import TokenSwapEnv


class CandidateGraphWrapper(gym.Wrapper, RecordConstructorArgs):
    def __init__(self, env: gym.Env, candidate_num: int):
        # Initialize the wrapper with the environment
        RecordConstructorArgs.__init__(self, candidate_num=candidate_num)
        super().__init__(env)
        self.env = env
        # Access node_num from the unwrapped environment
        self.node_num = env.unwrapped.node_num  # type: ignore
        self.max_edge_num = comb(self.node_num, 2)
        self.candidate_num = candidate_num
        self.candidate = []

        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(self.candidate_num)
        self.observation_space = gym.spaces.Dict(
            {
                "nodes": gym.spaces.Box(
                    low=0,
                    high=self.node_num - 1,
                    shape=(self.node_num, 2),
                    dtype=np.int32,
                ),
                "edges": gym.spaces.Box(
                    low=0, high=1, shape=(self.max_edge_num,), dtype=np.int32
                ),
                "edge_links": gym.spaces.Box(
                    low=0,
                    high=self.node_num - 1,
                    shape=(self.max_edge_num, 2),
                    dtype=np.int32,
                ),
            }
        )

    def reset(self, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self.observation(
            obs["graph"], obs["current_map"], obs["final_map"]
        ), info

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], SupportsFloat, bool, bool, dict[str, Any]]:
        if len(self.candidate) <= action:
            swap_action = (0, 0)  # Default action if out of range
        else:
            swap_action = self.candidate[action][0]
        obs, reward, done, truncated, info = self.env.step(swap_action)
        return (
            self.observation(obs["graph"], obs["current_map"], obs["final_map"]),
            reward,
            done,
            truncated,
            info,
        )

    def observation(
        self,
        graph: gym.spaces.GraphInstance,
        current_map: np.ndarray,
        final_map: np.ndarray,
    ) -> dict[str, Any]:
        node_features = np.append(
            np.expand_dims(current_map, axis=1),
            np.expand_dims(final_map, axis=1),
            axis=1,
        )
        candidate_nodes = np.where(node_features[:, 0] != node_features[:, 1])[0]
        if graph.edge_links is None:
            raise ValueError("Graph must have edge_links defined.")
        edge_links = np.pad(
            graph.edge_links,
            ((0, self.max_edge_num - graph.edge_links.shape[0]), (0, 0)),
            mode="constant",
            constant_values=-1,
        )
        edge = np.full(self.max_edge_num, -1, dtype=np.int32)
        self.candidate = []
        for i, (u, v) in enumerate(graph.edge_links):
            if u in candidate_nodes or v in candidate_nodes:
                self.candidate.append((graph.edge_links[i], i))
        random.shuffle(self.candidate)
        if len(self.candidate) > self.candidate_num:
            self.candidate = self.candidate[: self.candidate_num]
        for i, (_, idx) in enumerate(self.candidate):
            edge[idx] = i

        return {
            "nodes": node_features,
            "edge_links": edge_links,
            "edges": edge,
        }


if __name__ == "__main__":
    wrapped_env = CandidateGraphWrapper(
        TokenSwapEnv(node_num=4, seed=100), candidate_num=10
    )
    obs, info = wrapped_env.reset()
    print("Initial Observation:", obs)
    while True:
        action = int(input("Select candidate edge (0-9): "))
        obs, reward, done, _, _ = wrapped_env.step(action)
        print("Observation:", obs)
        print("Reward:", reward)
        print("Done:", done)
        if done:
            break
