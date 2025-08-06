import gymnasium as gym
import numpy as np
import random
import networkx as nx

from typing import Any, SupportsFloat
from gymnasium.utils import RecordConstructorArgs

from src.envs import TokenSwapEnv


class CandidateDistanceWrapper(gym.Wrapper, RecordConstructorArgs):
    def __init__(self, env: gym.Env, candidate_num: int):
        RecordConstructorArgs.__init__(self, candidate_num=candidate_num)
        super().__init__(env)
        self.env = env
        self.node_num = env.unwrapped.node_num  # type: ignore
        self.candidate_num = candidate_num
        self.candidate = []

        self.action_space = gym.spaces.Discrete(self.candidate_num)
        self.observation_space = gym.spaces.Box(
            low=-2, high=2, shape=(self.candidate_num,), dtype=np.int32
        )

    def reset(self, **kwargs) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self.observation(
            obs["graph"],
            obs["current_map"],
            obs["final_map"],
            self.env.unwrapped.graph,  # type: ignore
        ), info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        if len(self.candidate) <= action:
            swap_action = (0, 0)
        else:
            swap_action = self.candidate[action]
        obs, reward, done, truncated, info = self.env.step(swap_action)
        return (
            self.observation(
                obs["graph"],
                obs["current_map"],
                obs["final_map"],
                self.env.unwrapped.graph,  # type: ignore
            ),
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
        networkx_graph: nx.Graph,
    ):
        print(graph)
        print(current_map)
        print(final_map)
        node_features = np.append(
            np.expand_dims(current_map, axis=1),
            np.expand_dims(final_map, axis=1),
            axis=1,
        )
        candidate_nodes = np.where(node_features[:, 0] != node_features[:, 1])[0]
        if graph.edge_links is None:
            raise ValueError("Graph does not have edge links.")
        self.candidate = []
        for i, (u, v) in enumerate(graph.edge_links):
            if u in candidate_nodes or v in candidate_nodes:
                self.candidate.append(graph.edge_links[i])
        random.shuffle(self.candidate)
        print(self.candidate)

        distance_changes = np.zeros(self.candidate_num, dtype=np.int32)
        if len(self.candidate) > self.candidate_num:
            self.candidate = self.candidate[: self.candidate_num]

        spl = dict(nx.all_pairs_shortest_path_length(networkx_graph))
        for i, (u, v) in enumerate(self.candidate):
            token_u, token_v = current_map[u], current_map[v]
            before_distance = (
                spl[u][np.where(final_map == token_u)[0][0]]
                + spl[v][np.where(final_map == token_v)[0][0]]
            )
            new_distance = (
                spl[u][np.where(final_map == token_v)[0][0]]
                + spl[v][np.where(final_map == token_u)[0][0]]
            )
            distance_changes[i] = new_distance - before_distance
        return distance_changes


if __name__ == "__main__":
    wrapped_env = CandidateDistanceWrapper(
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
