import numpy as np

from env_port import Env


def policy_from_value_function(value_function: np.ndarray, env: Env, gamma: float) -> np.ndarray:
    policy = np.zeros((len(env.observation_space), len(env.action_space)))
    for s in env.observation_space:
        best_action = None
        best_action_value = float("-inf")
        policy[s, :] = 0
        for a in env.action_space:
            action_value = 0
            for s_ in env.observation_space:
                p, r = env.P[s][a][s_]
                action_value += p * (r + gamma * value_function[s_])
            if action_value > best_action_value:
                best_action = a
                best_action_value = action_value
        policy[s, best_action] = 1
    return policy