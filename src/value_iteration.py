import json

import numpy as np

from env_port import Env
from policy_from_value_function import policy_from_value_function


def value_iteration(env: Env, theta: float, gamma: float) -> np.ndarray:
    V = np.zeros(len(env.observation_space))
    print(f"env p: {json.dumps(env.P, indent=4)}")
    while True:
        delta = 0
        for s in env.observation_space:
            v = V[s]
            best_action_value = float("-inf")
            for a in env.action_space:
                action_value = 0
                for s_ in env.observation_space:
                    print(f"env.P[s][a]:{s}, {a}, {env.P[s][a]}")
                    p, r = env.P[s][a][s_]
                    action_value += p * (r + gamma * V[s_])
                best_action_value = max(best_action_value, action_value)
            V[s] = best_action_value
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V


def run_value_iteration_on_line_world() -> None:
    from envs.line_world import LineWorld, pretty_print_line_world

    width = 5
    gamma = 0.9
    theta = 1e-5
    env = LineWorld(width=width)
    V = value_iteration(env, theta=theta, gamma=gamma)
    print(f"Final value function: {V}")
    pretty_print_line_world(env.render())
    print(V)
    policy = policy_from_value_function(V, env, gamma=gamma)
    print(f"Policy: {policy}")


if __name__ == '__main__':
    run_value_iteration_on_line_world()
