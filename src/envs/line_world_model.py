from typing import Dict, Tuple, TypeVar

import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class LineWorldModel:
    def __init__(self, num_states) -> None:
        self.width = num_states
        self.action_space = [
            0,  # Move left
            1,  # Move right
        ]
        self.observation_space = list(range(num_states))
        self.P = self._build_transition_probabilities()

    def _build_transition_probabilities(self) -> Dict[ObsType, Dict[ActType, Dict[ObsType, Tuple[float, float]]]]:
        # P[s][a][s'] = (probability, reward)
        P = {}
        for s in self.observation_space:
            P[s] = {}
            for a in self.action_space:
                P[s][a] = {}
                for s_ in self.observation_space:
                    if s_ == s - 1 and a == 0:  # Si on va a gauche et qu'on est pas au bord
                        P[s][a][s_] = (1.0, -1.0)  # On a 100% de chance d'aller a gauche
                    elif s_ == s + 1 and a == 1:  # Si on va a droite et qu'on est pas au bord
                        P[s][a][s_] = (1.0, -1.0)  # On a 100% de chance d'aller a droite
                    elif s_ == s:  # Si on reste sur place
                        if a == 0 and s == 0:  # Si on est au début
                            P[s][a][s_] = (1.0, -1.0)
                        elif a == 1 and s == self.width - 1:  # Si on est a la fin
                            P[s][a][s_] = (1.0, -1.0)
                        else:
                            P[s][a][s_] = (0.0, -1.0)
                    else:
                        P[s][a][s_] = (0.0, -1.0)  # Dans tous les autres cas, on a 0% de chance d'aller ailleurs
        return P

    def show(self) -> None:
        frame = np.zeros(self.width)
        frame[0] = 1
        frame[self.width - 1] = 2
        text = ""
        for i in range(frame.shape[0]):
            match frame[i]:
                case 0:
                    text += "."
                case 1:
                    text += "S"
                case 2:
                    text += "G"
        print("=" * 10)
        print(text)
        print("=" * 10)
