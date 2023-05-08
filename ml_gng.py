import numpy as np
from gngu import GNGU

class MLGNG:
    def __init__(self, actions_num: int, input_dim: int, initial_nodes: int, max_age: int) -> None:
        self.layers = [GNGU(input_dim, initial_nodes, max_age, 3, 3, 0.5, 0.5) for _ in range(actions_num)]

    # Must be between 0 and len(actions_num)-1.
    def update(self, action: int, signal):
        self.layers[action].fit(signal)

    def get_policy(self, signal):
        candidates = []
        for layer in self.layers:
            distances = []
            for neuron in layer:
                distances = np.append(distances, np.linalg.norm(signal - layer.nodes[neuron]))
            candidates = np.append(candidates, np.min(distances))
        return np.argmin(candidates)
