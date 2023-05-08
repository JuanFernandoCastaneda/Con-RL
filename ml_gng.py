from gngu import GNGU

class MLGNG:
    def __init__(self, actions_num: int, input_dim: int, initial_nodes: int, max_age: int) -> None:
        self.layers = [GNGU(input_dim, initial_nodes, max_age, 0.5, 0.5) for _ in range(actions_num)]

    # Must be between 0 and len(actions_num)-1.
    def update(self, action: int, signal):
        self.layers[action].fit(signal)