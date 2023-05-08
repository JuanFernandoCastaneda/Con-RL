import numpy as np

class GNGU:
    def __init__(self, input_dim, num_nodes, max_age):
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.max_age = max_age
        self.nodes = np.random.rand(num_nodes, input_dim)
        self.errors = np.zeros(num_nodes)
        self.ages = np.zeros((num_nodes, num_nodes))

    def fit(self,observation):
        input_vector = observation
        distances = np.linalg.norm(self.nodes - input_vector, axis=1)
        winner_indices = np.argsort(distances)[:2]

        # Increment age of edges connected to winning nodes
        for (i, j), age in np.ndenumerate(self.ages):
            if i == j:
                continue
            if age < self.max_age:
                self.ages[i, j] += 1
                self.ages[j, i] = self.ages[i, j]
        # Add new node if necessary
        if np.sum(self.ages[winner_indices[0], :]) > self.max_age:
            new_node_index = np.argmax(self.errors)
            self.nodes[new_node_index] = input_vector
            self.errors[new_node_index] = 0
            self.ages[winner_indices[0], new_node_index] = 0
            self.ages[new_node_index, winner_indices[0]] = 0

        # Move winning nodes closer to input vector
        for i in winner_indices:
            self.nodes[i] += 0.5 * (input_vector - self.nodes[i])

        # Decrease error values of winning nodes and their neighbors
        for i in np.concatenate((winner_indices, np.where(self.ages[winner_indices, :] == 1)[1])):
            self.errors[i] += distances[i] ** 2
