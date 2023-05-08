import numpy as np

class GNGU:
    def __init__(self, input_dim, num_nodes, max_age, node_creation_error, insertion_parameter, utility_factor, winner_adaptation, neighbour_adaptation):
        self.signal_count = 0
        self.input_dim = input_dim
        self.max_age = max_age
        self.node_creation_error = node_creation_error
        self.insertion_parameter = insertion_parameter
        self.utility_factor = utility_factor
        self.winner_adaptation = winner_adaptation
        self.neighbour_adaptation = neighbour_adaptation
        # self.nodes = np.random.rand(num_nodes, input_dim)
        self.nodes = np.full((num_nodes, input_dim), 0.0)
        self.errors = np.full(num_nodes, 0.0)
        self.utilities = np.full(num_nodes, 1.0)
        self.ages = np.full((num_nodes, num_nodes), 0)

    # signal is not discretized.
    def fit(self, signal):
        self.signal_count += 1

        # If there aren't enough nodes, add a new one with the signal and get out.
        if len(self.nodes) < 2:
            self.nodes.append(signal)
            return
        
        # Else.
        distances = [np.linalg.norm(i - signal) for i in self.nodes]
        winner_indices = np.argsort(distances)[:2]
        first_winner, second_winner = winner_indices
        # Increment age of edges connected to winning nodes
        print("ages igual a nodes", len(self.nodes) == len(self.ages))
        for connection in range(len(self.nodes)):
            if connection != first_winner and self.ages[first_winner][connection] >= 0:
                self.ages[first_winner][connection] += 1
                self.ages[connection][first_winner] += 1
    
        # Update error and utility of winner. Never going to drop error because
        # nodes len always gonna be greater or equal than 2.
        self.errors[first_winner] = self.errors[first_winner] \
            + np.power(np.linalg.norm(self.nodes[first_winner] - signal), 2)
        self.utilities[first_winner] = self.utilities[first_winner] \
            + np.power(np.linalg.norm(self.nodes[second_winner] - signal), 2) \
               - np.power(np.linalg.norm(self.nodes[first_winner] - signal), 2)

        # --------------- ADAPT -------------------

        self.nodes[first_winner] = self.nodes[first_winner] \
            + self.winner_adaptation * (signal - self.nodes[first_winner])

        # Move winning nodes closer to input vector
        for connection in range(len(self.ages[first_winner])):
            # If age is >= 0 then there exists a connection.
            if connection != first_winner and self.ages[first_winner][connection] >= 0:
                self.nodes[connection] = self.nodes[connection]\
                    + self.neighbour_adaptation * (signal - self.nodes[connection])
                
        # if self.ages[first_winner][second_winner] > 0 else the same
        self.ages[first_winner][second_winner] = 0
        self.ages[second_winner][first_winner] = 0

        # Remove edges over age limit.
        for (i, j), age in np.ndenumerate(self.ages):
            if age >= self.max_age: 
                self.ages[i, j] = -1
                self.ages[j, i] = -1
        # Remove non-connected nodes.
        for node_index in range(len(self.ages)):
            # If all my connections are non existant (-1).
            if np.sum(self.ages[node_index]) == -len(self.ages[node_index]):
                self.remove_node_and_edges(node_index)
        # Check lowest utility node.
        m = np.argmin(self.utilities)
        u = np.argmax(self.errors)
        if self.errors[u]/self.utilities[m] > self.utility_factor:
            self.remove_node_and_edges(m)

        # Check if its multiple of insertion parameter lambda.
        if self.signal_count/self.insertion_parameter == self.signal_count//self.insertion_parameter:
            f = 0
            for node in range(len(self.ages)):
                if node != u and self.ages[u][node] >= 0 and self.errors[node] > self.errors[f]:
                    f = node
            new_w = 0.5*(self.nodes[u]+self.nodes[f])
            r = len(self.ages)
            self.add_node(new_w)
            self.ages[r][u], self.ages[u][r] = 0, 0
            self.ages[r][f] = 0
            self.ages[f][r] = 0
            self.ages[u][f] = -1
            self.ages[f][u] = -1
            self.errors[u] = self.node_creation_error * self.errors[u]
            self.errors[f] = self.node_creation_error * self.errors[f]
            
        # Decrease error values of winning nodes and their neighbors
        for i in np.concatenate((winner_indices, np.where(self.ages[winner_indices, :] == 1)[1])):
            self.errors[i] += distances[i] ** 2

    def add_node(self, new_w):
        print("before", self.nodes.shape, self.errors.shape, self.utilities.shape, self.ages.shape)
        new_nodes = np.full((len(self.nodes)+1, 2), 0)
        for i in range(len(self.nodes)):
            new_nodes[i] = self.nodes[i]
        new_nodes[len(self.nodes)] = new_w
        self.nodes = new_nodes
        self.errors = np.append(self.errors, 0)
        self.utilities = np.append(self.utilities, 1)
        new_ages = np.full((len(self.ages)+1, len(self.ages)+1), 0)
        for i in range(len(self.ages)):
            for j in range(len(self.ages)):
                new_ages[i][j] = self.ages[i][j]
        self.ages = new_ages
        print("after", self.nodes.shape, self.errors.shape, self.utilities.shape, self.ages.shape)
    
    def remove_node_and_edges(self, node_index):
        self.nodes = np.delete(self.nodes, node_index, 0)
        self.errors = np.delete(self.errors, node_index, 0)
        self.utilities = np.delete(self.utilities, node_index, 0)
        # Delete both column and row.
        self.ages = np.delete(np.delete(self.ages, node_index, 0), node_index, 1)
