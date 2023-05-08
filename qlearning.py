import numpy as np

class OffPolicyControl:

    def __init__(self, env: object, bin: int, epsilon: float, alpha: float, gamma: float):
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.env_dx = (self.env.observation_space.high - self.env.observation_space.low) / bin
        self.Q = np.ndarray(shape=(bin, bin, env.action_space.n))
        self.action = []
        self.state = []

    def get_discrete_state(self,state):
        discrete_state = np.floor((state - self.env.observation_space.low) / self.env_dx).astype(int)
        self.state.append(discrete_state)
        return discrete_state

    def action_epsilon_greedy(self, state: int):
        p = state[0]
        v = state[1]
        if np.random.rand() > self.epsilon:
            action = np.argmax(self.Q[p][v])
        else:
            action = self.env.action_space.sample()
        self.action.append(action)
        return action

    def update_value(self, state, action: int, reward: int, new_state, action_max: int):
        old_p = state[0]
        old_v = state[1]
        new_p = new_state[0]
        new_v = new_state[1]

        self.Q[old_p][old_v][action] = self.Q[old_p][old_v][action] + self.alpha * (
            reward + self.gamma * self.Q[new_p][new_v][action_max] - self.Q[old_p][old_v][action])

    def episode(self,state):
        state = self.get_discrete_state(state)
        done = False
        i=0
        while not done:
            action = self.action_epsilon_greedy(state)
            new_state, reward, done, truncated, info = self.env.step(action)
            new_state = self.get_discrete_state(new_state)
            p = new_state[0]
            v = new_state[1]
            action_max = np.argmax(self.Q[p][v])
            self.update_value(state, action, reward, new_state, action_max)
            state = new_state
            i =i+1

