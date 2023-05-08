import gym
from gngu import GNGU
from ml_gng import MLGNG
from qlearning import  OffPolicyControl
import numpy as np

env = gym.make('MountainCar-v0')
q_learning = OffPolicyControl(env, bin=20, epsilon=0.01, alpha=0.9, gamma=1)
mlgng = MLGNG(actions_num=env.action_space.n, input_dim=2, initial_nodes=20, max_age=10)

### start
state = env.reset()[0]
state = q_learning.get_discrete_state(state)
action = q_learning.action_epsilon_greedy(state)
# State is: [x_position, velocity]
new_state, reward, done, truncated, info = q_learning.env.step(action)
for i in range(1):
    new_state_discre = q_learning.get_discrete_state(new_state)
    p = new_state_discre[0]
    v = new_state_discre[1]
    action_max = np.argmax(q_learning.Q[p][v])
    q_learning.update_value(state, action, reward, new_state_discre, action_max)
    ## if():
    mlgng.update(action, new_state)
    ## selecion policy
    a_qlearn = q_learning.action_epsilon_greedy(new_state_discre)
    ##a_gngu =
    action_selected = a_qlearn #select_action(a_qlearn,a_gngu)
    state = new_state_discre
    new_state, reward, done, truncated, info = q_learning.env.step(action_selected)
    if  done:
        break