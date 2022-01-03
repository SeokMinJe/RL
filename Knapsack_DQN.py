from Knapsack_env import KnapsackEnv
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = KnapsackEnv()
score_history = [] # array to store reward

# hyperparameters definition
EPISODES = 3000
EPS_START = 0.5527517282492975 # 0.9
EPS_END = 0.11341763105951902 # 0.05
EPS_DECAY = 292.577488609503 # 200
GAMMA = 0.6816764048146292 # 0.8 # discount factor
LR = 0.00026828004195570424 # 0.001 # learning rate
BATCH_SIZE = 64 # 256 # batch size
TARGET_UPDATE = 200 # 50
node = 32

Transition = namedtuple('Transition',
                                    ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def memorize(self, state, action, reward, next_state):
        self.memory.append((torch.FloatTensor(state),
                            torch.tensor([action]),
                            torch.FloatTensor([reward]),
                            torch.FloatTensor([next_state])))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, inputs, outputs, node):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, node)
        self.fc2 = nn.Linear(node, node)
        self.fc3 = nn.Linear(node, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def choose_action(state):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if random.random() > eps_threshold: # if random value > epsilon value
        with torch.no_grad():
            return policy_net(state).data.max(-1)[1].item() # neural network result
    else:
        return np.random.randint(item_num * 2) # random integer result


def learn():
    if len(memory) < BATCH_SIZE:
        return

    batch = memory.sample(BATCH_SIZE) # random batch sampling
    states, actions, rewards, next_states = zip(*batch) # separate batch by element list

    # Tensor list
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    next_states = torch.stack(next_states)

    current_q = policy_net(states).gather(1, actions)# 1 dim's actions value, size[64, 1]

    ''' DQN on policy_net'''
    # max_next_q = policy_net(next_states).max(1)[0].unsqueeze(1) # get max_next_q at poicy_net, size[64]
    # unsqueeze(): create 1 dim
    # squeeze(): remove 1 dim ex) [3, 1, 20, 128] -> [3, 20, 128]

    ''' DQN on target_net'''
    max_next_q = target_net(next_states).max(1)[0].unsqueeze(1) # get max_next_q at targety_net, size[64]


    expected_q = rewards + (GAMMA * max_next_q) # rewards + future value

    loss = F.mse_loss(current_q, expected_q)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train_dqn():
    for e in range(1, EPISODES + 1):
        state = env.reset()
        steps = 0
        while True:
            state = torch.FloatTensor(state) # tensorize state
            action = choose_action(state) # integer
            action = torch.tensor([action]) # tensorize action

            next_state, reward, done = env.step(action)
            reward = torch.tensor([reward]) # tensorize reward
            next_state = torch.FloatTensor(next_state) # tensorize next_state

            memory.push(state, action, reward, next_state) # memory experience
            learn()

            state = next_state
            steps += 1

            if done:
                print("Episode:{0} step: {1} reward: {2}".format(e, steps, reward.item()))
                if e % 10 == 0:
                    score_history.append(reward.numpy())
                break

        if e % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print("Policy_net to Target_net")


if __name__ == '__main__':
    # get env's state & action spaces
    n_states = env.get_state_space()  # item_num
    n_actions = env.get_action_space()  # item_num * 2
    item_num = env.get_item_num()

    policy_net = DQN(n_states, n_actions, node).to(device) # policy net = main net
    target_net = DQN(n_states, n_actions, node).to(device) # target net
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters(), LR)
    memory = ReplayMemory(10000)
    steps_done = 0

    train_dqn()

    # np.save('DQN.npy', score_history)
    plt.figure()
    plt.plot(score_history)
    plt.xlabel('Episode')
    plt.ylabel('reward')
    plt.title('DQN')
    plt.show()
