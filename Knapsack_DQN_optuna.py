import math
import random
from collections import namedtuple, deque

import pandas
import plotly
import joblib
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt


from Knapsack_env import KnapsackEnv

score_history = []  # array to store reward
loss_history = [] # array to store loss

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


def choose_action(state, EPS_END, EPS_START, EPS_DECAY, policy_net, item_num):
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1

    if random.random() > eps_threshold:  # if random value > epsilon value
        with torch.no_grad():
            return policy_net(state).data.max(-1)[1].item() # neural network result
    else:
        return np.random.randint(item_num * 2) # random integer result


def learn(memory, BATCH_SIZE, policy_net, target_net, GAMMA, optimizer):
    if len(memory) < BATCH_SIZE:
        return

    batch = memory.sample(BATCH_SIZE)  # random batch sampling
    states, actions, rewards, next_states = zip(*batch)  # separate batch by element list

    # Tensor list
    states = torch.stack(states)
    actions = torch.stack(actions)
    rewards = torch.stack(rewards)
    next_states = torch.stack(next_states)

    current_q = policy_net(states).gather(1, actions) # 1 dim's actions value, size[64, 1]

    ''' DQN on policy_net'''
    # max_next_q = policy_net(next_states).max(1)[0].unsqueeze(1) # get max_next_q at poicy_net, size[64]
    # unsqueeze(): create 1 dim
    # squeeze(): remove 1 dim ex) [3, 1, 20, 128] -> [3, 20, 128]

    ''' DQN on target_net'''
    max_next_q = target_net(next_states).max(1)[0].unsqueeze(1) # get max_next_q at targety_net, size[64]


    expected_q = rewards + (GAMMA * max_next_q)  # rewards + future value

    loss = F.mse_loss(current_q, expected_q)
    loss_history.append(loss.detach().numpy())
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def train(EPISODES, memory, EPS_START, EPS_END, EPS_DECAY, item_num, GAMMA, BATCH_SIZE, TARGET_UPDATE, target_net, policy_net, optimizer):
    for e in range(1, EPISODES + 1):
        state = env.reset()
        steps = 0
        while True:
            state = torch.FloatTensor(state)  # tensorize state
            action = choose_action(state, EPS_START, EPS_END, EPS_DECAY, policy_net, item_num) # integer
            action = torch.tensor([action]) # tensorize action

            next_state, reward, done = env.step(action)
            reward = torch.tensor([reward]) # tensorize reward
            next_state = torch.FloatTensor(next_state) # tensorize next_state

            memory.push(state, action, reward, next_state)  # memory experience
            learn(memory, BATCH_SIZE, policy_net, target_net, GAMMA, optimizer)

            state = next_state
            steps += 1

            if done:
                # print("Episode:{0} step: {1} reward: {2}".format(e, steps, reward.item()))
                if e % 10 == 0:
                    score_history.append(reward.numpy())
                break

        if e % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            # print("duplicate Policy_net to Target_net")


def train_knapsack(trial):
    # get env's state & action spaces
    n_states = env.get_state_space()  # item_num
    n_actions = env.get_action_space()  # item_num * 2
    item_num = env.get_item_num()

    # using 'optuna' to optimizie hyperparameters
    cfg = {'device': "cuda" if torch.cuda.is_available() else "cpu",
           # 'node': trial.suggest_int(name='node', low=16, high=128, step=16),
           'node': 32,
           'n_epochs': 1000,
           'eps_start': trial.suggest_uniform(name="eps_start", low=0.5, high=1.0),
           'eps_end': trial.suggest_uniform(name="eps_end", low=0.0, high=0.4),
           'eps_decay': trial.suggest_uniform(name="eps_decay", low=0, high=300),
           'gamma': trial.suggest_uniform(name="gamma", low=0.6, high=1.0),
           'lr': trial.suggest_float(name='lr', low=1e-4, high=1e-3, log=True),
           'batch_size': trial.suggest_int(name='batch_size', low=64, high=256, step=64),
           'target_update': trial.suggest_int(name='target_update', low=50, high=250, step=50),
           'optimizer': optim.RMSprop,
           }

    # hyperparameters definition
    device = cfg['device']
    EPISODES = cfg['n_epochs']
    EPS_START = cfg['eps_start']
    EPS_END = cfg['eps_end']
    EPS_DECAY = cfg['eps_decay']
    GAMMA = cfg['gamma']  # discount factor
    LR = cfg['lr']  # learning rate
    BATCH_SIZE = cfg['batch_size']  # batch size
    TARGET_UPDATE = cfg['target_update'] # update cycle

    policy_net = DQN(n_states, n_actions, cfg['node']).to(device)  # policy net = main net
    target_net = DQN(n_states, n_actions, cfg['node']).to(device)  # target net
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = cfg['optimizer'](policy_net.parameters(), LR)
    memory = ReplayMemory(10000)

    train(EPISODES, memory, EPS_START, EPS_END, EPS_DECAY, item_num,
          GAMMA, BATCH_SIZE, TARGET_UPDATE, target_net, policy_net, optimizer)

    return max(score_history[-400:], key=score_history.count) # convergent reward


if __name__ == '__main__':
    # print(optuna.visualization.is_available())
    env = KnapsackEnv()
    Transition = namedtuple('Transition',
                           ('state', 'action', 'next_state', 'reward'))
    steps_done = 0

    '''main function'''
    # train_knapsack()

    '''execute optuna '''
    # study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
    # study.optimize(train_knapsack, n_trials=20)
    # joblib.dump(study, 'save/DQN_optuna_node32.pkl')

    '''optimal hyperparameter'''
    study = joblib.load('save/DQN_optuna_node32.pkl')
    optuna.visualization.plot_param_importances(study).show()  # hyperparameter importance
    optuna.visualization.plot_optimization_history(study).show()  # hyperparameter optimization step
    print('best params\n', study.best_params)
    df = study.trials_dataframe().drop(['state','datetime_start','datetime_complete'], axis=1)
    print('-------------------------------')
    print('top 5 params')
    print(df.head(5))
    print('-------------------------------')
    print('all params')
    for i in study.trials:
        print(i)
