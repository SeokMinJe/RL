# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from random import random

import numpy
import numpy as np

# item 개수
import torch

item_num = 8

class KnapsackEnv():
    def __init__(self):
        init_state, init_action, init_reward, weights, prices, capacity = self.create_knapsack(item_num)
        self.state = init_state
        self.action = init_action
        self.reward = init_reward
        self.weights = weights
        self.prices = prices
        self.capacity = capacity
        self.done = 0 # episode not finish
        print('-------------------------------\ninit_state: ',
            init_state, '\nweights: ', weights, '\nprices: ', prices, '\ncapacity:', capacity,
              '\n-------------------------------')

    def create_knapsack(self, item_num):
        init_state = np.zeros(item_num)
        init_action = -100 # NULL
        weights = np.array([13, 10, 13, 7, 2, 5, 6, 8]) # np.random.randint(1, 30, item_num)
        prices = np.array([  8, 7,   9, 6, 4, 5, 10, 7]) # np.random.randint(1, 99, item_num)
        capacity = 30 #np.random.randint(1, 99)
        init_reward = 0
        return init_state, init_action, init_reward, weights, prices, capacity

    # print item_num
    def get_item_num(self):
        return item_num

    # get state after integer action
    def get_state(self, action):
        if action < item_num: # 0~4
            self.state[action] = 1
        else: # 5~9
            self.state[action-item_num] = 0
        return self.state

    # get random action used by testing
    def get_rand_action(self):
        action = np.random.randint(item_num*2) # 0~9
        return action

    # get state_space
    def get_state_space(self):
        return item_num

    # get action_space
    def get_action_space(self):
        return item_num * 2

    # get reward about present state
    def get_reward(self):
        weight_sum = 0
        price_sum = 0
        for i in np.where(self.state == 1):
            for j in self.weights[i]:
                weight_sum += j
            for j in self.prices[i]:
                price_sum += j

        weight_sum = numpy.asscalar(np.array([weight_sum])) # int
        price_sum = numpy.asscalar(np.array([price_sum])) # int

        if weight_sum > self.capacity:
            self.done = 1 # episode finish
            self.reward += 0
            return self.reward
        else:
            self.reward = price_sum
            return self.reward # torch.int32 type

    # get 1 step at env
    def step(self, action):
        self.state = self.get_state(action)
        self.reward = self.get_reward()
        # print(type(self.reward))
        # print((type(self.state)))
        return self.state, self.reward, self.done

    # reset env
    def reset(self):
        self.state= np.zeros(item_num)
        self.action = -100
        self.reward = 0
        self.done = 0
        return self.state

# env = KnapsackEnv()
#
# for i_episode in range(5):
#     env.reset()
#
#     for t in range(50):
#         action = env.get_rand_action()
#         print('random action select: ', action)
#         next_state, reward, done= env.step(action)
#         print(next_state, 'done:', done, '\n')
#         if done == 1:
#             print("Episode finished after {} timesteps".format(t+1))
#             break

