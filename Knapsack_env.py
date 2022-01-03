import numpy as np

class KnapsackEnv():
    def __init__(self):
        init_state, init_action, init_reward, weights, prices, capacity, item_num = self.create_knapsack()
        self.state = init_state
        self.action = init_action
        self.reward = init_reward
        self.weights = weights
        self.prices = prices
        self.capacity = capacity
        self.done = 0 # initialize 0 meaning that episode isn't finish
        self.item_num = item_num
        print('-------------------------------\ninit_state: ',
            init_state, '\nweights: ', weights, '\nprices: ', prices, '\ncapacity:', capacity, '\nitem_num: ', item_num,
              '\n-------------------------------')

    def create_knapsack(self):
        weights = np.array([2, 4, 5, 7, 9]) # np.random.randint(1, 30, item_num)
        prices = np.array([3, 4, 7, 4, 12]) # np.random.randint(1, 99, item_num)
        capacity = 20 # np.random.randint(1, 99)
        item_num = weights.size
        init_state = np.zeros(item_num)
        init_action = -100  # NULL
        init_reward = 0
        return init_state, init_action, init_reward, weights, prices, capacity, item_num

    # get item_num
    def get_item_num(self):
        return self.item_num

    # get state after integer action
    def get_state(self, action):
        if action < self.item_num: # 0~4
            self.state[action] = 1
        else: # 5~9
            self.state[action-self.item_num] = 0
        return self.state

    # get random action used by testing
    def get_rand_action(self):
        action = np.random.randint(self.item_num*2) # 0~9
        return action

    # get state_space
    def get_state_space(self):
        return self.item_num

    # get action_space
    def get_action_space(self):
        return self.item_num * 2

    # get reward about present state
    def get_reward(self):
        weight_sum = 0
        price_sum = 0
        for i in np.where(self.state == 1):
            for j in self.weights[i]:
                weight_sum += j
            for j in self.prices[i]:
                price_sum += j

        if weight_sum > self.capacity:
            self.done = 1 # episode finish
            self.reward += 0
            return self.reward
        else:
            self.reward = price_sum
            return self.reward

    # get 1 step at env
    def step(self, action):
        self.state = self.get_state(action)
        self.reward = self.get_reward()
        return self.state, self.reward, self.done

    # reset env
    def reset(self):
        self.state= np.zeros(self.item_num)
        self.action = -100
        self.reward = 0
        self.done = 0
        return self.state


