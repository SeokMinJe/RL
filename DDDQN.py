from Knapsack_env import KnapsackEnv
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
env = KnapsackEnv()
score_history = [] # 점수 저장용

# hyperparameters
EPISODES = 3000   # 애피소드(총 플레이할 게임 수) 반복횟수
EPS_START = 0.9  # 학습 시작시 에이전트가 무작위로 행동할 확률
# ex) 0.5면 50% 절반의 확률로 무작위 행동, 나머지 절반은 학습된 방향으로 행동
# random하게 EPisolon을 두는 이유는 Agent가 가능한 모든 행동을 경험하기 위함.
EPS_END = 0.1   # 학습 막바지에 에이전트가 무작위로 행동할 확률
# EPS_START에서 END까지 점진적으로 감소시켜줌.
# --> 초반에는 경험을 많이 쌓게 하고, 점차 학습하면서 똑똑해지니깐 학습한대로 진행하게끔
EPS_DECAY = 200  # 학습 진행시 에이전트가 무작위로 행동할 확률을 감소시키는 값
GAMMA = 0.9      # 할인계수 : 에이전트가 현재 reward를 미래 reward보다 얼마나 더 가치있게 여기는지에 대한 값.
# 일종의 할인율
LR = 0.001       # 학습률
BATCH_SIZE = 64  # 배치 크기
TARGET_UPDATE = 20

Transition = namedtuple('Transition',
                                    ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    # Buffer 생성
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(inputs, 16)
        self.fc_value = nn.Linear(16, 32)
        self.fc_adv = nn.Linear(16, 32)

        self.value = nn.Linear(32, 1)
        self.adv = nn.Linear(32, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        value = F.relu(self.fc_value(x))
        adv = F.relu(self.fc_adv(x))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=-1, keepdim=True)
        Q = value + adv - advAverage
        return Q


# gym 행동 공간에서 행동의 숫자를 얻습니다.
n_states = env.get_state_space() # item_num
n_actions = env.get_action_space() # item_num * 2

policy_net = DQN(n_states, n_actions).to(device)
target_net = DQN(n_states, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(), LR)
memory = ReplayMemory(10000)
steps_done = 0
item_num = env.get_item_num()

def choose_action(state):  # 행동 선택(담당)함수
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    # 무작위 숫자와 엡실론을 비교

    # Epsilon-Greedy
    # : 초기엔 엡실론을 높게=최대한 경험을 많이 하도록 / 엡실론을 낮게 낮춰가며 = 신경망이 선택하는 비율 상승
    if random.random() > eps_threshold:  # 무작위 값 > 앱실론값 : 학습된 신경망이 옳다고 생각하는 쪽으로,
        with torch.no_grad():
            # print(type(policy_net(state).data.max(-1)[1].numpy().item())) # torch.Tensor
            return policy_net(state).data.max(-1)[1].item()
    else:  # 무작위 값 < 앱실론값 : 무작위로 행동
        return np.random.randint(item_num * 2) # 0~9 # int


def learn():
    # 메모리에 저장된 에피소드가 batch 크기보다 작으면 그냥 학습을 거름.
    if len(memory) < BATCH_SIZE:
        return

    # 경험이 충분히 쌓일 때부터 학습 진행
    batch = memory.sample(BATCH_SIZE)  # 메모리에서 무작위로 Batch 크기만큼 가져와서 학습
    states, actions, rewards, next_states = zip(*batch)  # 기존의 batch를 element별 리스트로 분리해줄 수 있게끔

    # Tensor list
    states = torch.stack(states)
    # print(states)
    actions = torch.stack(actions)
    # print(actions)
    rewards = torch.stack(rewards)
    # print(rewards)
    next_states = torch.stack(next_states)


    # 모델의 입력으로 states를 제공, 현 상태에서 했던 행동의 가치(Q값)을 current_q로 모음
    # https://velog.io/@nawnoes/torch.gather%EB%9E%80
    # print(policy_net(states).size()) # torch.Size([64,10])
    current_q = policy_net(states).gather(1, actions)# 1 dim의 actions 값을 가져옴 [64, 1]
    # print(current_q) # actions index에 해당하는 action-value

    a = policy_net(states).data.max(-1)[1].unsqueeze(1)

    ''' DDDQN'''
    max_next_q = target_net(next_states).gather(1, a)

    # max_next_q = target_net(next_states).max(1)[0].unsqueeze(1) # 에이전트가 보는 행동의 미래 가치(max_next_q)
    # unsqueeze(): 1인 차원을 생성

    # print(max_next_q.size()) # [64]
    # detach(): gradient 전파가 되지 않는 텐서 복사 생성
    expected_q = rewards + (GAMMA * max_next_q)  # rewards(보상) + 미래가치
    # print(expected_q)

    # 행동은 expected_q를 따라가게끔, MSE_loss로 오차 계산, 역전파, 신경망 학습
    loss = F.mse_loss(current_q, expected_q) # squeeze(): 차원이 1인 차원 모두 제거 ex) [3, 1, 20, 128] -> [3, 20, 128]
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-10, 10)
    optimizer.step()


for e in range(1, EPISODES + 1):  # 50번의 플레이(EPISODE수 만큼)
    state = env.reset()  # 매 시작마다 환경 초기화
    steps = 0
    # print('episode', e, 'start')
    while True:  # episode가 끝날때까지 무한루프
        state = torch.FloatTensor(state)  # 현 상태를 Tensorize
        # print(state)

        # 에이전트의 act 함수의 입력으로 state 제공
        action = choose_action(state) # integer
        action = torch.tensor([action])
        # action = torch.from_numpy(np.array([action])) # integer -> 1d numpy -> torch.Tensor
        # action : tensor, item 함수로 에이전트가 수행한 행동의 번호 추출

        # step함수의 입력에 제공 ==> 다음 상태, reward, 종료 여부(done, Boolean Value) 출력
        next_state, reward, done = env.step(action)
        reward = torch.tensor([reward])
        # reward = torch.from_numpy(np.array([reward]))  # integer -> 1d numpy -> torch.Tensor
        next_state = torch.FloatTensor(next_state)  # torch.Tensor화

        memory.push(state, action, reward, next_state)  # 경험(에피소드) 기억

        learn()

        state = next_state
        steps += 1

        if done:
            print("Episode:{0} step: {1} reward: {2}".format(e, steps, reward.item()))
            if e % 10 == 0:
                score_history.append(reward.numpy())  # score history에 점수 저장
            break

    if e % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print("Policy_net to Target_net")

plt.figure()
plt.plot(score_history)
plt.xlabel('Episode')
plt.ylabel('reward')
plt.title('D3QN')
plt.show()