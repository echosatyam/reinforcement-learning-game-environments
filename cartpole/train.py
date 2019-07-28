import math
import os
import random
import sys
from collections import namedtuple
from itertools import count

import cv2
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from scipy.misc import imshow
from tqdm import tqdm, trange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = 'cpu'


# class DQN(nn.Module):
#     def __init__(self, img_height, img_width):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features=img_height *
#                              img_width*3, out_features=24)
#         self.fc2 = nn.Linear(in_features=24, out_features=32)
#         self.out = nn.Linear(in_features=32, out_features=2)

#     def forward(self, t):
#         t = t.flatten(start_dim=1)
#         t = F.relu(self.fc1(t))
#         t = F.relu(self.fc2(t))
#         t = self.out(t)
#         return t
class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 2)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))



Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward')
                        )
e = Experience(2, 3, 1, 4)


class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def __len__(self):
        return len(self.memory)

    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size


class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) *\
            math.exp(-1.*current_step*self.decay)


class Agent():
    def __init__(self, strategy, num_actions, device):
        self.current_step = 0
        self.strategy = strategy
        self.num_actions = num_actions
        self.device = device

    def select_action(self, state, policy_net):
        rate = self.strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        if rate > random.random():
            return torch.tensor(random.randrange(self.num_actions)).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(state.unsqueeze(dim=0)).argmax(dim=1).to(self.device)


class CartPoleEnvManager():
    def __init__(self, device):
        self.device = device
        self.env = gym.make("CartPole-v0").unwrapped
        self.env.reset()
        self.current_screen = None
        self.done = False

    def reset(self):
        self.env.reset()
        self.current_screen = None

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render(mode)

    def num_actions_available(self):
        return self.env.action_space.n

    def take_action(self, action):
        _, reward, self.done, _ = self.env.step(action.item())
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None

    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s1-s2

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        screen = self.render('rgb_array').transpose((2, 0, 1))
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)

    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        top = int(screen_height*.4)
        bottom = int(screen_height*.8)
        screen = screen[:, top:bottom, :]
        return screen

    def transform_screen_data(self, screen):
        screen = np.ascontiguousarray(screen, dtype=np.float32)/255
        screen = torch.from_numpy(screen)
        resize = T.Compose([
            T.ToPILImage(),
            T.Resize((40, 90)),
            T.ToTensor()
        ])
        return resize(screen).unsqueeze(0).to(self.device)


def plot(values, moving_avg_period):
    plt.figure(2)
    plt.clf()
    plt.title('training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)
    plt.plot(get_moving_average(moving_avg_period, values))
    plt.pause(.001)
    # plt.waitforbuttonpress()


def get_moving_average(period, values):
    values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1)\
            .mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        return torch.zeros(len(values)).numpy()


def test():
    em = CartPoleEnvManager(device)
    em.reset()
    screen = em.render('rgb_array')
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    cv2.imshow("Not- processed", screen)
    screen = em.get_processed_screen().squeeze(0).permute(1, 2, 0).numpy()
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    cv2.imshow("processed", screen)
    for i in range(2):
        em.take_action(torch.tensor([1]))
        screen = em.get_state().squeeze(0).permute(1, 2, 0).numpy()
    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
    cv2.imshow("processed2", screen)
    plot(np.random.rand(300)*10, 100)
    em.close()
    if cv2.waitKey(0):
        cv2.destroyAllWindows()


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    experiences = memory.sample(BATCH_SIZE)

    batch = Experience(*zip(*experiences))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device,
                                  dtype=torch.uint8)
    non_final_next_states = torch.stack([s for s in batch.next_state
                                         if s is not None])
    # print(batch.state.shape)
    state_batch = torch.stack(batch.state)

    action_batch = torch.stack(
        [torch.tensor([s], device=device) for s in batch.action])
    reward_batch = torch.stack(batch.reward)
    # print(state_batch.shape)
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    state_action_values = state_action_values.unsqueeze(1)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(
        non_final_next_states).max(1)[0].detach()

    next_state_values = next_state_values.unsqueeze(1)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values,
                            expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = .01
TARGET_UPDATE = 10
num_episodes = 5000

cartpole = CartPoleEnvManager(device)
cartpole.reset()
screen_height = cartpole.get_screen_height()
screen_width = cartpole.get_screen_width()
cartpole.close()
policy_net = DQN(screen_height, screen_width).to(device)
target_net = DQN(screen_height, screen_width).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

strategy = EpsilonGreedyStrategy(EPS_START, EPS_END, EPS_DECAY)
agent = Agent(strategy, cartpole.num_actions_available(), device)


def train(episodes):
    num_episodes = episodes
    episode_durations = []
    trainer = trange(num_episodes, desc='Training', leave=True)
    for i_episode in trainer:
        total_reward = 0
        cartpole.reset()
        state = cartpole.get_state().squeeze(dim=0)
        for t in count():
            action = agent.select_action(state, policy_net)
            reward = cartpole.take_action(action)
            next_state = cartpole.get_state().squeeze(dim=0)
            memory.push((state, action, next_state, reward))
            state = next_state
            # print(state.shape)
            optimize_model()
            if cartpole.done:
                trainer.set_description(f"episode: {i_episode}: count: {t+1}")
                episode_durations.append(t + 1)
                plot(episode_durations, 100)
                break
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
    torch.save(policy_net.state_dict(),'./cartpole/model.pt')
    cartpole.close()

try:
    print("Keyboard Interrupt to exit")
    train(num_episodes)
except KeyboardInterrupt:
    cartpole.close()
    # exit()
