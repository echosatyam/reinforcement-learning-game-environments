import math
import os
import random
from collections import namedtuple
from itertools import count
import sys
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
import cv2
from scipy.misc import imshow
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = 'cpu'


class DQN(nn.Module):
    def __init__(self, img_height, img_width):
        super().__init__()
        self.fc1 = nn.Linear(in_features=img_height *
                             img_width*3, out_feature=24)
        self.fc2 = nn.Linear(in_features=24, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=2)

    def forward(self, t):
        t = t.flatten(start_dim=1)
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


Experience = namedtuple('Experience',
                        ('state', 'action', 'next_state', 'reward')
                        )
e = Experience(2, 3, 1, 4)


class ReplayMemory():
    def __init(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0

    def push(self, experience):
        if len(self.memory < self.capacity):
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
            return random.randrange(self.num_actions).to(self.device)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim=1).item().to(self.device)


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
        _, reward, self.done._ = self.env.step(action.item())
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
        return screen.shape[1]

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


em = CartPoleEnvManager(device)
em.reset()

screen = em.render('rgb_array')
screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
cv2.imshow("Not- processed", screen)
screen = em.get_processed_screen().squeeze(0).permute(1, 2, 0).numpy()
screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
cv2.imshow("processed", screen)
em.close()
if cv2.waitKey(0):
    cv2.destroyAllWindows()
