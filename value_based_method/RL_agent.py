import torch
import random
import numpy as np

from net import Net

class DQNAgent:
    def __init__(self, cfg):
        self.n_state = cfg.n_state
        self.n_action = cfg.n_action
        self.epsilon = cfg.epsilon
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay
        self.net = Net(self.n_state, self.n_action)

    def choose_action(self, s):
        if np.random.rand() <= self.epsilon:
           return random.randrange(self.n_action)
        self.net.eval()
        s = torch.FloatTensor(s)
        act_values = self.net(s)
        return np.argmax(act_values.detach().numpy())

    def refresh_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay