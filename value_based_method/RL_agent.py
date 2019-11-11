import torch
import random
import numpy as np

from net import Net

class DQNAgent:
    def __init__(self, cfg):
        self.n_state = cfg.n_state
        self.n_action = cfg.n_action
        self.epsilon = cfg.epsilon
        self.cuda = cfg.cuda
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay = cfg.epsilon_decay
        self.eval_net = Net(self.n_state, self.n_action, cfg.n_mid_neuron, cfg.use_dueling_net)
        if not cfg.cuda:
            if cfg.double_net:
                self.target_net = Net(self.n_state, self.n_action,cfg.n_mid_neuron, cfg.use_dueling_net)
        else:
            self.eval_net.cuda()
            if cfg.double_net:
                self.target_net.cuda()


    def choose_action(self, s):
        if np.random.rand() <= self.epsilon:
           return random.randrange(self.n_action)
        self.eval_net.eval()
        if self.cuda:
            s = torch.FloatTensor(s).cuda()
            act_values = self.eval_net(s)
            return np.argmax(act_values.cpu().detach().numpy())
        else:
            s = torch.FloatTensor(s)
            act_values = self.eval_net(s)
            return np.argmax(act_values.detach().numpy())
        
        

    def refresh_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay