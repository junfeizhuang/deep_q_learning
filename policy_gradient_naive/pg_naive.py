import gym
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.autograd import Variable
from torch.distributions import Categorical
from random import shuffle

class Config:
    episodes = 100
    learning_rate = 0.0001
    gamma = 0.99
    mid = 64


class Net(nn.Module):
    def __init__(self,n_state,n_action,mid):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_state, mid)
        self.fc2 = nn.Linear(mid, mid)
        self.fc3 = nn.Linear(mid, n_action)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight.data, 0 ,0.2)
                nn.init.constant_(m.bias.data,0)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))

        return x


class Memory(object):
    def __init__(self):
        self.states = np.array([])
        self.actions = np.array([])
        self.rewards = np.array([])
        self.length = 0

    def record(self,r,log_prob):
        if self.length == 0:
            # self.states = np.array([s])
            self.rewards = np.array([r])
            self.log_probs = np.array([log_prob])
        else:
            # self.states = np.concatenate((self.states,np.array([s])))
            self.rewards = np.concatenate((self.rewards,np.array([r])))
            self.log_probs = np.concatenate((self.log_probs,np.array([log_prob])))
        self.length +=1

    def shuffle(self):
        indexs = list(range(self.length))
        shuffle(indexs)
        self.rewards = self.rewards[indexs]
        #print(self.rewards,self.log_probs)
        self.log_probs = self.log_probs[indexs]
        return indexs

def plot(x_list,y_list,x_label,y_label,name,env_name):
    plt.figure(figsize=(9, 5))
    plt.title(env_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_list, y_list, color='red', label=name)
    plt.legend(name)
    plt.savefig(env_name + '.jpg')
    

def main(env_name):
    cfg = Config()
    env = gym.make(env_name)
    if env_name == 'MountainCar-v0':
        setattr(cfg.__class__,'episodes',100)
        env = env.unwrapped
    else:
        setattr(cfg.__class__,'episodes',1000)
    setattr(cfg.__class__,'n_state',env.observation_space.shape[0]) 
    setattr(cfg.__class__,'n_action',env.action_space.n) 

    net = Net(cfg.n_state, cfg.n_action, cfg.mid)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)
    reward_recoder = list()
    loss_recoder = list()

    for e in tqdm(range(cfg.episodes)):
        memory = Memory()
        s = env.reset()
        r_sum = 0
        while True:
            s = torch.FloatTensor(s)
            probs = net(Variable(s)) # net output = action
            m = Categorical(probs)
            
            a = m.sample()
            log_prob = m.log_prob(a)
            s_, r, done, _ = env.step(a.detach().numpy())
            memory.record(r,log_prob)
            r_sum += r

            if done:
                break
            s = s_

        reward_recoder.append(r_sum)

        # train net
        reward_pool = [] # record dicount reward
        loss_pool = []

        
        running_add = 0
        discounted_reward=np.zeros_like(memory.rewards)
        # discount reward
        for i in reversed(range(memory.length)):
            running_add = running_add* cfg.gamma + memory.rewards[i]
            discounted_reward[i] = running_add

        # Normalize 
        discounted_reward -= np.mean(discounted_reward)
        discounted_reward /= np.std(discounted_reward)

        # gradient desent
        
        loss_list = list()
        for i in range(memory.length):
            log_prob = memory.log_probs[i]
            r = torch.from_numpy(discounted_reward)[i].float()
            loss = -log_prob * r
            loss_list.append(loss)
        optimizer.zero_grad()
        loss = torch.stack(loss_list).sum()
        loss.backward()
        optimizer.step()
        loss_recoder.append(loss.detach().numpy())

    # plot
    es = list(range(cfg.episodes))
    rs = reward_recoder
    ls = loss_recoder
    es2 = list(range(len(ls)))
    plot(es,rs, 'episodes','rewards','reward',env_name)
    # plot(es2,ls, 'episodes','losses','loss')
    

if __name__ == '__main__':
    env_name_list = ['CartPole-v0','MountainCar-v0']
    for env_name in env_name_list:
        main(env_name)

