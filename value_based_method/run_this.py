import time
import gym
import numpy as np

import torch

from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from Config import cfg
from utils import plot
from RL_agent import DQNAgent
from dataset import RLDataset

def train(env, agent, cur_LR):
    train_dataset = RLDataset(cfg)
    
    loss_fun = torch.nn.MSELoss()
    record_done, done = False, False
    losses = list()

    optimizer = torch.optim.Adam(agent.net.parameters(), lr=cur_LR)
    
    # record (state, action, reward, next_state) for dataset
    while True:
        state = env.reset()
        while True:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            record_done = train_dataset.record(state, action, reward, next_state)
            if done or record_done:
                break
            state = next_state
        if record_done:
            break

    # train 
    train_loader = DataLoader(train_dataset,batch_size=cfg.batch_size, \
        shuffle=True, pin_memory=True, num_workers=1, drop_last=True)
    for idx, data in enumerate(train_loader):
        b_s, b_a, b_r, b_s_ = data
        b_s = Variable(b_s, requires_grad=True)
        q_predict = agent.net(b_s).gather(1, b_a)

        #b_s_ = Variable(b_s_, requires_grad=True)
        q_target = agent.net(b_s_).detach()
        q_target = (b_r + cfg.gamma * q_target.max(1)[0].view(cfg.batch_size, 1))

        optimizer.zero_grad()
        loss = loss_fun(q_predict, q_target)
        losses.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
    # refersh epsilon 
    agent.refresh_epsilon()
    return agent, np.mean(losses)

def test(env, agent):
    state = env.reset()
    score = 0
    for i in range(cfg.max_play_iters):
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break
        score += reward
    return score


def main():
    score_list, loss_list = list(), list()
    
    env = gym.make('CartPole-v1')
    agent = DQNAgent(cfg)

    optimizer = torch.optim.Adam(agent.net.parameters(), lr=cfg.learning_rate)
    scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.lr_gamma)

    for e in tqdm(range(cfg.episodes)):
        cur_LR = optimizer.state_dict()['param_groups'][0]['lr']
        agnet,loss = train(env, agent, cur_LR)
        score = test(env, agent)
        score_list.append(score)
        loss_list.append(loss)
        scheduler.step()
        print(score,loss,cur_LR)

    plot(score_list,loss_list, cfg.plot_frequency)

if __name__ == '__main__':
    main()

