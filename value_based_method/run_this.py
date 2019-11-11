import time
import gym
import numpy as np

import torch

from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

from Config import config
from utils import plot
from RL_agent import DQNAgent
from dataset import RLDataset

def train(cfg, env, agent, cur_LR):
    train_dataset = RLDataset(cfg)

    loss_fun = torch.nn.MSELoss()
    record_done, done = False, False
    losses = list()

    optimizer = torch.optim.Adam(agent.eval_net.parameters(), lr=cur_LR)
    
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
    
    if cfg.double_net:
        agent.target_net.load_state_dict(agent.eval_net.state_dict())     # update the net
        
       
    for idx, data in enumerate(train_loader):
        b_s, b_a, b_r, b_s_ = data
        b_s = Variable(b_s, requires_grad=True)
        b_s_ = Variable(b_s_, requires_grad=True)
        
        if cfg.cuda:
            b_s = b_s.cuda()
            b_s_ = b_s_.cuda()
            b_r = b_r.cuda()
            b_a = b_a.cuda()
         
        q_predict = agent.eval_net(b_s).gather(1, b_a)

           
        if not cfg.double_net:
            q_target = agent.eval_net(b_s_).detach()
        else:
            q_target = agent.target_net(b_s_).detach()
        q_target = (b_r + cfg.gamma * q_target.max(1)[0].view(cfg.batch_size, 1))

        optimizer.zero_grad()
        loss = loss_fun(q_predict, q_target)
        if cfg.cuda:
            losses.append(loss.cpu().detach().numpy())
        else:
            losses.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
    # refersh epsilon 
    agent.refresh_epsilon()
    return agent, np.mean(losses)

def test(cfg, env, agent):
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


def main(cfg):
    
    score_list, loss_list = list(), list()
    
    env = gym.make('CartPole-v1')
    agent = DQNAgent(cfg)

    optimizer = torch.optim.Adam(agent.eval_net.parameters(), lr=cfg.learning_rate)
    scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.lr_gamma)

    for e in tqdm(range(cfg.episodes)):
        cur_LR = optimizer.state_dict()['param_groups'][0]['lr']
        agnet,loss = train(cfg, env, agent, cur_LR)
        score = test(cfg, env, agent)
        score_list.append(score)
        loss_list.append(loss)
        scheduler.step()
        print(score,loss,cur_LR)
    return score_list, loss_list
    

if __name__ == '__main__':
    cfg = config()
    cfg.update_config(double_net=False)
    dqn_score, dqn_loss = main(cfg)
    cfg.update_config(double_net=True)
    ddqn_score, ddqn_loss = main(cfg)
    cfg.update_config(use_dueling_net=True)
    dueling_score, dueling_loss = main(cfg)
    plot(cfg.plot_frequency, \
        dqn_score_list=dqn_score,\
        dqn_loss_list=dqn_loss,\
        ddqn_score_list=ddqn_score,\
        ddqn_loss_list=ddqn_loss,\
        dueling_score_list=dueling_score,\
        dueling_loss_list=dueling_loss)

