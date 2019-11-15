import gym
import time
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.autograd import Variable

from Config import config
from DDPG import DDPG
from dataset import RLDataset
from utils import *

def train(env, cfg, agent):
    dataset = RLDataset(cfg)

    actor_optimizer = torch.optim.Adam(agent.actor_eval.parameters(),cfg.learning_rate)
    critic_optimizer = torch.optim.Adam(agent.critic_eval.parameters(),cfg.learning_rate)

    record_done, done = False, False
    critic_losses, actor_losses = list(), list()

    critic_loss_fun = torch.nn.MSELoss(reduction='mean')
    # record for dataset 
    while True:
        s = env.reset()
        while True:
            a = agent.choose_action(s, training = True)
            s_, r, done, _ = env.step(a*2)
            record_done = dataset.record(s,a,r,s_, done)
            if record_done or done:
                break
            s = s_
        if record_done:
            break

    # train
    train_loader = DataLoader(dataset,batch_size=cfg.batch_size, \
        shuffle=True, pin_memory=True, num_workers=1, drop_last=True)

    for idx, data in enumerate(train_loader):
        s, a, r, s_, done = data
        s = Variable(s)
        s_ = Variable(s_)
        a = Variable(a)
        r = Variable(r)
        done = Variable(done)

        Q_val = agent.critic_eval(s,a)
        a_ = agent.actor_pred(s_)
        Q_val_ = agent.critic_pred(s_, a_)
        Q_target = r + cfg.gamma * Q_val_* (1 - done)
        # print(a_.shape)
        # print(Q_target.shape, Q_val.shape, Q_val_.shape)
        # print(s.shape, a.shape, r.shape, s_.shape, done.shape)
        # time.sleep(1000)
        
        critic_loss = critic_loss_fun(Q_val, Q_target)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_losses.append(critic_loss.detach().numpy())
        critic_optimizer.step()

        # train actor_eval
        a = agent.actor_eval(s)
        actor_loss = -1*torch.mean(agent.critic_eval(s,a))
        actor_optimizer.zero_grad()
        actor_loss.backward(actor_loss)
        actor_losses.append(actor_loss.detach().numpy())
        actor_optimizer.step()
        # print(s.grad)
        # soft update actor_pred and critic_pred
        soft_update(agent.actor_pred, agent.actor_eval, cfg.tau_actor)
        soft_update(agent.critic_pred, agent.critic_eval, cfg.tau_critic)

    #agent.refresh_epsilon()

    return agent, critic_losses, actor_losses

def test(env,cfg,agent):
    state = env.reset()
    score = 0
    for i in range(cfg.max_play_iters):
        action = agent.choose_action(state, training = False)
        # print(action)
        next_state, reward, done, _ = env.step(action*2)
        state = next_state
        score += reward
        if done:
            break    
    return score

def main(name):
    cfg = config()
    env = gym.make(name).unwrapped
    setattr(cfg.__class__,'n_action',env.action_space.shape[0]) 
    setattr(cfg.__class__,'n_state',env.observation_space.shape[0]) 
    agent = DDPG(cfg)
    
    scores,critic_losses, actor_losses = list(), list(), list()

    for e in tqdm(range(cfg.max_epoches)):
        agent, critic_loss, actor_loss = train(env, cfg, agent)
        score = test(env, cfg, agent)
        
        scores.append(score)
        critic_losses.extend(critic_loss)
        actor_losses.extend(actor_loss)
        print('score: {:.3f}, critic_loss: {:.3f}, actor_loss: {:.3f}'.format(score, \
                                np.mean(critic_losses), \
                                np.mean(actor_losses)))

    plot(name, scores, critic_losses, actor_losses, cfg.plot_frequency)
        

if __name__ == '__main__':
    name = 'Pendulum-v0'
    main(name)