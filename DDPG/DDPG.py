import torch
import numpy as np

from Net import Actor, Critic
from utils import clip, hard_update

class DDPG(object):
	def __init__(self, cfg):
		self.critic_eval = Critic(cfg.n_state, cfg.n_action, cfg.mid_critic)
		self.critic_pred = Critic(cfg.n_state, cfg.n_action, cfg.mid_critic)
		self.actor_eval = Actor(cfg.n_state, cfg.n_action, cfg.mid_actor)
		self.actor_pred = Actor(cfg.n_state, cfg.n_action, cfg.mid_actor)
		hard_update(self.actor_pred, self.actor_eval)
		hard_update(self.critic_pred, self.critic_eval)
		self.noise = OUANoise()
		self.cfg = cfg
		self.epsilon = cfg.epsilon

	def choose_action(self, s, training=True):
		a = self.actor_eval(torch.FloatTensor(s))
		if training:
			a = (1 - self.epsilon)*a + self.epsilon* self.noise.sample()
			a = clip(a,self.cfg.map_action_low, self.cfg.map_action_upper)
		return a.detach().numpy()

	def refresh_epsilon(self):
		if self.epsilon > self.cfg.epsilon_min:
			self.epsilon *=  self.cfg.epsilon_decay
		
		


# copy from https://github.com/vy007vikas/PyTorch-ActorCriticRL
class OUANoise:
	def __init__(self, action_dim = 1, mu = 0, theta = 0.15, sigma = 0.2):
		self.action_dim = action_dim
		self.mu = mu
		self.theta = theta
		self.sigma = sigma
		self.reset()

	def reset(self):
		self.X = np.ones(self.action_dim) * self.mu

	def sample(self):
		dx = self.theta * (self.mu - self.X)
		dx = dx + self.sigma * np.random.randn(len(self.X))
		self.X = self.X + dx
		return torch.FloatTensor(self.X)