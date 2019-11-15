import torch
import numpy as np

from torch.utils.data.dataset import Dataset


class RLDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.memory = np.zeros((self.cfg.memory_capacity, self.cfg.n_state * 2 + self.cfg.n_action + 2)) # 4 state *2 + action + score
        self.record_counter = 0
    
    def record(self,s, a, r, s_, done):
        self.memory[self.record_counter, :] = np.hstack((s, a, r, done, s_))
        self.record_counter += 1
        if self.record_counter >= self.cfg.memory_capacity:
            return True
        else:
            return False
                
    def __getitem__(self,idx):
        memory = self.memory[idx,:]
        b_s = torch.FloatTensor(memory[:self.cfg.n_state])
        b_a = torch.FloatTensor(memory[self.cfg.n_state:self.cfg.n_state+self.cfg.n_action])
        b_r = torch.FloatTensor(memory[self.cfg.n_state+self.cfg.n_action:self.cfg.n_state+self.cfg.n_action+1])
        b_done = torch.FloatTensor(memory[self.cfg.n_state+self.cfg.n_action+1:self.cfg.n_state+self.cfg.n_action+2])
        b_s_ = torch.FloatTensor(memory[self.cfg.n_state+self.cfg.n_action+2:])
        # print(b_s, b_a, b_r, b_s_, b_done)
        # import time
        # time.sleep(10000)
        return b_s, b_a, b_r, b_s_, b_done

    def __len__(self):
        return self.cfg.memory_capacity