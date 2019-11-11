import torch
import numpy as np

from torch.utils.data.dataset import Dataset

class RLDataset(Dataset):
    def __init__(self, cfg):
        self.memory_capacity = cfg.memory_capacity
        self.n_state = cfg.n_state
        self.reset()


    def reset(self):
        self.memory = np.zeros((self.memory_capacity, self.n_state * 2 + 2)) # 4 state *2 + action + score
        self.record_counter = 0
    
    def record(self,s, a, r, s_):
        self.memory[self.record_counter, :] = np.hstack((s, a, r, s_))
        self.record_counter += 1
        if self.record_counter >= self.memory_capacity:
            return True
        else:
            return False
                
    def __getitem__(self,idx):
        memory = self.memory[idx,:]

        b_s = torch.FloatTensor(memory[:self.n_state])
        b_a = torch.LongTensor(memory[self.n_state:self.n_state+1])
        b_r = torch.FloatTensor(memory[self.n_state+1:self.n_state+2])
        b_s_ = torch.FloatTensor(memory[-self.n_state:])
        return (b_s, b_a, b_r, b_s_)

    def __len__(self):
        return self.memory_capacity