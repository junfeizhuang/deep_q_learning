import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,n_state, n_action):
        super(Net, self).__init__()
        self.L1 = nn.Linear(n_state,24)
        self.L2 = nn.Linear(24,24)
        self.L3 = nn.Linear(24,n_action)
        self.L1.weight.data.normal_(0, 0.1)
        self.L1.bias.data.zero_()
        self.L2.weight.data.normal_(0, 0.1)
        self.L2.bias.data.zero_()
        self.L3.weight.data.normal_(0, 0.1)
        self.L3.bias.data.zero_()

    def forward(self,x):
        x = F.relu(self.L1(x))
        x = F.relu(self.L2(x))
        x = self.L3(x)
        return x