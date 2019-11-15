import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)
# actor net input shape torch.Size([n_state])
# critic net input shape torch.Size([n_state]),torch.Size([n_action])

class Actor(nn.Module):
    def __init__(self, n_state, n_action, mid):
        super(Actor, self).__init__()
        self.fc1 =  nn.Linear(n_state,mid)
        self.fc2 =  nn.Linear(mid,mid)
        self.fc3 =  nn.Linear(mid,mid)
        self.fc4 = nn.Linear(mid,n_action)
        self.init_weights()
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                #m.weight.data.uniform_ = (-0.003,+0.003)
                nn.init.normal_(m.weight.data, 0 ,0.01)
                nn.init.constant_(m.bias.data,0)

class Critic(nn.Module):
    def __init__(self, n_state, n_action, mid):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_state, mid)
        self.fc2 = nn.Linear(mid, mid)
        self.fc3 = nn.Linear(n_action, mid)
        self.out_layer1 = nn.Linear(2*mid, mid)
        self.out_layer2 = nn.Linear(mid, 1)
        self.init_weights()

    def forward(self,s,a):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        a = F.relu(self.fc3(a))
        out = torch.cat((s,a),dim=1)
        out = F.relu(self.out_layer1(out))
        out = self.out_layer2(out)
        return out

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.out_layer1.weight.data.uniform_(*hidden_init(self.out_layer1))
        self.out_layer2.weight.data.uniform_(-3e-3, 3e-3)
    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                # m.weight.data.uniform_ = (-0.003,+0.003)
                nn.init.normal_(m.weight.data, 0 ,0.01)
                nn.init.constant_(m.bias.data,0)        