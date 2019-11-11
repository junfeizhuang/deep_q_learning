import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,n_state, n_action, mid, use_dueling_net):
        super(Net, self).__init__()
        self.use_dueling_net = use_dueling_net
        self.features = nn.Sequential(
                    nn.Linear(n_state,mid),
                    nn.ReLU(inplace=True),
                    nn.Linear(mid,mid),
                    nn.ReLU(inplace=True),
                    nn.Linear(mid,mid),
                    nn.ReLU(inplace=True),
                            )

        if not self.use_dueling_net:
            self.action_value = nn.Linear(mid,n_action)
        else:
            self.action_value = nn.Linear(mid,n_action)
            self.state_value =  nn.Linear(mid,1)  
        #self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Linear):
                nn.init.normal_(m.weight.data, 0 ,0.2)
                nn.init.constant_(m.bias.data,0)

    def forward(self,x):
        x = self.features(x)
        if not self.use_dueling_net:
            out = self.action_value(x)
        else:
            action_v = self.action_value(x)
            state_v = self.state_value(x)
            action_v_mean = torch.mean(action_v)
            action_v_center = action_v - action_v_mean
            out = state_v + action_v_center
        return out