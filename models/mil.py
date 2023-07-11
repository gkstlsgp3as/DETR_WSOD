import torch
import torch.nn as nn
import torch.nn.functional as F

class MIL(nn.Module):
    def __init__(self):#, dim_in, dim_out):
        super().__init__()
        #self.mil_score0 = nn.Linear(dim_in, dim_out)
        #self.mil_score1 = nn.Linear(dim_in, dim_out)

        #self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mil_score0.weight, std=0.01)
        nn.init.constant_(self.mil_score0.bias, 0)
        nn.init.normal_(self.mil_score1.weight, std=0.01)
        nn.init.constant_(self.mil_score1.bias, 0)

    def forward(self, x): # x.shape = (100, 92)
        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        #mil_score0 = self.mil_score0(x)
        #mil_score1 = self.mil_score1(x)
        #mil_score = F.softmax(mil_score0, dim=0) * F.softmax(mil_score1, dim=1)
        mil_score = F.softmax(x, dim=2) * F.softmax(x, dim=1)
        
        return mil_score