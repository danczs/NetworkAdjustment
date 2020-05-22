import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import time
import random

class DropChannel(nn.Module):
    def __init__(self, drop_type='gaussian', esp=1e-6):
        super(DropChannel,self).__init__()
        self.drop_type = drop_type.lower()
        self.esp = esp

    def forward(self, x, p=0.):
        dshape = x.shape
        if p > self.esp:
            if self.drop_type in ['g','gaussian']:
                sigma = torch.sqrt(torch.tensor(p / (1-p) ))
                mask = torch.randn(size=(dshape[0], dshape[1], 1, 1), dtype=x.dtype, device=x.device)
                mask = mask * sigma + 1.0
                out = mask * x
            elif self.drop_type in ['b','binary','bernoulli']:
                mask = torch.rand(size=(dshape[0], dshape[1], 1, 1), dtype=x.dtype, device=x.device)
                mask = torch.ceil(mask - p)
                out = mask * x / (1.0 - p )
            else:
                print("no such drop type:{}".format(self.drop_type))
                out = x
        else:
            out = x
        return out
