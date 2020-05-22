import torch
import torch.nn as nn
import torch.nn.functional as F
#from opertions import *
from torch.nn.parameter import Parameter
import math
import torch.nn.init as init
from .drop import DropChannel


class Cell(nn.Module):
    def __init__(self, C_in, C_outs, reduction, drop_type='b', channel_padding='local'):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.C_in = C_in
        self.kernel_size = 3
        self.channel_padding = channel_padding
        if self.reduction:
            self.stride = 2
        else:
            self.stride = 1
        self.C_outs = C_outs

        self.conv1 = Parameter(torch.Tensor(size=(C_outs[0], C_in, self.kernel_size, self.kernel_size)))
        self.conv2 = Parameter(torch.Tensor(size=(C_outs[1], C_outs[0], self.kernel_size, self.kernel_size)))
        self.reset_parameters()
        self.bn1 = nn.BatchNorm2d(C_outs[0])
        self.bn2 = nn.BatchNorm2d(C_outs[1])
        self.drop = DropChannel(drop_type=drop_type)

    def reset_parameters(self):
        init.kaiming_uniform_(self.conv1, a=math.sqrt(5))
        init.kaiming_uniform_(self.conv2, a=math.sqrt(5))

    def forward(self, h_in, drop_rates):
        c1 = F.conv2d(h_in, self.conv1, stride=self.stride, padding=1)
        c1 = self.bn1(c1)
        c1 = F.relu(c1)
        c1 = self.drop(c1, p = drop_rates[0])

        c2 = F.conv2d(c1, self.conv2, stride=1, padding=1)
        c2 = self.bn2(c2)
        c2 = self.drop(c2, p = drop_rates[1])

        out = c2
        short = h_in
        if self.reduction:
            short = F.avg_pool2d(short, kernel_size=2, stride=2)
        if self.C_in < self.C_outs[1]:
            short = F.pad(short, (0,0,0,0, (self.C_outs[1] - self.C_in),0))
        elif self.C_in > self.C_outs[1]:
            if self.channel_padding == 'local':
                short = short[:,:self.C_outs[1]]
            elif self.channel_padding == 'max':
                out = F.pad(out,(0,0,0,0,(self.C_in - self.C_outs[1]), 0))

        out = short + out
        out = F.relu(out)
        return out

class ResNetCifar(nn.Module):
    def __init__(self, init_channels, num_classes, depth, channel_numbers, channel_padding):
        super(ResNetCifar, self).__init__()
        self.init_channels = init_channels
        self.num_classes = num_classes
        self.channel_padding = channel_padding
        self.depth = depth
        self.cell_num = (depth - 2) // 2

        c_in, c_outm, c_out, c_final = self.parse_channel_config(self.init_channels, channel_numbers, self.channel_padding)
        self.stem = nn.Sequential(
            nn.Conv2d(3,self.init_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.init_channels)
        )
        self.cells = nn.ModuleList()
        self.drop_rates = []
        cell_per_stage = self.cell_num // 3
        for j in range(self.cell_num):
            if j % cell_per_stage == 0 and j!=0:
                reduction = True
            else:
                reduction = False

            cell = Cell(c_in[j], [c_outm[j],c_out[j]], reduction=reduction, channel_padding=self.channel_padding)
            self.cells += [cell]

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_final, self.num_classes)

    def drop_layers(self):
        return self.depth - 2

    def set_drop_rates(self, drop_rates):
        self.drop_rates = drop_rates
        return self.drop_rates

    def parse_channel_config(self, in_channel, channel_numbers, channel_padding):
        channel_list = list(channel_numbers.values())
        c_in = []
        c_outm = []
        c_out = []
        assert channel_padding in ['local','max']
        for i in range(self.cell_num):
            c_in.append(in_channel)
            c_outm.append(channel_list[2*i])
            c_out.append(channel_list[2*i+1])
            if channel_padding == 'local':
                in_channel = c_out[-1]
            elif channel_padding == 'max':
                in_channel = max(c_out[-1], in_channel)
        return c_in, c_outm, c_out, in_channel

    def forward(self, input):
        s = self.stem(input)
        for i, cell in enumerate(self.cells):
            s = cell(s, drop_rates=self.drop_rates[2*i:2*i+2])
        out = self.global_pooling(s)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

