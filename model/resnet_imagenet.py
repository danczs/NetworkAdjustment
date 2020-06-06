import torch
import torch.nn as nn
import torch.nn.functional as F
#from opertions import *
from torch.nn.parameter import Parameter
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

        self.conv1 = Parameter(torch.Tensor(C_outs[0], C_in, self.kernel_size, self.kernel_size))
        self.conv2 = Parameter(torch.Tensor(C_outs[1], C_outs[0], self.kernel_size, self.kernel_size))
        self.bn1 = nn.BatchNorm2d(C_outs[0])
        self.bn2 = nn.BatchNorm2d(C_outs[1])
        self.drop = DropChannel(drop_type=drop_type)
        if self.reduction:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.C_in, self.C_outs[1], 1, stride=2, bias=False),
                nn.BatchNorm2d(self.C_outs[1])
            )
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_normal_(self.conv1, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.conv2, mode='fan_out', nonlinearity='relu')
        init.constant_(self.bn1.weight, 1)
        init.constant_(self.bn1.bias, 0)
        init.constant_(self.bn2.weight, 1)
        init.constant_(self.bn2.bias, 0)

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
            short = self.downsample(short)
            self.C_in = self.C_outs[1]
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

class ResNetImageNet(nn.Module):
    def __init__(self, init_channels, num_classes, depth, channel_numbers, channel_padding):
        super(ResNetImageNet, self).__init__()
        self.init_channels = init_channels
        self.num_classes = num_classes
        self.channel_padding = channel_padding
        self.depth = depth
        self.cell_num = (depth - 2) // 2
        self.c_red = [4, 8, 12] if depth == 18 else [6, 14, 26]  # resnet-18 & 34
        c_in, c_outm, c_out, c_final = self.parse_channel_config(self.init_channels, channel_numbers,
                                                                 self.channel_padding)
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.init_channels, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.init_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.cells = nn.ModuleList()

        self.drop_rates =[]
        for j in range(self.cell_num):
            if j*2 in self.c_red:
                reduction = True
            else:
                reduction = False
            cell = Cell(c_in[j], [c_outm[j], c_out[j]], reduction=reduction, channel_padding=self.channel_padding)
            self.cells += [cell]
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c_final, self.num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1.)
                init.constant_(m.bias, 0.)

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
        assert channel_padding in ['local', 'max']
        for i in range(self.cell_num):
            c_in.append(in_channel)
            c_outm.append(channel_list[2 * i])
            c_out.append(channel_list[2 * i + 1])
            if channel_padding == 'local':
                in_channel = c_out[-1]
            elif channel_padding == 'max':
                in_channel = c_out[-1] if i*2 in self.c_red else max(c_out[-1], in_channel)
        return c_in, c_outm, c_out, in_channel

    def forward(self, input):
        s = self.stem(input)
        for i, cell in enumerate(self.cells):
            s = cell(s, drop_rates=self.drop_rates[2 * i:2 * i + 2])
        out = self.global_pooling(s)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits
