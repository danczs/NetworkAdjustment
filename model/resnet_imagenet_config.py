from sympy import *
import numpy as np
import collections
from .resnet_imagenet import ResNetImageNet
from .registry import *
from .channel_config import ChannelConfig
import logging

@register_channel_config
class ResNet_ImageNet_Config(ChannelConfig):
    def __init__(self, depth, width_multiplier, init_channels=64, channel_padding='local', classes=1000, **kwargs):
        super(ResNet_ImageNet_Config, self).__init__(depth, width_multiplier, init_channels, channel_padding, classes, **kwargs)
        assert depth in [18, 34]
        self.kernel_size = 3
        self.input_size = 224
        self.c_red = [4,8,12] if depth==18 else [6,14,26]
        c = self.init_channels
        for i in range(self.searched_layers):
            if i in self.c_red:
                c = c * 2
            self.channel_numbers['c' + str(i)] = int(c)
        self.channel_numbers_init = self.channel_numbers.copy()
        self.flops_fn = self.update_flops_fn()
        self.flops_ori = self.get_flops()

    def build_model(self):
        model = ResNetImageNet(self.init_channels, self.classes, self.depth, self.channel_numbers, channel_padding=self.channel_padding)
        model = model.cuda()
        return model
    def get_c_input_sym(self, in_channel, channel_padding):
        channel_list = list(self.channel_numbers.values())
        c_in = []
        c_in_sym = []
        c_inm_sym = []
        c_out_sym = []
        c_out = []
        cell_num = len(channel_list) // 2

        in_channel_sym = in_channel
        assert channel_padding in ['local','max']

        for i in range(cell_num):
            c_in_sym.append(in_channel_sym)
            c_in.append(in_channel)
            c_inm_sym.append(symbols('c'+str(2*i)))
            c_out_sym.append(symbols('c'+str(2*i+1)))
            c_out.append(channel_list[2*i+1])

            if channel_padding == 'local':
                in_channel = c_out[-1]
                in_channel_sym = c_out_sym[-1]
            elif channel_padding == 'max':
                in_channel_sym = c_out_sym[-1] if (c_out[-1] >= in_channel or 2*i in self.c_red ) else in_channel_sym
                in_channel = max(c_out[-1], in_channel) if 2*i not in self.c_red else c_out[-1]

        input_sym = []
        for i in range(len(c_in_sym)):
            input_sym.append(c_in_sym[i])
            input_sym.append(c_inm_sym[i])
        return input_sym, in_channel_sym

    def update_flops_fn(self):
        fm_size = self.input_size
        input_sym, final_sym = self.get_c_input_sym(self.init_channels, self.channel_padding)

        fm_size = fm_size // 2
        flops_fn = 3 * self.init_channels * 7 * 7 * fm_size * fm_size
        fm_size = (fm_size + 1) // 2
        for i in range(self.searched_layers):
            cs_s = symbols('c' + str(i))
            c_in = input_sym[i]
            if i in self.c_red:
                fm_size = int((fm_size+1) // 2)
                flops_fn += cs_s * c_in * self.kernel_size * self.kernel_size * fm_size * fm_size
                flops_fn += c_in * symbols('c' + str(i+1)) * 1 * 1 * fm_size * fm_size
            else:
                flops_fn += cs_s * c_in * self.kernel_size * self.kernel_size * fm_size * fm_size
        flops_fn += final_sym * self.classes
        self.flops_fn = flops_fn
        logging.info(flops_fn)
        return self.flops_fn
