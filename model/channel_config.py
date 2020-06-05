from sympy import *
import numpy as np
import collections
from .registry import *
import logging

class ChannelConfig(object):
    def __init__(self, depth, width_multiplier, init_channels=16, channel_padding='local', classes=1000, **kwargs):
        super(ChannelConfig, self).__init__()
        self.init_channels = round(width_multiplier * init_channels)
        self.depth = depth
        self.searched_layers = depth - 2
        self.channel_numbers = collections.OrderedDict()
        self.classes = classes
        self.channel_padding = channel_padding
        self.kwargs = kwargs
        self.channel_diff = collections.OrderedDict()
        self.fur = collections.OrderedDict()

    def build_model(self):
        pass

    def get_c_input_sym(self, inchannel, padding_type):
        pass

    def update_flops_fn(self):
        pass

    def get_flops_fn(self):
        return self.flops_fn

    def get_flops(self):
        return self.flops_fn.subs(self.channel_numbers)

    def get_channel_diff(self):
        flops_fn = self.get_flops_fn()
        for i in range(self.searched_layers):
            c_s = 'c' + str(i)
            diff_flops = diff(flops_fn, symbols(c_s))
            self.channel_diff[c_s] = diff_flops.subs(self.channel_numbers)
        return self.channel_diff

    def get_channel_numbers(self):
        return self.channel_numbers

    def get_searched_layers(self):
        return self.searched_layers

    def update_channel(self, top_index, bot_index, arch_learning_rate, arch_learning_rate_decay=0, min_num=4):
        for i, ti in enumerate(top_index):
            si = 'c' + str(ti)
            a_lr = max(arch_learning_rate - arch_learning_rate_decay * i, 0)
            self.channel_numbers[si] = round(self.channel_numbers[si] + a_lr * self.channel_numbers_init[si])
        for i, bi in enumerate(bot_index):
            si = 'c' + str(bi)
            a_lr = max(arch_learning_rate - arch_learning_rate_decay * i, 0)
            self.channel_numbers[si] = max(round(self.channel_numbers[si] - a_lr * self.channel_numbers_init[si]), min_num)
        self.update_flops_fn()


    def get_scale_flops(self, scale):
        tmp_channel_numbers = self.channel_numbers.copy()
        self.scale_channels(scale)
        self.update_flops_fn()
        scaled_flops = self.get_flops()
        self.channel_numbers = tmp_channel_numbers
        self.update_flops_fn()
        return scaled_flops

    def scale_to_ori_flops(self):
        flops = self.get_flops()
        error_rate = (flops - self.flops_ori) / flops
        bound_begin = 1.0
        bound_end = max( 1.0 - error_rate, 0.)
        k = 0
        while (abs(error_rate) > 0.01 and k < 20):
            k = k + 1
            scale = (bound_begin + bound_end) / 2.
            flops = self.get_scale_flops(scale)
            error_rate = (flops - self.flops_ori) / flops
            if scale > 1.0:
                if error_rate > 0:
                    bound_end = scale
                else:
                    bound_begin = scale
            else:
                if error_rate > 0 :
                    bound_begin = scale
                else:
                    bound_end = scale
        if k > 0:
            self.scale_channels(scale)
            self.update_flops_fn()

    def scale_channels(self, scale):
        for i in range(self.searched_layers):
            key = 'c' + str(i)
            self.channel_numbers[key] = round(self.channel_numbers[key] * scale)

