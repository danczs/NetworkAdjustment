from .registry import name_to_channel_config
import logging
from .resnet_cifar_config import *
from .resnet_imagenet_config import *
import sys
from .trian_and_infer import infer

class ModelConfig(object):
    def __init__(self, arch, **kwargs):
        self.model = None
        self.arch = arch.lower()
        self.channel_padding = kwargs['channel_padding']

        channel_config_cls = name_to_channel_config(self.arch + '_config')
        self.channel_config = channel_config_cls(**kwargs)
        self.fur_criterion = None

    def build_model(self):
        if self.model:
            del self.model
        self.model = self.channel_config.build_model()
        return self.model

    def set_fur_criterion(self, criterion):
        self.fur_criterion = criterion

    def computing_fur(self, valid_loader, base_drop_rate, times, world_size=0, distributed=False, local_rank=0):
        channel_diff = self.channel_config.get_channel_diff()
        channel_numbers = self.channel_config.get_channel_numbers()
        searched_layers = self.channel_config.get_searched_layers()
        base_flops = 0
        for i in range(searched_layers):
            d_index = 'c' + str(i)
            base_flops += channel_numbers[d_index] * channel_diff[d_index]
        base_flops /= searched_layers
        fur = np.zeros(searched_layers)

        self.model.set_drop_rates([0.]*searched_layers)
        base_acc, base_loss = infer(valid_loader, self.model, self.fur_criterion, report_freq=100,
                                    world_size=world_size, distributed=distributed)
        for i in range(searched_layers):
            d_index = 'c' + str(i)
            drop_rate = base_drop_rate * base_flops / (channel_numbers[d_index] * channel_diff[d_index])
            drop_rates = [0] * searched_layers
            drop_rates[i] = float(drop_rate)
            if local_rank == 0:
                logging.info('Drop rate in each layer:')
                logging.info(drop_rates)
            self.model.set_drop_rates(drop_rates)
            for t in range(times):
                acc, loss = infer(valid_loader, self.model, self.fur_criterion, report_freq=100,
                                  world_size=world_size, distributed=distributed)
                fur[i] += (loss - base_loss) / times
        self.fur = fur
        if local_rank == 0:
            logging.info('FUR in each layer:')
            logging.info(fur)
        return fur

    def update_chanel_with_fur(self, update_number, arch_learning_rate, arch_learning_rate_decay=0, local_rank=0):
        sort_index = np.argsort(self.fur, axis=-1)
        searched_layers = self.channel_config.get_searched_layers()
        assert update_number <= searched_layers
        bot_index = sort_index[:update_number]
        top_index = sort_index[-update_number:][::-1]
        if local_rank == 0:
            logging.info('adjusted layers:')
            logging.info([bot_index, top_index])
        self.channel_config.update_channel(top_index, bot_index, arch_learning_rate, arch_learning_rate_decay)

    def scale_to_ori_flops(self):
        self.channel_config.scale_to_ori_flops()

    def list_channel_config(self, local_rank=0):
        if local_rank == 0:
            logging.info("*****channel config******")
            logging.info(self.channel_config.get_channel_numbers())
            logging.info("flops: %d" % (self.channel_config.get_flops()))

    def logging_config(self, local_rank):
        if local_rank == 0:
            self.list_channel_config()
            logging.info('parameter count: %d' % (sum([m.numel() for m in self.model.parameters()])))

    def set_channel_numbers(self, channel_numbers):
        self.channel_config.channel_numbers = channel_numbers
        self.channel_config.update_flops_fn()