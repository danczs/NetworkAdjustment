import sys
import os
import argparse
import time
import logging
import torch
import torch.nn as nn
import torchvision.utils
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
import init_channel_config
from torch.autograd import Variable
from data import create_dataset_loader
from model import ModelConfig, train, infer
from scheduler import get_scheduler
from optimizer import get_optimizer
from criterion import get_criterion
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')

parser = argparse.ArgumentParser(description='Search Config')
#cuda parameters
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--local_rank', type=int, default=0)

#optim paramters
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument('--learning_rate', type=float, default=0.15, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-3, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for optimizer')
parser.add_argument('--weight_decay', type=float, default=3e-4)
parser.add_argument('--epochs', type=int,default=20)

parser.add_argument('--warmup', type=str2bool, default=False, help='if using warmup during training')
parser.add_argument('--warmup_lr', type=float, default=1e-3)
parser.add_argument('--warmup_epochs', type=int, default=3)
parser.add_argument('--opt', default='sgd',type=str,help='optimizer')
parser.add_argument('--sched', default='cosine', type=str, help='scheduler')
parser.add_argument('--drop_rate', type=float, default=0.)
parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing')
parser.add_argument('--report_freq',type=int, default=100)
parser.add_argument('--epd', type=str2bool, default=True, help='decay drop rate with epochs')
parser.add_argument('--layerd', type=str2bool, default=True, help='decay drop rate with layer index')

#dataet parameters
parser.add_argument('--train_portion', type=float, default=0.9, help='portion of training data for evaluation, only works for CIFAR-100')
parser.add_argument('--dataset', type=str, default='cifar100')
parser.add_argument('--dataset_dir', type=str, default='../data')
parser.add_argument('--con_type', type=str, default='flops')#will support other constraint in the future
parser.add_argument('--workers', type=int, default=8, help='workers for loading data')

#model parameters
parser.add_argument('--arch', type=str, default='resnet_cifar', help='network architecture')
parser.add_argument('--channel_padding', type=str, default='max', help='the channel padding type for channel number mismatch')
parser.add_argument('--depth', type=int, default=20, help='network depth for resnet')
parser.add_argument('--classes', type=int, default=100, help='classes')
parser.add_argument('--init_channels', type=int, default=16, help='the channels number in the first layer')
parser.add_argument('--init_channel_config', type=str, default=None, help='the initial channel numbers')
parser.add_argument('--eval', type=str2bool, default=False, help='evaluate the init channel numbers')
parser.add_argument('--width_multiplier', type=float, default=1.0, help='network width multiplier')
parser.add_argument('--save', type=str, default='./', help='dir for saving log')

#search parameters
parser.add_argument('--arch_learning_rate', type=float, default=0.1, help='learning rate for adjusting channel numbers')
parser.add_argument('--arch_learning_rate_decay', type=float, default=0.,
                    help='the decay of arch learning rate in adjusted layers')
parser.add_argument('--iters', type=int, default=10, help='the search iterations')
parser.add_argument('--times', type=int, default=5, help='times for evalutation to overcome the randomness of dropout')
parser.add_argument('--base_drop_rate', type=float, default=0.1, help='the used drop rate for fur evaluation')
parser.add_argument('--update_num', type=int, default=3, help='the number of adjusted layers during channel adjustment')
parser.add_argument('--update_num_decay', type=float, default=0., help='update number decay during training')

args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def main():
    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    cond_logging('args = %s', args)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE'])
    args.device = 'cdua:0'
    args.world_size = 1
    args.rank = 0
    if args.distributed:
        args.num_gpu = 1
        args.device = 'cuda:%d' % args.local_rank
        torch.cuda.set_device(args.local_rank)
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ['RANK'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://',rank=args.rank, world_size=args.world_size)
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()

    assert args.rank >= 0
    cond_logging("new args = %s", args)
    train_queue, valid_queue, test_queue = create_dataset_loader(args.dataset, args.dataset_dir, args.train_portion,
                                                                 args.batch_size, args.workers, args.distributed)
    model_config = ModelConfig(arch=args.arch,
                               depth=args.depth,
                               width_multiplier=args.width_multiplier,
                               init_channels=args.init_channels,
                               channel_padding=args.channel_padding,
                               classes=args.classes
                               )
    model_config.set_fur_criterion(get_criterion(args.classes, 0.))
    if args.init_channel_config:
        init_channel_numbers = eval('init_channel_config.%s' % args.init_channel_config)
        model_config.set_channel_numbers(init_channel_numbers)

    for i in range(args.iters):
        cond_logging('******* search iter:{} *********'.format(i))
        model = model_config.build_model()
        if args.distributed:
            if args.distributed:
                model = DDP(model, device_ids=[args.local_rank])
        cond_logging("before adjustment:")
        model_config.logging_config(args.local_rank)
        model = train_new_model(model, train_queue, valid_queue, test_queue)
        if args.eval:
            break
        model_config.computing_fur(valid_queue, args.base_drop_rate, args.times, args.world_size, args.distributed,
                                   args.local_rank)
        update_num = max(round(args.update_num - i * args.update_num_decay), 0)
        model_config.update_chanel_with_fur(update_num, args.arch_learning_rate, args.arch_learning_rate_decay, args.local_rank)
        model_config.scale_to_ori_flops()
        cond_logging("After adjustment:")
        model_config.list_channel_config(args.local_rank)
        del model
        torch.cuda.synchronize()

def train_new_model(model, train_queue, valid_queue, test_queue):
    ori_model = model.module if args.distributed else model
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    drop_layers = ori_model.drop_layers()
    criterion = get_criterion(args.classes, args.label_smoothing)

    for epoch in range(args.epochs):
        scheduler.step()
        if args.warmup and epoch < args.warmup_epochs:
            lr = args.learning_rate * epoch / args.warmup_epochs + args.warmup_lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            cond_logging('epoch %d lr %e', epoch, lr)
        else:
            lr = scheduler.get_lr()[0]
            cond_logging('epoch %d lr %e', epoch, lr)

        if args.distributed:
            train_queue.sampler.set_epoch(epoch)
        if args.epd:
            drop_rate = args.drop_rate * epoch / args.epochs
        else:
            drop_rate = args.drop_rate
        drop_rates = [drop_rate] * drop_layers
        if args.layerd:
            for i in range(drop_layers):
                drop_rates[i] = drop_rates[i] * (i + 1) / drop_layers
        ori_model.set_drop_rates(drop_rates)
        cond_logging('drop rates:')
        cond_logging(ori_model.drop_rates)

        #training
        train_acc, train_obj = train(train_queue, model, criterion, optimizer, lr, args.report_freq, args.world_size,
                                     args.distributed, args.local_rank)

        cond_logging('train acc %f', train_acc)
        #validation
        drop_rates = [0] * drop_layers
        ori_model.set_drop_rates(drop_rates)
        valid_acc, valid_obj = infer(valid_queue, model, criterion, args.report_freq, args.world_size,
                                     args.distributed, args.local_rank)
        cond_logging('valid acc %f', valid_acc)
        test_acc, test_obj = infer(test_queue, model, criterion, args.report_freq, args.world_size,
                                     args.distributed, args.local_rank)
        cond_logging('test acc %f', test_acc)
    return model

def cond_logging(*logging_args):
    if args.local_rank == 0:
        logging.info(*logging_args)




if __name__ == '__main__':
    main()

