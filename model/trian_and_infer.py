from torch.autograd import Variable
from torch import distributed as dist
from .utils import AverageMeter, accuracy
import torch.nn as nn
import logging
import torch

def train(train_queue, model, criterion, optimizer, lr, report_freq=100, world_size=0, distributed=False, local_rank=0):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)

        input = Variable(input, requires_grad=False).cuda()
        target = Variable(target, requires_grad=False).cuda(async=True)
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)

        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        prec1, prec5 = accuracy(logits, target, topk=(1,5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % report_freq == 0 and local_rank==0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    return top1.avg, top5.avg

def infer(valid_queue, model, criterion, report_freq=100, world_size=0, distributed=False, local_rank=0):
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    for step, (input, target) in enumerate(valid_queue):
        with torch.no_grad():
            input = Variable(input).cuda()
            target = Variable(target).cuda(async=True)
            logits = model(input)
            loss = criterion(logits, target)

        prec1, prec5 = accuracy(logits, target, topk=(1,5))
        n = input.size(0)
        if distributed:
            reduced_loss = reduce_tensor(loss.data, world_size)
            prec1 = reduce_tensor(prec1, world_size)
            prec5 = reduce_tensor(prec5, world_size)
        else:
            reduced_loss = loss.data
        torch.cuda.synchronize()

        objs.update(reduced_loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % report_freq == 0 and local_rank==0:
            logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    return top1.avg, objs.avg

def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt

