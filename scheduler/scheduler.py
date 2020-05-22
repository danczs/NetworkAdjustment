import torch
def get_scheduler(optimizer, args):
    if args.sched == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    elif args.sched == 'step':
        scheduler = None
    return scheduler