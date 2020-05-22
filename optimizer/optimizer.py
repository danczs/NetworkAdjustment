import torch
def get_optimizer(model, args):
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'rmsproptf':
        optimizer = None

    return optimizer