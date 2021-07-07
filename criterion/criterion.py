import torch
import torch.nn as nn
def get_criterion(classes, label_smoothing):
    if label_smoothing <= 0.:
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
    else:
        criterion = CrossEntropyLabelSmoothing(classes, label_smoothing)
        criterion = criterion.cuda()
    return criterion

class CrossEntropyLabelSmoothing(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmoothing, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1,targets.unsqueeze(1), 1)
        targets = (1- self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
