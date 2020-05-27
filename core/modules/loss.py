import torch.nn as nn
import torch


class SequenceCrossEntropyLoss(nn.Module):
    def __init__(self, pad_idx=0):
        super(SequenceCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, y_pred, targets):
        y_pred = y_pred.contiguous().view(-1, y_pred.size(-1))
        targets = targets.contiguous().view(-1)
        return self.criterion(y_pred, targets)


class SequenceNLLLoss(nn.Module):
    def __init__(self, pad_idx=0):
        super(SequenceNLLLoss, self).__init__()
        self.criterion = nn.NLLLoss(ignore_index=pad_idx)

    def forward(self, y_pred, targets):
        y_pred = y_pred.contiguous().view(-1, y_pred.size(-1))
        targets = targets.contiguous().view(-1)
        return self.criterion(y_pred, targets)