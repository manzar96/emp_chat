import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from core.utils.masks import pad_mask, subsequent_mask
from core.utils.tensors import mktensor


class Bert2BertCollator(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        inputs, targets, labels = map(list, zip(*batch))
        inputs_lengths = torch.tensor(
            [len(s) for s in inputs], device=self.device)

        targets_lengths = torch.tensor(
            [len(s) for s in targets], device=self.device)

        # create attention masks
        max_length = max(inputs_lengths)
        inputs_pad_mask = pad_mask(inputs_lengths, max_length=max_length,
                       device=self.device)
        max_length = max(targets_lengths)
        targets_pad_mask = pad_mask(targets_lengths, max_length=max_length,
                       device=self.device)
        sub_m = subsequent_mask(max_length)

        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(inputs, batch_first=True, padding_value=self.pad_indx)
                .to(self.device))
        padded_targets = (
            pad_sequence(targets, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))

        return padded_inputs, inputs_lengths, inputs_pad_mask, padded_targets, \
               targets_lengths, targets_pad_mask


class GPT2Collator(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def __call__(self, batch):
        inputs, targets, labels = map(list, zip(*batch))
        inputs_lengths = torch.tensor(
            [len(s) for s in inputs], device=self.device)

        targets_lengths = torch.tensor(
            [len(s) for s in targets], device=self.device)

        # create attention masks
        max_length = max(inputs_lengths)
        inputs_pad_mask = pad_mask(inputs_lengths, max_length=max_length,
                       device=self.device)
        max_length = max(targets_lengths)
        targets_pad_mask = pad_mask(targets_lengths, max_length=max_length,
                       device=self.device)
        sub_m = subsequent_mask(max_length)

        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(inputs, batch_first=True, padding_value=self.pad_indx)
                .to(self.device))
        padded_targets = (
            pad_sequence(targets, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))

        return padded_inputs, inputs_lengths, inputs_pad_mask, padded_targets, \
               targets_lengths, targets_pad_mask