import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from core.utils.masks import pad_mask, subsequent_mask
from core.utils.tensors import mktensor


class EncoderDecoderTransformerCollatorEmpChat(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def replace_pad_labels(self, mytensor, value):
        tmp = mytensor.clone()
        tmp[mytensor == 0] = value
        return tmp

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

        replaced_targ = self.replace_pad_labels(padded_targets, -100)
        return padded_inputs, inputs_pad_mask, padded_targets, replaced_targ,\
               targets_pad_mask


class EncoderDecoderTransformerCollatorPersChat(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def replace_pad_labels(self,mytensor,value):
        tmp = mytensor.clone()
        tmp[mytensor==0] = value
        return tmp

    def __call__(self, batch):
        inputs, targets = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in inputs], device=self.device)
        targets_lengths = torch.tensor(
            [len(s) for s in targets], device=self.device)
        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        max_length = max(targets_lengths)
        targets_pad_mask = pad_mask(targets_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(inputs, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        padded_targets = (
            pad_sequence(targets, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        replaced_targ = self.replace_pad_labels(padded_targets, -100)
        return padded_inputs, inputs_pad_mask, padded_targets,replaced_targ, \
               targets_pad_mask


class GPT2CollatorEmpChat(object):

    def __init__(self, endofinput_indx, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device
        # we concatenate target with input! so why it needs to be done is to
        # add a token between input and target to separate them.
        self.eoi = endofinput_indx

    def replace_pad_labels(self,mytensor,value):
        tmp = mytensor.clone()
        tmp[mytensor==0] = value
        return tmp

    def __call__(self, batch):
        inputs, targets, labels = map(list, zip(*batch))

        eoi_list = [torch.tensor([self.eoi], device=self.device) for i in
                    range(len(inputs))]
        cat_input = [torch.cat((inputs[i],eoi_list[i])) for i in range(len(
            inputs))]

        cat_all = [torch.cat((cat_input[i],targets[i])) for i in range(len(
            inputs))]

        cat_lengths = torch.tensor(
            [len(s) for s in cat_all], device=self.device)

        # attention mask
        max_length = max(cat_lengths)
        inputs_pad_mask = pad_mask(cat_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(cat_all, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        replaced_inp = self.replace_pad_labels(padded_inputs, -100)
        return padded_inputs,inputs_pad_mask,replaced_inp


class GPT2CollatorPersChat(object):
    def __init__(self, endofinput_indx, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device
        # we concatenate target with input! so why it needs to be done is to
        # add a token between input and target to separate them.
        self.eoi = endofinput_indx

    def replace_pad_labels(self,mytensor,value):
        tmp = mytensor.clone()
        tmp[mytensor==0] = value
        return tmp

    def __call__(self, batch):
        inputs, targets = map(list, zip(*batch))

        eoi_list = [torch.tensor([self.eoi], device=self.device) for i in
                    range(len(inputs))]
        cat_input = [torch.cat((inputs[i],eoi_list[i])) for i in range(len(
            inputs))]

        cat_all = [torch.cat((cat_input[i],targets[i])) for i in range(len(
            inputs))]

        cat_lengths = torch.tensor(
            [len(s) for s in cat_all], device=self.device)

        # attention mask
        max_length = max(cat_lengths)
        inputs_pad_mask = pad_mask(cat_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(cat_all, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        replaced_inp = self.replace_pad_labels(padded_inputs, -100)
        return padded_inputs,inputs_pad_mask,replaced_inp

class T5CollatorEmpChat(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def replace_pad_labels(self,mytensor,value):
        tmp = mytensor.clone()
        tmp[mytensor==0] = value
        return tmp

    def __call__(self, batch):
        inputs, targets, labels = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in inputs], device=self.device)
        targets_lengths = torch.tensor(
            [len(s) for s in targets], device=self.device)
        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        max_length = max(targets_lengths)
        targets_pad_mask = pad_mask(targets_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(inputs, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        padded_targets = (
            pad_sequence(targets, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        replaced_targ = self.replace_pad_labels(padded_targets, -100)
        return padded_inputs, inputs_pad_mask, padded_targets,replaced_targ, \
               targets_pad_mask


class T5CollatorPersChat(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def replace_pad_labels(self,mytensor,value):
        tmp = mytensor.clone()
        tmp[mytensor==0] = value
        return tmp

    def __call__(self, batch):
        inputs, targets = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in inputs], device=self.device)
        targets_lengths = torch.tensor(
            [len(s) for s in targets], device=self.device)
        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        max_length = max(targets_lengths)
        targets_pad_mask = pad_mask(targets_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(inputs, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        padded_targets = (
            pad_sequence(targets, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        replaced_targ = self.replace_pad_labels(padded_targets, -100)
        return padded_inputs, inputs_pad_mask, padded_targets,replaced_targ, \
               targets_pad_mask


class TransformerVaswaniCollatorEmpChat(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def replace_pad_labels(self,mytensor,value):
        tmp = mytensor.clone()
        tmp[mytensor==0] = value
        return tmp

    def __call__(self, batch):
        inputs, targets, labels = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in inputs], device=self.device)
        targets_lengths = torch.tensor(
            [len(s) for s in targets], device=self.device)
        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        max_length = max(targets_lengths)
        targets_pad_mask = pad_mask(targets_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(inputs, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        padded_targets = (
            pad_sequence(targets, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        replaced_targ = self.replace_pad_labels(padded_targets, -100)
        return padded_inputs, inputs_pad_mask, padded_targets,replaced_targ, \
               targets_pad_mask


class TransformerVaswaniCollatorEmpChatMultitask(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def replace_pad_labels(self,mytensor,value):
        tmp = mytensor.clone()
        tmp[mytensor==0] = value
        return tmp

    def __call__(self, batch):
        inputs, targets, emo_labels = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in inputs], device=self.device)
        targets_lengths = torch.tensor(
            [len(s) for s in targets], device=self.device)
        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        max_length = max(targets_lengths)
        targets_pad_mask = pad_mask(targets_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(inputs, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        padded_targets = (
            pad_sequence(targets, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        replaced_targ = self.replace_pad_labels(padded_targets, -100)
        emo_labels = mktensor(emo_labels, dtype=torch.long)
        return padded_inputs, inputs_pad_mask, padded_targets, replaced_targ, \
               targets_pad_mask, emo_labels


class TransformerVaswaniCollatorPersChat(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def replace_pad_labels(self,mytensor,value):
        tmp = mytensor.clone()
        tmp[mytensor==0] = value
        return tmp

    def __call__(self, batch):
        inputs, targets = map(list, zip(*batch))

        input_lengths = torch.tensor(
            [len(s) for s in inputs], device=self.device)
        targets_lengths = torch.tensor(
            [len(s) for s in targets], device=self.device)
        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        max_length = max(targets_lengths)
        targets_pad_mask = pad_mask(targets_lengths, max_length=max_length,
                                   device=self.device)
        # Pad inputs and targets
        padded_inputs = (
            pad_sequence(inputs, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        padded_targets = (
            pad_sequence(targets, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        replaced_targ = self.replace_pad_labels(padded_targets, -100)
        return padded_inputs, inputs_pad_mask, padded_targets,replaced_targ, \
               targets_pad_mask