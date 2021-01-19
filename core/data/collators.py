import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from core.utils.masks import pad_mask, subsequent_mask
from core.utils.tensors import mktensor
import random

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


class T5CollatorEmpChatEmo(object):
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
        emo_labels = mktensor(labels, dtype=torch.long)
        return padded_inputs, inputs_pad_mask, padded_targets,replaced_targ, \
               targets_pad_mask,emo_labels


class T5CollatorEmpChatEmoNegSampling(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def replace_pad_labels(self,mytensor,value):
        tmp = mytensor.clone()
        tmp[mytensor==0] = value
        return tmp

    def __call__(self, batch):
        inputs, targets, labels, neg = map(list, zip(*batch))
        batch_size = len(inputs)
        neg = [item for s in neg for item in s]
        input_lengths = torch.tensor(
            [len(s) for s in inputs], device=self.device)
        targets_lengths = torch.tensor(
            [len(s) for s in targets], device=self.device)
        neg_lengths = torch.tensor(
            [len(s) for s in neg], device=self.device)

        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        max_length = max(targets_lengths)
        targets_pad_mask = pad_mask(targets_lengths, max_length=max_length,
                                   device=self.device)
        max_length = max(neg_lengths)
        neg_pad_mask = pad_mask(neg_lengths, max_length=max_length,
                                    device=self.device).reshape(batch_size,-1,
                                                                max_length)

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
        padded_neg = (
            pad_sequence(neg, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        replaced_neg = self.replace_pad_labels(padded_neg, -100)
        replaced_neg = replaced_neg.reshape(batch_size,-1,max_length)
        emo_labels = mktensor(labels, dtype=torch.long)
        number_of_samples = replaced_neg.shape[1]
        # perform sampling
        sample_indexes = [random.randint(0,number_of_samples-1) for i in \
                range(batch_size)]
        list_neg = [replaced_neg[i,sample_indexes[i],:] for i in range(
            batch_size)]
        replaced_neg = torch.stack(list_neg)
        list_neg_pad_mask = [neg_pad_mask[i,sample_indexes[i],:] for i in
                             range(batch_size)]
        neg_pad_mask = torch.stack(list_neg_pad_mask)
        return padded_inputs, inputs_pad_mask, padded_targets,replaced_targ, \
               targets_pad_mask,emo_labels,replaced_neg,neg_pad_mask

class T5CollatorEmpChatEmoPosNegSampling(object):
    def __init__(self, pad_indx=0, device='cpu'):
        self.pad_indx = pad_indx
        self.device = device

    def replace_pad_labels(self,mytensor,value):
        tmp = mytensor.clone()
        tmp[mytensor==0] = value
        return tmp

    def __call__(self, batch):
        inputs, targets, labels, ans_same, ans_wrong = map(list, zip(*batch))

        batch_size = len(inputs)

        input_lengths = torch.tensor(
            [len(s) for s in inputs], device=self.device)
        targets_lengths = torch.tensor(
            [len(s) for s in targets], device=self.device)
        same_lengths = torch.tensor(
            [len(s) for s in ans_same], device=self.device)
        wrong_lengths = torch.tensor(
            [len(s) for s in ans_wrong], device=self.device)

        # attention mask
        max_length = max(input_lengths)
        inputs_pad_mask = pad_mask(input_lengths, max_length=max_length,
                                   device=self.device)
        max_length = max(targets_lengths)
        targets_pad_mask = pad_mask(targets_lengths, max_length=max_length,
                                   device=self.device)
        max_length = max(same_lengths)
        same_pad_mask = pad_mask(same_lengths, max_length=max_length,
                                    device=self.device)
        max_length = max(wrong_lengths)
        wrong_pad_mask = pad_mask(wrong_lengths, max_length=max_length,
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
        padded_same = (
            pad_sequence(ans_same, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        replaced_same = self.replace_pad_labels(padded_same, -100)
        padded_wrong = (
            pad_sequence(ans_wrong, batch_first=True,
                         padding_value=self.pad_indx)
                .to(self.device))
        replaced_wrong = self.replace_pad_labels(padded_wrong, -100)

        emo_labels = mktensor(labels, dtype=torch.long)
        return padded_inputs, inputs_pad_mask, padded_targets,replaced_targ, \
               targets_pad_mask,emo_labels,replaced_same,same_pad_mask, \
               replaced_wrong, wrong_pad_mask

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