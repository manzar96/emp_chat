import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration


class T5ConditionalGenerationDoubleHead(nn.Module):
    """
    This is T5 model with 2 heads.
    An LM head + a classification head
    """
    def __init__(self, model_version, num_classes=3, device='cpu'):
        super(T5ConditionalGenerationDoubleHead, self).__init__()
        self.num_classes = num_classes
        self.lm_model = T5ForConditionalGeneration.from_pretrained(model_version)
        self.config = self.lm_model.config

        # self.clf_layer = nn.Linear(in_features=self.lm_model.config.d_model,
        #                            out_features=num_classes)
        self.clf_enc = nn.Sequential(
            nn.Linear(in_features=self.lm_model.config.d_model,
                      out_features=300,bias=True),
            # nn.Dropout(0.1),
            # nn.BatchNorm1d(300),  # applying batch norm
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(in_features=300,
                      out_features=self.num_classes)

        )
        self.device = device

    def forward(self, *args, **kwargs):

        emo_label = kwargs['emolabel']
        kwargs.pop('emolabel', None)
        outputs = self.lm_model(**kwargs, output_hidden_states=True,
                                return_dict=True)
        lm_loss = outputs['loss']
        lm_logits = outputs['logits']
        dec_hidden_states = outputs['decoder_hidden_states']
        enc_last_hidden = outputs['encoder_last_hidden_state']
        enc_hidden_states = outputs['encoder_hidden_states']
        last_dec_hidden = dec_hidden_states[-1]
        enc_last_hidden_last_timestep = enc_last_hidden[:,-1,:]
        clf_logits = self.clf_enc(enc_last_hidden_last_timestep)
        return lm_loss, lm_logits, clf_logits


class T5ConditionalGenerationTripleHead(nn.Module):
    """
    This is T5 model with 2 heads.
    An LM head + a classification head
    """
    def __init__(self, model_version, num_classes=32, device='cpu'):
        super(T5ConditionalGenerationTripleHead, self).__init__()
        self.num_classes = num_classes
        self.lm_model = T5ForConditionalGeneration.from_pretrained(model_version)
        self.config = self.lm_model.config
        self.clf_enc = nn.Sequential(
            nn.Linear(in_features=self.lm_model.config.d_model,
                      out_features=300,bias=True),
            nn.GELU(),
            nn.Linear(in_features=300,
                      out_features=self.num_classes)
        )
        self.clf_dec = nn.Sequential(
            nn.Linear(in_features=self.lm_model.config.d_model,
                      out_features=300,bias=True),
            nn.GELU(),
            nn.Linear(in_features=300,
                      out_features=self.num_classes)
        )
        self.device = device

    def forward(self, *args, **kwargs):
        batch_size = kwargs['input_ids'].shape[0]
        emo_label = kwargs['emolabel']
        kwargs.pop('emolabel', None)
        outputs = self.lm_model(**kwargs, output_hidden_states=True,
                                return_dict=True)
        lm_loss = outputs['loss']
        lm_logits = outputs['logits']
        dec_hidden_states = outputs['decoder_hidden_states']
        enc_last_hidden = outputs['encoder_last_hidden_state']
        enc_hidden_states = outputs['encoder_hidden_states']
        dec_last_hidden = dec_hidden_states[-1]

        clf_logits_enc = self.clf_enc(enc_last_hidden)
        clf_logits_dec = self.clf_dec(dec_last_hidden)

        sequence_lengths_input = torch.ne(kwargs['input_ids'],
                                          self.config.pad_token_id).sum(-1) - 1
        sequence_lengths_targets = torch.ne(kwargs['labels'], -100).sum(-1) - 1

        clf_logits_enc = clf_logits_enc[range(batch_size),
                                       sequence_lengths_input]
        clf_logits_dec= clf_logits_dec[range(batch_size),
                                       sequence_lengths_targets]
        return lm_loss, lm_logits, clf_logits_enc, clf_logits_dec


class T5ConditionalGenerationEmotions(nn.Module):
    """
    This is T5 model for conditional generation extended with an encoder and a
    decoder
    clf head for classifying
    emotions.

    """
    def __init__(self, lm_model, num_classes=32, drop=0.2, device='cpu'):
        super(T5ConditionalGenerationEmotions, self).__init__()
        self.config = lm_model.config
        self.num_classes = num_classes
        self.lm_model = lm_model
        self.dropout = drop
        self.clf_enc_layer1 = nn.Sequential(
            nn.Linear(in_features=self.lm_model.config.d_model,
                      out_features=300,bias=True),
            nn.Dropout(self.dropout),
            nn.GELU())
        self.clf_enc_layer2 = nn.Sequential(
            nn.Linear(in_features=300,
                      out_features=self.num_classes),
            nn.Dropout(self.dropout),
            nn.GELU())

        self.clf_dec_layer1 = nn.Sequential(
            nn.Linear(in_features=self.lm_model.config.d_model,
                      out_features=300,bias=True),
            nn.Dropout(self.dropout),
            nn.GELU())
        self.clf_dec_layer2 = nn.Sequential(
            nn.Linear(in_features=300,
                      out_features=self.num_classes),
            nn.Dropout(self.dropout),
            nn.GELU())

        self.device = device

    def forward(self, *args, **kwargs):
        batch_size = kwargs['input_ids'].shape[0]
        emo_label = kwargs['emolabel']
        kwargs.pop('emolabel', None)
        outputs = self.lm_model(**kwargs, output_hidden_states=True,
                                return_dict=True)
        lm_loss = outputs['loss']
        lm_logits = outputs['logits']
        dec_hidden_states = outputs['decoder_hidden_states']
        enc_last_hidden = outputs['encoder_last_hidden_state']
        enc_hidden_states = outputs['encoder_hidden_states']
        dec_last_hidden = dec_hidden_states[-1]

        clf_enc_emo_repr = self.clf_enc_layer1(enc_last_hidden)
        clf_enc_logits_emo = self.clf_enc_layer2(clf_enc_emo_repr)

        clf_dec_emo_repr = self.clf_dec_layer1(dec_last_hidden)
        clf_dec_logits_emo = self.clf_dec_layer2(clf_dec_emo_repr)

        sequence_lengths_input = torch.ne(kwargs['input_ids'],
                                          self.config.pad_token_id).sum(-1) - 1
        sequence_lengths_targets = torch.ne(kwargs['labels'], -100).sum(-1) - 1

        clf_enc_emo_repr = clf_enc_emo_repr[range(batch_size),
                                       sequence_lengths_input]
        clf_enc_logits_emo = clf_enc_logits_emo[range(batch_size),
                                       sequence_lengths_input]

        clf_dec_emo_repr = clf_dec_emo_repr[range(batch_size),
                                       sequence_lengths_targets]
        clf_dec_logits_emo = clf_dec_logits_emo[range(batch_size),
                                       sequence_lengths_targets]
        return lm_loss, lm_logits, clf_enc_logits_emo, clf_enc_emo_repr, \
               clf_dec_logits_emo, clf_dec_emo_repr


class T5ConditionalGenerationEmotionsShared(nn.Module):
    """
    This is T5 model for conditional generation extended with an encoder and a
    decoder
    clf head for classifying
    emotions.

    """
    def __init__(self, lm_model, num_classes=32, drop=0.2, device='cpu'):
        super(T5ConditionalGenerationEmotionsShared, self).__init__()
        self.num_classes = num_classes
        self.config = lm_model.config
        self.lm_model = lm_model
        self.dropout = drop
        self.clf_layer1 = nn.Sequential(
            nn.Linear(in_features=self.lm_model.config.d_model,
                      out_features=300,bias=True),
            nn.Dropout(self.dropout),
            nn.GELU())
        self.clf_layer2 = nn.Sequential(
            nn.Linear(in_features=300,
                      out_features=self.num_classes),
            nn.Dropout(self.dropout),
            nn.Sigmoid())

        self.device = device

    def forward(self, *args, **kwargs):
        batch_size = kwargs['input_ids'].shape[0]
        emo_label = kwargs['emolabel']
        kwargs.pop('emolabel', None)
        outputs = self.lm_model(**kwargs, output_hidden_states=True,
                                return_dict=True)
        lm_loss = outputs['loss']
        lm_logits = outputs['logits']
        dec_hidden_states = outputs['decoder_hidden_states']
        enc_last_hidden = outputs['encoder_last_hidden_state']
        enc_hidden_states = outputs['encoder_hidden_states']
        dec_last_hidden = dec_hidden_states[-1]

        sequence_lengths_input = torch.ne(kwargs['input_ids'],
                                          self.config.pad_token_id).sum(-1) - 1
        sequence_lengths_targets = torch.ne(kwargs['labels'], -100).sum(-1) - 1

        enc_last_hidden = enc_last_hidden[range(batch_size),
                                       sequence_lengths_input]

        dec_last_hidden = dec_last_hidden[range(batch_size),
                                       sequence_lengths_targets]

        clf_enc_emo_repr = self.clf_layer1(enc_last_hidden)
        clf_enc_logits_emo = self.clf_layer2(clf_enc_emo_repr)

        clf_dec_emo_repr = self.clf_layer1(dec_last_hidden)
        clf_dec_logits_emo = self.clf_layer2(clf_dec_emo_repr)

        # sequence_lengths_input = torch.ne(kwargs['input_ids'],
        #                                   self.config.pad_token_id).sum(-1) - 1
        # sequence_lengths_targets = torch.ne(kwargs['labels'], -100).sum(-1) - 1

        # clf_enc_emo_repr = clf_enc_emo_repr[range(batch_size),
        #                                sequence_lengths_input]
        # clf_enc_logits_emo = clf_enc_logits_emo[range(batch_size),
        #                                sequence_lengths_input]
        #
        # clf_dec_emo_repr = clf_dec_emo_repr[range(batch_size),
        #                                sequence_lengths_targets]
        # clf_dec_logits_emo = clf_dec_logits_emo[range(batch_size),
        #                                sequence_lengths_targets]

        return lm_loss, lm_logits, clf_enc_logits_emo, clf_enc_emo_repr, \
               clf_dec_logits_emo, clf_dec_emo_repr


class T5ConditionalGenerationEmotionsNeg(nn.Module):
    """
    This is T5 model for conditional generation extended with an encoder and a
    decoder
    clf head for classifying
    emotions.

    """
    def __init__(self, lm_model, num_classes=32, drop=0.2, device='cpu'):
        super(T5ConditionalGenerationEmotionsNeg, self).__init__()
        self.config = lm_model.config
        self.num_classes = num_classes
        self.lm_model = lm_model
        self.dropout = drop
        self.clf_enc_layer1 = nn.Sequential(
            nn.Linear(in_features=self.lm_model.config.d_model,
                      out_features=300,bias=True),
            nn.Dropout(self.dropout),
            nn.GELU())
        self.clf_enc_layer2 = nn.Sequential(
            nn.Linear(in_features=300,
                      out_features=self.num_classes),
            nn.Dropout(self.dropout),
            nn.Sigmoid())

        self.clf_dec_layer1 = nn.Sequential(
            nn.Linear(in_features=self.lm_model.config.d_model,
                      out_features=300,bias=True),
            nn.Dropout(self.dropout),
            nn.GELU())
        self.clf_dec_layer2 = nn.Sequential(
            nn.Linear(in_features=300,
                      out_features=self.num_classes),
            nn.Dropout(self.dropout),
            nn.Sigmoid())

        self.device = device

    def forward(self, *args, **kwargs):
        batch_size = kwargs['input_ids'].shape[0]
        emo_label = kwargs['emolabel']
        kwargs.pop('emolabel', None)
        neg_ids = kwargs['neg_ids']
        kwargs.pop('neg_ids', None)
        outputs = self.lm_model(**kwargs, output_hidden_states=True,
                                return_dict=True)
        lm_loss = outputs['loss']
        lm_logits = outputs['logits']
        dec_hidden_states = outputs['decoder_hidden_states']
        enc_last_hidden = outputs['encoder_last_hidden_state']
        enc_hidden_states = outputs['encoder_hidden_states']
        dec_last_hidden = dec_hidden_states[-1]

        clf_enc_emo_repr = self.clf_enc_layer1(enc_last_hidden)
        clf_enc_logits_emo = self.clf_enc_layer2(clf_enc_emo_repr)

        clf_dec_emo_repr = self.clf_dec_layer1(dec_last_hidden)
        clf_dec_logits_emo = self.clf_dec_layer2(clf_dec_emo_repr)

        # receive logits from last timestep according to padding
        sequence_lengths_input = torch.ne(kwargs['input_ids'],
                                          self.config.pad_token_id).sum(-1) - 1
        sequence_lengths_targets = torch.ne(kwargs['labels'], -100).sum(-1) - 1

        clf_enc_emo_repr = clf_enc_emo_repr[range(batch_size),
                                       sequence_lengths_input]
        clf_enc_logits_emo = clf_enc_logits_emo[range(batch_size),
                                       sequence_lengths_input]

        clf_dec_emo_repr = clf_dec_emo_repr[range(batch_size),
                                       sequence_lengths_targets]
        clf_dec_logits_emo = clf_dec_logits_emo[range(batch_size),
                                       sequence_lengths_targets]
        # forward the negative sample
        outputs_neg = self.lm_model(input_ids=kwargs['input_ids'],
                                 attention_mask=kwargs['attention_mask'],
                                 labels=neg_ids,
                                 output_hidden_states=True,
                                return_dict=True)
        dec_hidden_states_neg = outputs_neg['decoder_hidden_states']
        dec_last_hidden_neg = dec_hidden_states_neg[-1]

        clf_dec_emo_repr_neg = self.clf_dec_layer1(dec_last_hidden_neg)
        clf_dec_logits_emo_neg = self.clf_dec_layer2(clf_dec_emo_repr_neg)
        # receive logits from last timestep according to padding
        sequence_lengths_targets_neg = torch.ne(neg_ids, -100).sum(-1)- 1

        clf_dec_emo_repr_neg = clf_dec_emo_repr_neg[range(batch_size),
                                       sequence_lengths_targets_neg]
        clf_dec_logits_emo_neg = clf_dec_logits_emo_neg[range(batch_size),
                                       sequence_lengths_targets_neg]

        return lm_loss, lm_logits, clf_enc_logits_emo, clf_enc_emo_repr, \
               clf_dec_logits_emo, clf_dec_emo_repr, clf_dec_logits_emo_neg, \
               clf_dec_emo_repr_neg

    def forward_validate(self, *args, **kwargs):
        batch_size = kwargs['input_ids'].shape[0]
        emo_label = kwargs['emolabel']
        kwargs.pop('emolabel', None)

        outputs = self.lm_model(**kwargs, output_hidden_states=True,
                                return_dict=True)
        lm_loss = outputs['loss']
        lm_logits = outputs['logits']
        dec_hidden_states = outputs['decoder_hidden_states']
        enc_last_hidden = outputs['encoder_last_hidden_state']
        enc_hidden_states = outputs['encoder_hidden_states']
        dec_last_hidden = dec_hidden_states[-1]

        clf_enc_emo_repr = self.clf_enc_layer1(enc_last_hidden)
        clf_enc_logits_emo = self.clf_enc_layer2(clf_enc_emo_repr)

        clf_dec_emo_repr = self.clf_dec_layer1(dec_last_hidden)
        clf_dec_logits_emo = self.clf_dec_layer2(clf_dec_emo_repr)

        # receive logits from last timestep according to padding
        sequence_lengths_input = torch.ne(kwargs['input_ids'],
                                          self.config.pad_token_id).sum(-1) - 1
        sequence_lengths_targets = torch.ne(kwargs['labels'], -100).sum(-1) - 1

        clf_enc_emo_repr = clf_enc_emo_repr[range(batch_size),
                                       sequence_lengths_input]
        clf_enc_logits_emo = clf_enc_logits_emo[range(batch_size),
                                       sequence_lengths_input]

        clf_dec_emo_repr = clf_dec_emo_repr[range(batch_size),
                                       sequence_lengths_targets]
        clf_dec_logits_emo = clf_dec_logits_emo[range(batch_size),
                                       sequence_lengths_targets]

        return lm_loss, lm_logits, clf_enc_logits_emo, clf_enc_emo_repr, \
               clf_dec_logits_emo, clf_dec_emo_repr


class T5ConditionalGenerationEmotionsSharedNeg(nn.Module):
    """
    This is T5 model for conditional generation extended with an encoder and a
    decoder
    clf head for classifying
    emotions.

    """
    def __init__(self, lm_model, num_classes=32, drop=0.2, device='cpu'):
        super(T5ConditionalGenerationEmotionsSharedNeg, self).__init__()
        self.num_classes = num_classes
        self.config = lm_model.config
        self.lm_model = lm_model
        self.dropout = drop
        self.clf_layer1 = nn.Sequential(
            nn.Linear(in_features=self.lm_model.config.d_model,
                      out_features=300,bias=True),
            nn.Dropout(self.dropout),
            nn.GELU())
        self.clf_layer2 = nn.Sequential(
            nn.Linear(in_features=300,
                      out_features=self.num_classes),
            nn.Dropout(self.dropout),
            nn.Sigmoid())

        self.device = device

    def forward(self, *args, **kwargs):
        batch_size = kwargs['input_ids'].shape[0]
        emo_label = kwargs['emolabel']
        kwargs.pop('emolabel', None)
        neg_ids = kwargs['neg_ids']
        kwargs.pop('neg_ids', None)
        outputs = self.lm_model(**kwargs, output_hidden_states=True,
                                return_dict=True)
        lm_loss = outputs['loss']
        lm_logits = outputs['logits']
        enc_last_hidden = outputs['encoder_last_hidden_state']
        enc_hidden_states = outputs['encoder_hidden_states']
        dec_hidden_states = outputs['decoder_hidden_states']
        dec_last_hidden = dec_hidden_states[-1]

        clf_enc_emo_repr = self.clf_layer1(enc_last_hidden)
        clf_enc_logits_emo = self.clf_layer2(clf_enc_emo_repr)

        clf_dec_emo_repr = self.clf_layer1(dec_last_hidden)
        clf_dec_logits_emo = self.clf_layer2(clf_dec_emo_repr)

        # receive logits from last timestep according to padding
        sequence_lengths_input = torch.ne(kwargs['input_ids'],
                                          self.config.pad_token_id).sum(-1) - 1
        sequence_lengths_targets = torch.ne(kwargs['labels'], -100).sum(-1) - 1

        clf_enc_emo_repr = clf_enc_emo_repr[range(batch_size),
                                       sequence_lengths_input]
        clf_enc_logits_emo = clf_enc_logits_emo[range(batch_size),
                                       sequence_lengths_input]

        clf_dec_emo_repr = clf_dec_emo_repr[range(batch_size),
                                       sequence_lengths_targets]
        clf_dec_logits_emo = clf_dec_logits_emo[range(batch_size),
                                       sequence_lengths_targets]

        # forward the negative sample
        outputs_neg = self.lm_model(input_ids=kwargs['input_ids'],
                                 attention_mask=kwargs['attention_mask'],
                                 labels=neg_ids,
                                 output_hidden_states=True,
                                return_dict=True)

        dec_hidden_states_neg = outputs_neg['decoder_hidden_states']
        dec_last_hidden_neg = dec_hidden_states_neg[-1]
        clf_dec_emo_repr_neg = self.clf_layer1(dec_last_hidden_neg)
        clf_dec_logits_emo_neg = self.clf_layer2(clf_dec_emo_repr_neg)

        # receive logits from last timestep according to padding
        sequence_lengths_targets_neg = torch.ne(neg_ids, -100).sum(-1)- 1
        clf_dec_emo_repr_neg = clf_dec_emo_repr_neg[range(batch_size),
                                       sequence_lengths_targets_neg]
        clf_dec_logits_emo_neg = clf_dec_logits_emo_neg[range(batch_size),
                                       sequence_lengths_targets_neg]
        return lm_loss, lm_logits, clf_enc_logits_emo, clf_enc_emo_repr, \
               clf_dec_logits_emo, clf_dec_emo_repr, clf_dec_logits_emo_neg, \
               clf_dec_emo_repr_neg

    def forward_validate(self, *args, **kwargs):
        batch_size = kwargs['input_ids'].shape[0]
        emo_label = kwargs['emolabel']
        kwargs.pop('emolabel', None)

        outputs = self.lm_model(**kwargs, output_hidden_states=True,
                                return_dict=True)
        lm_loss = outputs['loss']
        lm_logits = outputs['logits']
        dec_hidden_states = outputs['decoder_hidden_states']
        enc_last_hidden = outputs['encoder_last_hidden_state']
        enc_hidden_states = outputs['encoder_hidden_states']
        dec_last_hidden = dec_hidden_states[-1]

        clf_enc_emo_repr = self.clf_layer1(enc_last_hidden)
        clf_enc_logits_emo = self.clf_layer2(clf_enc_emo_repr)

        clf_dec_emo_repr = self.clf_layer1(dec_last_hidden)
        clf_dec_logits_emo = self.clf_layer2(clf_dec_emo_repr)

        # receive logits from last timestep according to padding
        sequence_lengths_input = torch.ne(kwargs['input_ids'],
                                          self.config.pad_token_id).sum(-1) - 1
        sequence_lengths_targets = torch.ne(kwargs['labels'], -100).sum(-1) - 1

        clf_enc_emo_repr = clf_enc_emo_repr[range(batch_size),
                                       sequence_lengths_input]
        clf_enc_logits_emo = clf_enc_logits_emo[range(batch_size),
                                       sequence_lengths_input]

        clf_dec_emo_repr = clf_dec_emo_repr[range(batch_size),
                                       sequence_lengths_targets]
        clf_dec_logits_emo = clf_dec_logits_emo[range(batch_size),
                                       sequence_lengths_targets]

        return lm_loss, lm_logits, clf_enc_logits_emo, clf_enc_emo_repr, \
               clf_dec_logits_emo, clf_dec_emo_repr
