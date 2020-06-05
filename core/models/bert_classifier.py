import torch.nn as nn

from transformers import BertTokenizer, BertModel


class BertClassifier(nn.Module):

    def __init__(self, num_classes):
        super(BertClassifier, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(self.encoder.config.hidden_dropout_prob)
        self.clf = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None):

        outputs = self.encoder(input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.clf(pooled_output)
        return logits