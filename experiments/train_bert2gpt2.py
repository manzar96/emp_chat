import math
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer, GPT2Tokenizer, EncoderDecoderModel,\
    EncoderDecoderConfig

from core.utils.parser import get_train_parser
from core.data.empdataset import EmpatheticDataset
from core.data.persona import PersonaChatDataset
from core.data.collators import EncoderDecoderTransformerCollatorEmpChat, \
    EncoderDecoderTransformerCollatorPersChat
from core.utils.transforms import ToTensor
from core.trainers import EncoderDecoderTransformerTrainer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# get args from cmdline
parser = get_train_parser()
options = parser.parse_args()

# load dataset
if options.dataset_name == "empchat":
    train_dataset = EmpatheticDataset("train", options.max_hist_len)
    val_dataset = EmpatheticDataset("valid", options.max_hist_len)
elif options.dataset_name =="persona":
    train_dataset = PersonaChatDataset("train", options.max_hist_len)
    val_dataset = PersonaChatDataset("valid", options.max_hist_len)
else:
    raise NotImplementedError


# make transforms using only bert tokenizer!
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# CLS token will work as BOS token
bert_tokenizer.bos_token = bert_tokenizer.cls_token
# SEP token will work as EOS token
bert_tokenizer.eos_token = bert_tokenizer.sep_token


# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


GPT2Tokenizer.build_inputs_with_special_tokens = \
    build_inputs_with_special_tokens
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token
# TODO: edw pad_token_id== unk_token_id == eos_token_id == bos_token_id einai
#  swsto???

# we dont use map on dataset! so transforms will be [] and HuggingFace
# tokenizers will be applied
train_dataset.tokenizer_hist = bert_tokenizer
train_dataset.tokenizer_ans = gpt2_tokenizer
val_dataset.tokenizer_hist = bert_tokenizer
val_dataset.tokenizer_ans = gpt2_tokenizer

# load data
if options.dataset_name == "empchat":
    collator_fn = EncoderDecoderTransformerCollatorEmpChat(device='cpu')
elif "persona":
    collator_fn = EncoderDecoderTransformerCollatorPersChat(device='cpu')
train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)
val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)

# create model

#loipon edw mallon prepei na to kanw me to config wste na valw cross_attention
#na swsw to modelo me: save_pretrained("mymodel") kai meta na to kanw load!!!
if options.modelckpt is not None:
    model = EncoderDecoderModel.from_pretrained(options.modelckpt)
else:
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        'bert-base-uncased', 'gpt2')
model.to(DEVICE)

model.config.decoder_start_token_id = gpt2_tokenizer.bos_token_id
model.config.eos_token_id = gpt2_tokenizer.eos_token_id
model.config.max_length = 142
model.config.min_length = 56

#freeze some layers:
# for i in range(0,12):
#     for p in model.encoder.encoder.layer[i].parameters():
#         if p.requires_grad:
#             p.requires_grad = False

# params and optimizer
numparams = sum([p.numel() for p in model.parameters()])
train_numparams = sum([p.numel() for p in model.parameters() if
                       p.requires_grad])
print('Total Parameters: {}'.format(numparams))
print('Trainable Parameters: {}'.format(train_numparams))
optimizer = Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=options.lr, weight_decay=1e-6)

if options.optimckpt is not None:
    state_dict = torch.load(options.optim, map_location='cpu')
    optimizer.load_state_dict(state_dict)

import ipdb;ipdb.set_trace()

# create trainer
trainer = EncoderDecoderTransformerTrainer(model=model,
                                           optimizer=optimizer,
                                           patience=5,
                                           scheduler=None,
                                           checkpoint_dir=options.ckpt,
                                           device=DEVICE)
# train model
trainer.fit(train_loader, val_loader, epochs=options.epochs)
