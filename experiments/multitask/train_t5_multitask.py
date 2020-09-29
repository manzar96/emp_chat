import math
import torch
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn
from core.modules.optims import Adafactor
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from core.utils.parser import get_train_parser
from core.data.empdataset import EmpatheticDataset
from core.data.persona import PersonaChatDataset
from core.data.collators import T5CollatorEmpChatMultitask, T5CollatorPersChat
from core.models.huggingface.t5_extended import T5ConditionalGenerationDoubleHead
from core.utils.transforms import ToTensor
from core.trainers import T5TransformerTrainerMultitask


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# get args from cmdline
parser = get_train_parser()
options = parser.parse_args()

# load dataset
if options.dataset_name == "empchat":
    train_dataset = EmpatheticDataset("train", options.max_hist_len)
    val_dataset = EmpatheticDataset("valid", options.max_hist_len)
elif options.dataset_name == "persona":
    train_dataset = PersonaChatDataset("train", options.max_hist_len)
    val_dataset = PersonaChatDataset("valid", options.max_hist_len)
else:
    raise NotImplementedError

# make transforms
tokenizer = T5Tokenizer.from_pretrained('t5-base')
# appender = lambda x: x+" </s>" # this one is used because we must add </s> at
# # the end of each input/target
# tokenize = lambda x: tokenizer.tokenize(x)
# to_tokens_ids = lambda x: tokenizer.convert_tokens_to_ids(x)
# to_tensor = ToTensor()
#
# # transform dataset
# train_dataset = train_dataset.map(appender).map(tokenize).map(to_tokens_ids).\
#     map(to_tensor)
# val_dataset = val_dataset.map(appender).map(tokenize).map(to_tokens_ids).map(
#     to_tensor)

"""Uncomment the above to use map with transforms"""
# we dont use map on dataset! so transforms will be [] and HuggingFace
# tokenizers will be applied
train_dataset.tokenizer_hist = tokenizer
train_dataset.tokenizer_ans = tokenizer
val_dataset.tokenizer_hist = tokenizer
val_dataset.tokenizer_ans = tokenizer

# load data
if options.dataset_name == "empchat":
    collator_fn = T5CollatorEmpChatMultitask(device='cpu')
elif "persona":
    collator_fn = T5CollatorPersChat(device='cpu')
train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)
val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)

# create model
model = T5ConditionalGenerationDoubleHead(model_version='t5-base',
                                          num_classes=32,
                                          device=DEVICE)
if options.modelckpt is not None:
    state_dict = torch.load(options.modelckpt, map_location='cpu')
    model.load_state_dict(state_dict)
model.lm_model.config.output_hidden_states = True
model.lm_model.config.dropout_rate = 0.2
model.to(DEVICE)

# params and optimizer
numparams = sum([p.numel() for p in model.parameters()])
train_numparams = sum([p.numel() for p in model.parameters() if
                       p.requires_grad])
print('Total Parameters: {}'.format(numparams))
print('Trainable Parameters: {}'.format(train_numparams))
optimizer = Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=options.lr, weight_decay=1e-6)
# optimizer = Adafactor(
#     [p for p in model.parameters() if p.requires_grad], weight_decay=1e-6)
# run with lr 0.001
if options.optimckpt is not None:
    state_dict = torch.load(options.optim, map_location='cpu')
    optimizer.load_state_dict(state_dict)


import ipdb;ipdb.set_trace()

criterion = nn.CrossEntropyLoss(ignore_index=-100)
# create trainer
trainer = T5TransformerTrainerMultitask(model=model, optimizer=optimizer,
                                 patience=5,criterion=criterion, scheduler=None,
                                 checkpoint_dir=options.ckpt, device=DEVICE)
# train model
trainer.fit(train_loader, val_loader, epochs=options.epochs)