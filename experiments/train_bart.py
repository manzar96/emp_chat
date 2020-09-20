import math
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration

from core.utils.parser import get_train_parser
from core.data.empdataset import EmpatheticDataset
from core.data.persona import PersonaChatDataset
from core.data.collators import T5CollatorEmpChat, T5CollatorPersChat
from core.utils.transforms import ToTensor
from core.trainers import BartTransformerTrainer


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
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
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
    collator_fn = T5CollatorEmpChat(device='cpu')
elif "persona":
    collator_fn = T5CollatorPersChat(device='cpu')
train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)
val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)

# create model
if options.modelckpt is not None:
    model = BartForConditionalGeneration.from_pretrained(options.modelckpt)
else:
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
model.config.output_hidden_states = True
model.to(DEVICE)
model.cuda()
# params and optimizer
numparams = sum([p.numel() for p in model.parameters()])
train_numparams = sum([p.numel() for p in model.parameters() if
                       p.requires_grad])
print('Total Parameters: {}'.format(numparams))
print('Trainable Parameters: {}'.format(train_numparams))
optimizer = Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=options.lr, weight_decay=1e-6)
# run with lr 0.001
if options.optimckpt is not None:
    state_dict = torch.load(options.optim, map_location='cpu')
    optimizer.load_state_dict(state_dict)


import ipdb;ipdb.set_trace()

# create trainer
trainer = BartTransformerTrainer(model=model, optimizer=optimizer,
                                 patience=5, scheduler=None,
                                 checkpoint_dir=options.ckpt, device=DEVICE)
# train model
trainer.fit(train_loader, val_loader, epochs=options.epochs)