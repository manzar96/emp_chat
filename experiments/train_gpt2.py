import math
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from core.utils.parser import get_options
from core.data.empdataset import EmpatheticDataset
from core.data.collators import GPT2Collator
from core.utils.transforms import ToTensor
from core.trainers import GPT2TransformerTrainer


DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
options = get_options()

# load dataset
if options.dataset_name == "empchat":
    train_dataset = EmpatheticDataset("train", options.max_hist_len)
    val_dataset = EmpatheticDataset("valid", options.max_hist_len)
else:
    raise NotImplementedError

# make transforms
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenize = lambda x: tokenizer.tokenize(x)
to_tokens_ids = lambda x: tokenizer.convert_tokens_to_ids(x)
to_tensor = ToTensor()
transforms = [tokenize, to_tokens_ids, to_tensor]
# TODO: set bos_index, eos_index sto bert tokenizer apo special tokens

# transform dataset
train_dataset = train_dataset.map(tokenize).map(to_tokens_ids).map(to_tensor)
val_dataset = val_dataset.map(tokenize).map(to_tokens_ids).map(to_tensor)

# load data
collator_fn = GPT2Collator(device='cpu')
train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)
val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)

# create model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.config.output_hidden_states=True
model.to(DEVICE)

# params and optimizer
numparams = sum([p.numel() for p in model.parameters()])
train_numparams = sum([p.numel() for p in model.parameters() if
                       p.requires_grad])
print('Total Parameters: {}'.format(numparams))
print('Trainable Parameters: {}'.format(train_numparams))
optimizer = Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.001, weight_decay=1e-6)

import ipdb;ipdb.set_trace()

# create trainer
trainer = GPT2TransformerTrainer(model=model, optimizer=optimizer,
                                 patience=5, scheduler=None,
                                 checkpoint_dir=options.ckpt, device=DEVICE)
# train model
trainer.fit(train_loader, val_loader, epochs=options.epochs)