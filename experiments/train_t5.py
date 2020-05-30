import math
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from core.utils.parser import get_train_parser
from core.data.empdataset import EmpatheticDataset
from core.data.collators import T5Collator
from core.utils.transforms import ToTensor
from core.trainers import T5TransformerTrainer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# get args from cmdline
parser = get_train_parser()
options = parser.parse_args()

# load dataset
if options.dataset_name == "empchat":
    train_dataset = EmpatheticDataset("train", options.max_hist_len)
    val_dataset = EmpatheticDataset("valid", options.max_hist_len)
else:
    raise NotImplementedError

# make transforms
tokenizer = T5Tokenizer.from_pretrained('t5-small')
appender = lambda x: x+" </s>" # this one is used because we must add </s> at
# the end of each input/target
tokenize = lambda x: tokenizer.tokenize(x)
to_tokens_ids = lambda x: tokenizer.convert_tokens_to_ids(x)
to_tensor = ToTensor()

# transform dataset
train_dataset = train_dataset.map(appender).map(tokenize).map(to_tokens_ids).\
    map(to_tensor)
val_dataset = val_dataset.map(appender).map(tokenize).map(to_tokens_ids).map(
    to_tensor)

# load data
collator_fn = T5Collator(device='cpu')
train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)
val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)

# create model
model = T5ForConditionalGeneration.from_pretrained('t5-small')
model.config.output_hidden_states = True

if options.modelckpt is not None:
    state_dict = torch.load(options.modelckpt, map_location='cpu')
    model.load_state_dict(state_dict)
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
if options.optimckpt is not None:
    state_dict = torch.load(options.optim, map_location='cpu')
    optimizer.load_state_dict(state_dict)


import ipdb;ipdb.set_trace()

# create trainer
trainer = T5TransformerTrainer(model=model, optimizer=optimizer,
                                 patience=5, scheduler=None,
                                 checkpoint_dir=options.ckpt, device=DEVICE)
# train model
trainer.fit(train_loader, val_loader, epochs=options.epochs)