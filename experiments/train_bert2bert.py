import math
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer,EncoderDecoderModel,EncoderDecoderConfig

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

# make transforms
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenize = lambda x: tokenizer.tokenize(x)
to_tokens_ids = lambda x: tokenizer.convert_tokens_to_ids(x)
to_tensor = ToTensor()
transforms = [tokenize, to_tokens_ids, to_tensor]
# CLS token will work as BOS token
tokenizer.bos_token = tokenizer.cls_token
# SEP token will work as EOS token
tokenizer.eos_token = tokenizer.sep_token
# TODO: set bos_index, eos_index sto bert tokenizer apo secial tokens

# transform dataset
train_dataset = train_dataset.map(tokenize).map(to_tokens_ids).map(to_tensor)
val_dataset = val_dataset.map(tokenize).map(to_tokens_ids).map(to_tensor)

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
if options.modelckpt is not None:
    model = EncoderDecoderModel.from_pretrained(options.modelckpt)
else:
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        'bert-base-uncased', 'bert-base-uncased')
model.to(DEVICE)

# #freeze some layers:
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
#run with lr 0.001
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
