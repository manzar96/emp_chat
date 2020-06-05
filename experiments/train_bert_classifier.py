import math
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam,SGD,AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer,BertForSequenceClassification
from core.models.bert_classifier import BertClassifier

from core.utils.parser import get_train_parser
from core.data.empdataset import EmpatheticDatasetEmoClassification
from core.data.collators import BertClassificationEmpChatCollator
from core.utils.transforms import ToTensor
from core.trainers import BertForClassificationTrainer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# get args from cmdline
parser = get_train_parser()
options = parser.parse_args()

# load dataset
if options.dataset_name == "empchat":
    train_dataset = EmpatheticDatasetEmoClassification("train")
    val_dataset = EmpatheticDatasetEmoClassification("valid")
else:
    raise NotImplementedError

# make transforms
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenize = lambda x: tokenizer.tokenize(x)
to_tokens_ids = lambda x: tokenizer.convert_tokens_to_ids(x)
to_tensor = ToTensor()
transforms = [tokenize, to_tokens_ids, to_tensor]
# TODO: set bos_index, eos_index sto bert tokenizer apo special tokens

# transform dataset
train_dataset = train_dataset.map(tokenize).map(to_tokens_ids).map(to_tensor)
val_dataset = val_dataset.map(tokenize).map(to_tokens_ids).map(to_tensor)

# load data
collator_fn = BertClassificationEmpChatCollator(device='cpu')
train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)
val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)

# create model
#model = BertClassifier(num_classes=32)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased',
                                                      num_labels=32)
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
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=options.lr, weight_decay=1e-6)
#run with lr 0.001
if options.optimckpt is not None:
    state_dict = torch.load(options.optim, map_location='cpu')
    optimizer.load_state_dict(state_dict)

criterion = nn.CrossEntropyLoss()
import ipdb;ipdb.set_trace()

# create trainer
trainer = BertForClassificationTrainer(model=model,
                                       optimizer=optimizer,
                                       criterion=criterion,
                                       patience=5,
                                       scheduler=None,
                                       checkpoint_dir=options.ckpt,
                                       device=DEVICE)
# train model
trainer.fit(train_loader, val_loader, epochs=options.epochs)