import math
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer,EncoderDecoderModel

from core.utils.parser import get_options
from core.data.empdataset import EmpatheticDataset
from core.data.collators import Bert2BertCollator
from core.utils.transforms import ToTensor



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
options = get_options()

# load dataset
if options.dataset_name == "empchat":
    train_dataset = EmpatheticDataset("train", options.max_hist_len)
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

# load data
collator_fn = Bert2BertCollator(device='cpu')
train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)

# create model
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    'bert-base-uncased', 'bert-base-uncased')
model1 = EncoderDecoderModel.from_encoder_decoder_pretrained(
    'bert-base-uncased', 'gpt2')
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

# train model
EPOCHS = 10

model.train()
for epoch in range(0, EPOCHS):
    avg_lm_loss = 0

    for index,batch in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()
        inp,inp_len,inp_mask,trg,trg_len,trg_mask = batch
        inp = inp.to(DEVICE)
        inp_mask = inp_mask.to(DEVICE)
        trg = trg.to(DEVICE)
        trg_mask = trg_mask.to(DEVICE)

        outputs = model(input_ids=inp, attention_mask=inp_mask,
                        decoder_input_ids=trg, decoder_attention_mask=trg_mask,
                        lm_labels=trg)

        lm_loss = outputs[0]
        pred_scores = outputs[1]
        last_hidden = outputs[2]
        avg_lm_loss += lm_loss.item()
        ppl = math.exp(avg_lm_loss)
        lm_loss.backward()
        # if clip is not None:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(),clip)
        optimizer.step()

    print("Epoch {} | Loss {} | PPL {}".format(epoch, avg_lm_loss, ppl))
