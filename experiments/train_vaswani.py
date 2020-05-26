import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from core.utils.parser import get_parser
from core.models.transformers.parser import add_cmdline_args
from core.utils.vocab import word2idx_from_dataset, Vocab
from core.utils.embeddings import EmbeddingsLoader, create_emb_file
from core.utils.tokens import DIALOG_SPECIAL_TOKENS
from core.data.empdataset import EmpatheticDataset
from core.data.collators import TransformerVaswaniCollator
from core.modules.loss import SequenceCrossEntropyLoss
from core.utils.transforms import DialogSpacyTokenizer,ToTensor, ToTokenIds
from core.trainers import TransformerVaswaniTrainer

from core.models.transformers.modules_parlai import TransformerEncodeDecoderVaswani

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
parser = get_parser()
parser = add_cmdline_args(parser)
options = parser.parse_args()

print(options)
# load dataset
if options.dataset_name == "empchat":
    train_dataset = EmpatheticDataset("train", options.max_hist_len)
    val_dataset = EmpatheticDataset("valid", options.max_hist_len)
else:
    raise NotImplementedError

# create tokenizer to split words and form vocab dict
tokenizer = DialogSpacyTokenizer(lower=True, append_eos=True,
                                 specials=DIALOG_SPECIAL_TOKENS)
# create dictionary with word counts
word_counts = train_dataset.word_counts(tokenizer)

if options.embeddings is not None:
    # TODO: na to kanw swsta! prepei na parw prwta gia oles tis lekseis ta
    #  embs kai meta na kratisw tis most frequent!
    # load embeddings (if given) and keep most frequent in Dataset
    new_emb_file = './cache/new_embs.txt'
    old_emb_file = options.embeddings
    freq_words_file = './cache/freq_words.txt'
    create_emb_file(new_emb_file, old_emb_file, freq_words_file, word_counts,
                    most_freq=10000)
    word2idx, idx2word, embeddings = EmbeddingsLoader(new_emb_file,
                                                      options.embeddings_size,
                                                      extra_tokens=
                                                      DIALOG_SPECIAL_TOKENS
                                                      ).load()
    options.embeddings_size = embeddings.shape[1]

else:
    # we create a dictionary from the dataset keeping again most frequent words
    word2idx, idx2word = word2idx_from_dataset(word_counts,
                                               most_freq=10000,
                                               extra_tokens=
                                               DIALOG_SPECIAL_TOKENS)
    embeddings = None


# make transforms
to_tokens_ids = ToTokenIds(word2idx, specials=DIALOG_SPECIAL_TOKENS)
to_tensor = ToTensor()


# transform dataset
train_dataset = train_dataset.map(tokenizer).map(to_tokens_ids).map(to_tensor)
val_dataset = val_dataset.map(tokenizer).map(to_tokens_ids).map(to_tensor)

# load data
collator_fn = TransformerVaswaniCollator(device='cpu')
train_loader = DataLoader(train_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)
val_loader = DataLoader(val_dataset, batch_size=options.batch_size,
                        drop_last=False, shuffle=True,
                        collate_fn=collator_fn)

import ipdb;ipdb.set_trace()
# create model
model = TransformerEncodeDecoderVaswani(options,
                                        dictionary=word2idx,
                                        embedding_size=options.embeddings_size,
                                        embedding_weights=embeddings,
                                        pad_idx=word2idx[
                                            DIALOG_SPECIAL_TOKENS.PAD.value],
                                        start_idx=word2idx[
                                            DIALOG_SPECIAL_TOKENS.SOS.value],
                                        end_idx=word2idx[
                                            DIALOG_SPECIAL_TOKENS.EOS.value],
                                        device=DEVICE)
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

criterion = SequenceCrossEntropyLoss(word2idx[DIALOG_SPECIAL_TOKENS.PAD.value])
import ipdb;ipdb.set_trace()

# create trainer
trainer = TransformerVaswaniTrainer(model=model,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    patience=5,
                                    scheduler=None,
                                    checkpoint_dir=options.ckpt,
                                    device=DEVICE)
# train model
trainer.fit(train_loader, val_loader, epochs=options.epochs)
