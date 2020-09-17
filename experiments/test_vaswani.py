import torch
import math
import os
import pickle
import numpy as np
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from core.utils.transforms import DialogSpacyTokenizer, ToTensor, ToTokenIds

from core.utils.parser import get_test_parser
from core.models.transformers.parser import add_cmdline_args,\
    add_cmdline_args_gen
from core.data.empdataset import EmpatheticDataset
from core.utils.tokens import DIALOG_SPECIAL_TOKENS
from core.data.collators import TransformerVaswaniCollatorEmpChat
from core.models.transformers.modules_parlai import \
    TransformerEncodeDecoderVaswani
from core.modules.loss import SequenceCrossEntropyLoss
from core.utils.tensors import to_device
from core.metrics.metrics import calc_sentence_bleu_score, \
    calc_word_error_rate


def calc_test_ppl(model, loader,criterion, device):
    with torch.no_grad():
        avg_loss = 0
        for index, batch in enumerate(tqdm(loader)):
            inputs = to_device(batch[0], device=device)
            inputs_att = to_device(batch[1], device=device)
            pad_targets = to_device(batch[2], device=device)
            repl_targets = to_device(batch[3], device=device)
            targets_att = to_device(batch[4], device=device)
            scores, preds, encoder_states = model(inputs, ys=pad_targets)
            lm_loss = criterion(scores, pad_targets)
            avg_loss += lm_loss.item()

        avg_loss = avg_loss / len(loader)

    print("Average Loss: {} | PPL {}".format(avg_loss, math.exp(avg_loss)))


def calc_metrics(options,tokenizer):
    outfile = open(os.path.join(options.outfolder, "gen_outs.txt"), "r")
    lines = outfile.readlines()
    bleu4 = []
    word_error_rate = []
    for line in lines:
        inp, out, trgt = line[:-1].split("\t\t")
        inp = tokenizer.encode(inp)
        out = tokenizer.encode(out)
        trgt = tokenizer.encode(trgt)
        bleu4.append(calc_sentence_bleu_score(trgt, out, n=4))
        word_error_rate.append(calc_word_error_rate(trgt, out))
    print("BLEU: {}".format(np.mean(bleu4)))
    print("Word Error Rate: {}".format(np.mean(word_error_rate)))


def _generate(options, model, loader, idx2word, device):

    if not os.path.exists(options.outfolder):
        os.makedirs(options.outfolder)
    outfile = open(os.path.join(options.outfolder, "gen_outs.txt"), "w")
    for index, batch in enumerate(tqdm(loader)):
        inputs = to_device(batch[0], device=device)
        inputs_att = to_device(batch[1], device=device)
        pad_targets = to_device(batch[2], device=device)
        repl_targets = to_device(batch[3], device=device)
        targets_att = to_device(batch[4], device=device)

        scores, preds, encoder_states = model(inputs, ys=pad_targets)
        import ipdb;ipdb.set_trace()
        inp_tokens = [idx2word[token.item()] for token in inputs[0]]
        pred_tokens = [idx2word[token.item()] for token in preds[0]]
        # inp_list = ["".join(tokenizer.decode(inputs[i])) for i in range(
        #     inputs.shape[0])]
        # out_list = ["".join(tokenizer.decode(outputs[i])) for i in range(
        #     inputs.shape[0])]
        # tgt_list = ["".join(tokenizer.decode(pad_targets[i])) for i in range(
        #     inputs.shape[0])]
        # for i in range(len(inp_list)):
        #     outfile.write(inp_list[i]+"\t\t"+out_list[i]+"\t\t"+tgt_list[
        #         i]+"\n")

    outfile.close()
    print(len(loader))


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
parser = get_test_parser()
parser = add_cmdline_args(parser)
parser = add_cmdline_args_gen(parser)
options = parser.parse_args()

# load dataset
if options.dataset_name == "empchat":
    test_dataset = EmpatheticDataset("test", options.max_hist_len)
else:
    raise NotImplementedError

# load vocab
if options.vocabckpt is None:
    raise (IOError, "give --vocabckpt argument")
file1 = open(os.path.join(options.vocabckpt,'word2idx.pkl'),'rb')
word2idx = pickle.load(file1)
file2 = open(os.path.join(options.vocabckpt,'idx2word.pkl'),'rb')
idx2word = pickle.load(file2)

# make transforms
tokenizer = DialogSpacyTokenizer(lower=True, append_eos=True,
                                 specials=DIALOG_SPECIAL_TOKENS)
to_tokens_ids = ToTokenIds(word2idx, specials=DIALOG_SPECIAL_TOKENS)
to_tensor = ToTensor()
# transform dataset
test_dataset = test_dataset.map(tokenizer).map(to_tokens_ids).map(to_tensor)

# load test data
collator_fn = TransformerVaswaniCollatorEmpChat(device='cpu')
test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                         drop_last=False, shuffle=True, collate_fn=collator_fn)


# load model from checkpoint
model = TransformerEncodeDecoderVaswani(options,
                                        dictionary=word2idx,
                                        embedding_size=options.embeddings_size,
                                        embedding_weights=None,
                                        pad_idx=word2idx[
                                            DIALOG_SPECIAL_TOKENS.PAD.value],
                                        start_idx=word2idx[
                                            DIALOG_SPECIAL_TOKENS.SOS.value],
                                        end_idx=word2idx[
                                            DIALOG_SPECIAL_TOKENS.EOS.value],
                                        device=DEVICE)
state_dict = torch.load(options.modelckpt, map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)

import ipdb;ipdb.set_trace()
# generate answers model
#_generate(options, model, test_loader, idx2word, DEVICE)

# calc and print metrics
criterion = SequenceCrossEntropyLoss(word2idx[DIALOG_SPECIAL_TOKENS.PAD.value])
calc_test_ppl(model, test_loader, criterion, DEVICE)
#calc_metrics(options, tokenizer)

#calc_similarity_trans(options)