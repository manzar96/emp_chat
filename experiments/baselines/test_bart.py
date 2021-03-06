import torch
import math
import os
import numpy as np
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer


from core.utils.parser import get_test_parser
from core.models.huggingface.parser import add_cmdline_args_gen
from core.data.empdataset import EmpatheticDataset
from core.data.collators import T5CollatorEmpChat
from core.utils.transforms import ToTensor
from core.utils.tensors import to_device
from core.metrics.metrics import calc_sentence_bleu_score, \
    calc_word_error_rate
from core.modules.loss import SequenceCrossEntropyLoss

def calc_similarity_trans(options):

    all_sentences = []
    outfile = open(os.path.join(options.outfolder, "gen_outs.txt"), "r")
    lines = outfile.readlines()
    for line in lines:
        inp, out, trgt = line[:-1].split("\t\t")
        all_sentences.append(inp)
        all_sentences.append(out)
        all_sentences.append(trgt)

    model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
    sentence_embeddings = model.encode(all_sentences)
    mydict = dict(zip(all_sentences, sentence_embeddings))

    cos_sim=[]
    for line in lines:
        inp, out, trgt = line[:-1].split("\t\t")
        tensor1 = torch.tensor(mydict[out]).unsqueeze(0)
        tensor2 = torch.tensor(mydict[trgt]).unsqueeze(0)
        cos_sim.append(cosine_similarity(tensor1, tensor2))

    print("Cosine Similairity: {}".format(np.mean(cos_sim)))


def calc_test_ppl(model, loader, device):
    crit = SequenceCrossEntropyLoss(pad_idx=-100)
    with torch.no_grad():
        avg_loss = 0
        for index, batch in enumerate(tqdm(loader)):
            inputs = to_device(batch[0], device=device)
            inputs_att = to_device(batch[1], device=device)
            pad_targets = to_device(batch[2], device=device)
            repl_targets = to_device(batch[3], device=device)
            targets_att = to_device(batch[4], device=device)

            outputs = model(input_ids=inputs, attention_mask=inputs_att,
                            decoder_input_ids=pad_targets)

            pred_scores = outputs[0]
            last_hidden = outputs[1]
            lm_loss = crit(pred_scores, repl_targets)
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


def _generate(options, model, loader, tokenizer, device):

    if not os.path.exists(options.outfolder):
        os.makedirs(options.outfolder)
    outfile = open(os.path.join(options.outfolder, "gen_outs.txt"), "w")
    for index, batch in enumerate(tqdm(loader)):
        inputs = to_device(batch[0], device=device)
        inputs_att = to_device(batch[1], device=device)
        pad_targets = to_device(batch[2], device=device)
        repl_targets = to_device(batch[3], device=device)
        targets_att = to_device(batch[4], device=device)

        outputs = model.generate(input_ids=inputs,
                       attention_mask=inputs_att,
                       max_length=50,
                       do_sample=options.sampling,
                       num_beams=options.beam_size,
                       temperature=options.temp,
                       top_k=options.topk,
                       top_p=options.topp,
                       num_return_sequences=options.Nbest,
                       )
        inp_list = ["".join(tokenizer.decode(inputs[i])) for i in range(
            inputs.shape[0])]
        out_list = ["".join(tokenizer.decode(outputs[i])) for i in range(
            inputs.shape[0])]
        tgt_list = ["".join(tokenizer.decode(pad_targets[i])) for i in range(
            inputs.shape[0])]
        for i in range(len(inp_list)):
            outfile.write(inp_list[i]+"\t\t"+out_list[i]+"\t\t"+tgt_list[
                i]+"\n")

    outfile.close()
    print(len(loader))


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# get args from cmdline
parser = get_test_parser()
parser = add_cmdline_args_gen(parser)
options = parser.parse_args()

# load dataset
if options.dataset_name == "empchat":
    test_dataset = EmpatheticDataset("test", options.max_hist_len)
else:
    raise NotImplementedError

# make transforms
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
# tokenize = lambda x: tokenizer.tokenize(x)
# to_tokens_ids = lambda x: tokenizer.convert_tokens_to_ids(x)
# to_tensor = ToTensor()
# transforms = [tokenize, to_tokens_ids, to_tensor]
#
# # transform dataset
# test_dataset = test_dataset.map(tokenize).map(to_tokens_ids).map(to_tensor)
"""Uncomment the above to use map with transforms"""
# we dont use map on dataset! so transforms will be [] and HuggingFace
# tokenizers will be applied
test_dataset.tokenizer_hist = tokenizer
test_dataset.tokenizer_ans = tokenizer


# load test data
collator_fn = T5CollatorEmpChat(device='cpu')
test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                         drop_last=False, shuffle=True, collate_fn=collator_fn)


# load model from checkpoint
model = BartForConditionalGeneration.from_pretrained(options.modelckpt)
model.to(DEVICE)

import ipdb;ipdb.set_trace()
# generate answers model
_generate(options, model, test_loader, tokenizer, DEVICE)

# calc and print metrics
#calc_test_ppl(model, test_loader, DEVICE)
#calc_metrics(options, tokenizer)

#calc_similarity_trans(options)