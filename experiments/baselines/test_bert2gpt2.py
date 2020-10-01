import torch
import math
import os
import numpy as np
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from transformers import BertTokenizer, GPT2Tokenizer, EncoderDecoderModel

from core.utils.parser import get_test_parser
from core.models.huggingface.parser import add_cmdline_args_gen
from core.data.collators import EncoderDecoderTransformerCollatorEmpChat
from core.data.empdataset import EmpatheticDataset

from core.utils.tensors import to_device

from core.metrics.metrics import calc_sentence_bleu_score


def calc_metrics(options,tokenizer1,tokenizer2):
    outfile = open(os.path.join(options.outfolder, "gen_outs.txt"), "r")
    lines = outfile.readlines()
    bleu1=[]
    bleu2=[]
    bleu3=[]
    bleu4 = []
    word_error_rate = []
    for line in lines:
        inp, out, trgt = line[:-1].split("\t\t")
        inp = tokenizer1.encode(inp)
        out = tokenizer2.encode(out)
        trgt = tokenizer2.encode(trgt)

        bleu1.append(calc_sentence_bleu_score(trgt, out, n=1))
        bleu2.append(calc_sentence_bleu_score(trgt, out, n=2))
        bleu3.append(calc_sentence_bleu_score(trgt, out, n=3))
        bleu4.append(calc_sentence_bleu_score(trgt, out, n=4))
        #word_error_rate.append(calc_word_error_rate(trgt, out))
    print("BLEU1: {}".format(np.mean(bleu1)))
    print("BLEU2: {}".format(np.mean(bleu2)))
    print("BLEU3: {}".format(np.mean(bleu3)))
    print("BLEU4: {}".format(np.mean(bleu4)))
    print("Average BLEU score: {}".format( (np.mean(bleu1)+np.mean(
        bleu2)+np.mean(bleu3)+np.mean(bleu4))/4.0 ) )
    #print("Word Error Rate: {}".format(np.mean(word_error_rate)))

def calc_test_ppl(model, loader, device):
    with torch.no_grad():
        avg_loss = 0
        for index, batch in enumerate(tqdm(loader)):
            inputs = to_device(batch[0], device=device)
            inputs_att = to_device(batch[1], device=device)
            padded_targets = to_device(batch[2], device=device)
            replaced_targets = to_device(batch[3], device=device)
            targets_att = to_device(batch[4], device=device)

            outputs = model(input_ids=inputs,
                            attention_mask=inputs_att,
                            decoder_input_ids=padded_targets,
                            decoder_attention_mask=targets_att,
                            labels=replaced_targets)

            lm_loss = outputs[0]
            pred_scores = outputs[1]
            last_hidden = outputs[2]
            avg_loss += lm_loss.item()

        avg_loss = avg_loss / len(loader)

    print("Average Loss: {} | PPL {}".format(avg_loss, math.exp(avg_loss)))

def _generate(options, model, loader, tokenizer1, tokenizer2, device):

    if not os.path.exists(options.outfolder):
        os.makedirs(options.outfolder)
    outfile = open(os.path.join(options.outfolder, "gen_outs.txt"), "w")
    for index, batch in enumerate(tqdm(loader)):
        inputs = to_device(batch[0], device=device)
        inputs_att = to_device(batch[1], device=device)
        padded_targets = to_device(batch[2], device=device)
        replaced_targets = to_device(batch[3], device=device)
        targets_att = to_device(batch[4], device=device)

        outputs = model.generate(input_ids=inputs,
                       attention_mask=inputs_att,
                                 max_length=20,
                                 length_penaly=0.8,
                       do_sample=options.sampling,
                       num_beams=options.beam_size,
                       temperature=options.temp,
                       top_k=options.topk,
                       top_p=options.topp,
                       num_return_sequences=options.Nbest,
                       )
        # isws prepei stin generate na dwsw kai pad token id!
        #print(tokenizer.decode(outputs[0], skip_special_tokens=True))

        inp_list = ["".join(tokenizer1.decode(inputs[i],
                                             skip_special_tokens=True)) for i in range(
            inputs.shape[0])]
        out_list = ["".join(tokenizer2.decode(outputs[i],
                                              skip_special_tokens=True)) for i in range(
            inputs.shape[0])]
        tgt_list = ["".join(tokenizer2.decode(padded_targets[i],
                                              skip_special_tokens=True)) for i in
                    range(
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


# make transforms using only bert tokenizer!
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# CLS token will work as BOS token
bert_tokenizer.bos_token = bert_tokenizer.cls_token
# SEP token will work as EOS token
bert_tokenizer.eos_token = bert_tokenizer.sep_token

# make sure GPT2 appends EOS in begin and end
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
    return outputs


GPT2Tokenizer.build_inputs_with_special_tokens = \
    build_inputs_with_special_tokens
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# set pad_token_id to unk_token_id -> be careful here as unk_token_id == eos_token_id == bos_token_id
gpt2_tokenizer.pad_token = gpt2_tokenizer.unk_token
# TODO: edw pad_token_id== unk_token_id == eos_token_id == bos_token_id einai
#  swsto???

# we dont use map on dataset! so transforms will be [] and HuggingFace
# tokenizers will be applied
test_dataset.tokenizer_hist = bert_tokenizer
test_dataset.tokenizer_ans = gpt2_tokenizer

# load data
collator_fn = EncoderDecoderTransformerCollatorEmpChat(device='cpu')
test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                          drop_last=False, shuffle=True,
                          collate_fn=collator_fn)

# load model from checkpoint
model = EncoderDecoderModel.from_pretrained(options.modelckpt)
model.to(DEVICE)

import ipdb;ipdb.set_trace()
# generate answers model
_generate(options, model, test_loader, bert_tokenizer, gpt2_tokenizer, DEVICE)

# calc and print metrics
calc_test_ppl(model, test_loader, DEVICE)
calc_metrics(options, bert_tokenizer, gpt2_tokenizer)

#calc_similarity_trans(options)