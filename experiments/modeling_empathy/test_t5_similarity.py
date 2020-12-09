import torch
import math
import os
import numpy as np
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from core.models.huggingface.t5_extended import \
    T5ConditionalGenerationEmotions
from core.utils.parser import get_test_parser
from core.models.huggingface.parser import add_cmdline_args_gen
from core.data.empdataset import EmpatheticDataset
from core.data.collators import T5CollatorEmpChatEmo
from core.utils.transforms import ToTensor
from core.utils.tensors import to_device
from core.metrics.metrics import calc_sentence_bleu_score, \
    calc_word_error_rate




def calc_test_ppl(model, loader, device):
    with torch.no_grad():
        avg_loss = 0
        for index, batch in enumerate(tqdm(loader)):
            inputs = to_device(batch[0], device=device)
            inputs_att = to_device(batch[1], device=device)
            pad_targets = to_device(batch[2], device=device)
            repl_targets = to_device(batch[3], device=device)
            targets_att = to_device(batch[4], device=device)
            emo_label = to_device(batch[5], device=device)
            outputs = model(emolabel=emo_label, input_ids=inputs,
                                 attention_mask=inputs_att,
                                 labels=repl_targets)

            lm_loss = outputs[0]
            lm_logits = outputs[1]
            clf_logits = outputs[2]
            enc_emo_repr = outputs[3]
            dec_emo_repr = outputs[5]
            avg_loss += lm_loss.item()

        avg_loss = avg_loss / len(loader)

    print("Average Loss: {} | PPL {}".format(avg_loss, math.exp(avg_loss)))


def calc_metrics(options,tokenizer):
    outfile = open(os.path.join(options.outfolder, "gen_outs.txt"), "r")
    lines = outfile.readlines()
    bleu1=[]
    bleu2=[]
    bleu3=[]
    bleu4 = []
    word_error_rate = []
    for line in lines:
        inp, out, trgt = line[:-1].split("\t\t")
        inp = tokenizer.encode(inp)
        out = tokenizer.encode(out)
        trgt = tokenizer.encode(trgt)
        bleu1.append(calc_sentence_bleu_score(trgt, out, n=1))
        bleu2.append(calc_sentence_bleu_score(trgt, out, n=2))
        bleu3.append(calc_sentence_bleu_score(trgt, out, n=3))
        bleu4.append(calc_sentence_bleu_score(trgt, out, n=4))
        word_error_rate.append(calc_word_error_rate(trgt, out))
    print("BLEU1: {}".format(np.mean(bleu1)))
    print("BLEU2: {}".format(np.mean(bleu2)))
    print("BLEU3: {}".format(np.mean(bleu3)))
    print("BLEU4: {}".format(np.mean(bleu4)))
    print("Average BLEU score: {}".format( (np.mean(bleu1)+np.mean(
        bleu2)+np.mean(bleu3)+np.mean(bleu4))/4.0 ) )
    #print("Word Error Rate: {}".format(np.mean(word_error_rate)))


def _generate(options, model, loader, tokenizer, device):

    if not os.path.exists(options.outfolder):
        os.makedirs(options.outfolder)
    outfile = open(os.path.join(options.outfolder, "gen_outs.txt"), "w")
    for index, batch in enumerate(tqdm(loader)):
        inputs = to_device(batch[0], device=device)
        inputs_att = to_device(batch[1], device=device)
        pad_targets = to_device(batch[2], device=device)

        outputs = model.lm_model.generate(input_ids=inputs,
                       attention_mask=inputs_att,
                       max_length=40,
                       length_penalty=0.6,
                       do_sample=options.sampling,
                       num_beams=options.beam_size,
                       temperature=options.temp,
                       top_k=options.topk,
                       top_p=options.topp,
                       num_return_sequences=options.Nbest,
                       )
        inp_list = ["".join(tokenizer.decode(inputs[i],
                                             skip_special_tokens=True)) for i
                    in range(
            inputs.shape[0])]
        out_list = ["".join(tokenizer.decode(outputs[i],
                                             skip_special_tokens=True)) for i
                    in range(
            inputs.shape[0])]
        tgt_list = ["".join(tokenizer.decode(pad_targets[i],
                                             skip_special_tokens=True)) for i
                    in range(
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
tokenizer = T5Tokenizer.from_pretrained('t5-base')

# we dont use map on dataset! so transforms will be [] and HuggingFace
# tokenizers will be applied
test_dataset.tokenizer_hist = tokenizer
test_dataset.tokenizer_ans = tokenizer


# load test data
collator_fn = T5CollatorEmpChatEmo(device='cpu')
test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                         drop_last=False, shuffle=True, collate_fn=collator_fn)

lm_model = T5ForConditionalGeneration.from_pretrained('t5-base')
model = T5ConditionalGenerationEmotions(lm_model=lm_model,
                                        num_classes=32,
                                        device=DEVICE)

# load model from checkpoint
state_dict = torch.load(options.modelckpt, map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)
#we set dropout to zero for testing!
model.config.dropout_rate = 0

import ipdb;ipdb.set_trace()
# generate answers model
_generate(options, model, test_loader, tokenizer, DEVICE)

# calc and print metrics
calc_test_ppl(model, test_loader, DEVICE)
calc_metrics(options, tokenizer)

#calc_similarity_trans(options)