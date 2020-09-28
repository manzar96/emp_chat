import torch
import math
import os
import numpy as np
from tqdm import tqdm
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from transformers import BertTokenizer,EncoderDecoderModel

from sentence_transformers import SentenceTransformer

from core.utils.parser import get_test_parser
from core.models.huggingface.parser import add_cmdline_args_gen
from core.data.collators import EncoderDecoderTransformerCollatorEmpChat
from core.data.empdataset import EmpatheticDataset

from core.utils.tensors import to_device
from core.metrics.metrics import calc_sentence_bleu_score, \
    calc_word_error_rate

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

def _generate(options, model, loader, tokenizer, device):

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
                       max_length=50,
                       do_sample=options.sampling,
                       num_beams=options.beam_size,
                       temperature=options.temp,
                       top_k=options.topk,
                       top_p=options.topp,
                       num_return_sequences=options.Nbest,
                       )
        # isws prepei stin generate na dwsw kai pad token id!
        #print(tokenizer.decode(outputs[0], skip_special_tokens=True))

        inp_list = ["".join(tokenizer.decode(inputs[i])) for i in range(
            inputs.shape[0])]
        out_list = ["".join(tokenizer.decode(outputs[i])) for i in range(
            inputs.shape[0])]
        tgt_list = ["".join(tokenizer.decode(padded_targets[i])) for i in range(
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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# CLS token will work as BOS token
tokenizer.bos_token = tokenizer.cls_token
# SEP token will work as EOS token
tokenizer.eos_token = tokenizer.sep_token

# we dont use map on dataset! so transforms will be [] and HuggingFace
# tokenizers will be applied
test_dataset.tokenizer_hist = tokenizer
test_dataset.tokenizer_ans = tokenizer

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
#_generate(options, model, test_loader, tokenizer, DEVICE)

# calc and print metrics
calc_test_ppl(model, test_loader, DEVICE)
#calc_metrics(options, tokenizer)

#calc_similarity_trans(options)