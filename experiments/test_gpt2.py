import math
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

from core.utils.parser import get_test_parser
from core.models.huggingface.parser import add_cmdline_args_gen
from core.data.empdataset import EmpatheticDataset
from core.data.collators import GPT2CollatorEmpChat
from core.utils.transforms import ToTensor
from core.utils.tensors import to_device


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

def _generate(options,model,test_loader,tokenizer,device):

    for index, batch in enumerate(tqdm(test_loader)):
        inputs = to_device(batch[0], device=device)
        inputs_att = to_device(batch[1], device=device)

        output = model.generate(input_ids=inputs,
                       attention_mask=inputs_att,
                       max_length=50,
                       do_sample=options.sampling,
                       num_beams=options.beam_size,
                       temperature=options.temp,
                       top_k=options.topk,
                       top_p=options.topp,
                       num_return_sequences=options.Nbest,
                       )

        import ipdb;ipdb.set_trace()


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
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', additional_special_tokens=[
    '<s>','<|eoi|>']) # remember to resize token_embeddings on model!
tokenizer.pad_token = tokenizer.eos_token
eoi_index = tokenizer.additional_special_tokens_ids[1]
# tokenize = lambda x: tokenizer.tokenize(x)
# to_tokens_ids = lambda x: tokenizer.convert_tokens_to_ids(x)
# to_tensor = ToTensor()
# transforms = [tokenize, to_tokens_ids, to_tensor]
#
# # transform dataset
# test_dataset = test_dataset.map(tokenize).map(to_tokens_ids).map(to_tensor)

test_dataset.tokenizer_hist = tokenizer
test_dataset.tokenizer_ans = tokenizer

# load test data
collator_fn = GPT2CollatorEmpChat(endofinput_indx=eoi_index,device='cpu')
test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                         drop_last=False, shuffle=True, collate_fn=collator_fn)


# load model from checkpoint
model = GPT2LMHeadModel.from_pretrained('gpt2')
state_dict = torch.load(options.ckpt, map_location='cpu')
model.load_state_dict(state_dict)
model.config.dropout_rate = 0
model.to(DEVICE)

import ipdb;ipdb.set_trace()
# test model
#_generate(options,model,test_loader,tokenizer,DEVICE)

# calc and print metrics
calc_test_ppl(model, test_loader, DEVICE)
#calc_metrics(options, tokenizer)

#calc_similarity_trans(options)