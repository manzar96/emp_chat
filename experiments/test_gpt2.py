import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

from core.utils.parser import get_test_parser
from core.models.huggingface.parser import add_cmdline_args_gen
from core.data.empdataset import EmpatheticDataset
from core.data.collators import GPT2Collator
from core.utils.transforms import ToTensor
from core.utils.tensors import to_device


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
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenize = lambda x: tokenizer.tokenize(x)
to_tokens_ids = lambda x: tokenizer.convert_tokens_to_ids(x)
to_tensor = ToTensor()
transforms = [tokenize, to_tokens_ids, to_tensor]
# TODO: set bos_index, eos_index sto bert tokenizer apo special tokens

# transform dataset
test_dataset = test_dataset.map(tokenize).map(to_tokens_ids).map(to_tensor)

# load test data
collator_fn = GPT2Collator(device='cpu')
test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                         drop_last=False, shuffle=True, collate_fn=collator_fn)


# load model from checkpoint
model = GPT2LMHeadModel.from_pretrained('gpt2')
state_dict = torch.load(options.ckpt, map_location='cpu')
model.load_state_dict(state_dict)
model.to(DEVICE)

import ipdb;ipdb.set_trace()
# test model
_generate(options,model,test_loader,tokenizer,DEVICE)