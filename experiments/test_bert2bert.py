import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer,EncoderDecoderModel,GPT2Model

from core.utils.parser import get_generate_options
from core.data.empdataset import EmpatheticDataset
from core.data.collators import Bert2BertCollator
from core.utils.transforms import ToTensor
from core.utils.tensors import from_checkpoint, to_device


def _generate(options,model,test_loader,device):

    for index,batch in enumerate(tqdm(test_loader)):
        inputs = to_device(batch[0], device=device)
        inputs_lens = to_device(batch[1], device=device)
        inputs_att = to_device(batch[2], device=device)
        targets = to_device(batch[3], device=device)
        targets_lens = to_device(batch[4], device=device)
        targets_att = to_device(batch[5], device=device)
        import ipdb;ipdb.set_trace()
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
options = get_generate_options()

# load dataset
if options.dataset_name == "empchat":
    test_dataset = EmpatheticDataset("test", options.max_hist_len)
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
test_dataset = test_dataset.map(tokenize).map(to_tokens_ids).map(to_tensor)

# load test data
collator_fn = Bert2BertCollator(device='cpu')
test_loader = DataLoader(test_dataset, batch_size=options.batch_size,
                         drop_last=False, shuffle=True, collate_fn=collator_fn)


# load model from checkpoint
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    'bert-base-uncased', 'bert-base-uncased')
# model = from_checkpoint(options.modelckpt, model, map_location='cpu')
# model = model.to(DEVICE)
model.to(DEVICE)
# test model
_generate(options,model,test_loader,DEVICE)
