
from datetime import datetime
import json
import logging
import os
import pickle
import tarfile
import tempfile
import socket
#from pytorch_transformers import cached_path
from transformers import cached_path
import torch
import numpy as np
from core.utils.tensors import mktensor
from torch.utils.data import Dataset



PERSONACHAT_URL = "https://s3.amazonaws.com/datasets.huggingface.co/personachat/personachat_self_original.json"

logger = logging.getLogger(__file__)


def get_dataset(tokenizer, dataset_path, dataset_cache):
    """ Get tokenized PERSONACHAT dataset from S3 or cache."""
    dataset_path = dataset_path or PERSONACHAT_URL
    # dataset_cache = dataset_cache + '_' + type(tokenizer).__name__  # To avoid using GPT cache for GPT-2 and vice-versa
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        dataset = torch.load(dataset_cache)
    else:
        logger.info("Download dataset from %s", dataset_path)
        personachat_file = cached_path(dataset_path)
        print(personachat_file)
        with open(personachat_file, "r", encoding="utf-8") as f:
            dataset = json.loads(f.read())
        # logger.info("Tokenize and encode the dataset")
        # def tokenize(obj):
        #     if isinstance(obj, str):
        #         return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        #     if isinstance(obj, dict):
        #         return dict((n, tokenize(o)) for n, o in obj.items())
        #     return list(tokenize(o) for o in obj)
        # dataset = tokenize(dataset)
        pickle.dump(dataset,
                    open('./data/Pesonachat_s3_amazonaws/personachat.pkl',
                         "wb"))
    return dataset


class PersonaChatDataset(Dataset):

    def __init__(self, splitname, maxhistorylen, tokenizer_hist=None,
                 tokenizer_ans=None):
        self.dataset_dict = pickle.load(open(
            './data/Pesonachat_s3_amazonaws/personachat.pkl', 'rb'))

        self.splitname = splitname
        self.maxhistorylen = maxhistorylen
        self.persona, self.data = self.get_data()
        self.transforms = []
        # we use different tokenizers for context and answers in case its
        # needed!
        self.tokenizer_hist = tokenizer_hist
        self.tokenizer_ans = tokenizer_ans

    def get_data(self):
        self.data_list = self.dataset_dict[self.splitname]
        persona = []
        data = []

        for index, dialog in enumerate(self.data_list):
            personality = dialog['personality']
            for turn in dialog['utterances']:
                cands = turn['candidates']
                hist = turn['history']
                hist = " </s> ".join(hist[-self.maxhistorylen:])
                # use this if we want to cut some samples according to
                # history len
                # if len(hist)>10:...

                persona.append(personality)
                # add to data last history turns and answer
                # from candidate answers we take as correct answer the last one
                data.append((hist, cands[-1]))
        return persona, data

    def map(self, t):
        self.transforms.append(t)
        return self

    def word_counts(self, tokenizer=None):
        voc_counts = {}
        for question, answer in self.data:
            if tokenizer is None:
                words, counts = np.unique(np.array(question.split(' ')),
                                          return_counts=True)
            else:
                words, counts = np.unique(np.array(tokenizer(question)),
                                          return_counts=True)
            for word, count in zip(words, counts):
                if word not in voc_counts.keys():
                    voc_counts[word] = count
                else:
                    voc_counts[word] += count

            if tokenizer is None:
                words, counts = np.unique(np.array(answer.split(' ')),
                                          return_counts=True)
            else:
                words, counts = np.unique(np.array(tokenizer(answer)),
                                          return_counts=True)
            for word, count in zip(words, counts):
                if word not in voc_counts.keys():
                    voc_counts[word] = count
                else:
                    voc_counts[word] += count

        return voc_counts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        personality = self.persona[index]
        hist, ans = self.data[index]

        if self.transforms == []:
            hist = self.tokenizer_hist(hist)
            ans = self.tokenizer_ans(ans)
            hist = mktensor(hist['input_ids'], dtype=torch.long)
            ans = mktensor(ans['input_ids'], dtype=torch.long)
        else:
            for t in self.transforms:
                hist = t(hist)
                ans = t(ans)
        # we dont use personality! if we want to return add below!
        return hist, ans

if __name__=="__main__":
    # call below to generate dataset pickle
    #get_dataset(None, PERSONACHAT_URL, None)
    mydataset = PersonaChatDataset(splitname='train',maxhistorylen=4)
    import ipdb;ipdb.set_trace()