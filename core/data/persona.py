
from datetime import datetime
import json
import logging
import os
import tarfile
import tempfile
import socket
#from pytorch_transformers import cached_path
from transformers import cached_path
import torch




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
        import ipdb;ipdb.set_trace()
        # logger.info("Tokenize and encode the dataset")
        # def tokenize(obj):
        #     if isinstance(obj, str):
        #         return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        #     if isinstance(obj, dict):
        #         return dict((n, tokenize(o)) for n, o in obj.items())
        #     return list(tokenize(o) for o in obj)
        # dataset = tokenize(dataset)
        # torch.save(dataset, dataset_cache)
    return dataset

if __name__=="__main__":
    mydataset = get_dataset(None,PERSONACHAT_URL,None)