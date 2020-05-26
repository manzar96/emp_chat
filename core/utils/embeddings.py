import errno
import pickle
import os
import warnings
import numpy as np
import functools
from typing import cast, Any, Dict, Optional
from core.utils import mytypes
from core.utils.tokens import SPECIAL_TOKENS


def create_emb_file(new_emb_file, old_emb_file, freq_words_file, voc,
                    most_freq=None):

    sorted_voc = sorted(voc.items(), key=lambda kv: kv[1])
    with open(freq_words_file, "w") as file:
        if most_freq is not None:
            for item in sorted_voc[-most_freq:]:
                file.write(item[0]+'\n')
        else:
            for item in sorted_voc:
                file.write(item[0]+'\n')
    file.close()

    os.system("awk 'FNR==NR{a[$1];next} ($1 in a)' " + freq_words_file +
              " " + old_emb_file + ">" + new_emb_file)


class EmbeddingsLoader(object):
    def __init__(self, embeddings_file, dim, extra_tokens=SPECIAL_TOKENS):
        self.embeddings_file = embeddings_file
        self.cache_ = self._get_cache_name()
        self.dim_ = dim
        self.extra_tokens = extra_tokens

    def _get_cache_name(self):
        head, tail = os.path.split(self.embeddings_file)
        filename, ext = os.path.splitext(tail)
        cache_name = os.path.join(head, f'{filename}.p')
        print(f'Cache: {cache_name}')
        return cache_name

    def _dump_cache(self, data):
        with open(self.cache_, 'wb') as fd:
            pickle.dump(data, fd)

    def _load_cache(self):
        with open(self.cache_, 'rb') as fd:
            data = pickle.load(fd)
        return cast(mytypes.Embeddings, data)

    def augment_embeddings(
            self,
            word2idx,
            idx2word,
            embeddings,
            token: str,
            emb = None):
        word2idx[token] = len(embeddings)
        idx2word[len(embeddings)] = token
        if emb is None:
            emb = np.random.uniform(
                low=-0.05, high=0.05, size=self.dim_)
        embeddings.append(emb)
        return word2idx, idx2word, embeddings

    def load(self):
        """
        Read the word vectors from a text file
        Returns:
            word2idx (dict): dictionary of words to ids
            idx2word (dict): dictionary of ids to words
            embeddings (numpy.ndarray): the word embeddings matrix
        """
        # in order to avoid this time consuming operation, cache the results
        try:
            cache = self._load_cache()
            print("Loaded word embeddings from cache.")
            return cache
        except OSError:
            warnings.warn(f"Didn't find embeddings cache file {self.embeddings_file}")
        # create the necessary dictionaries and the word embeddings matrix
        if not os.path.exists(self.embeddings_file):
            #log.critical(f"{self.embeddings_file} not found!")
            print(f"{self.embeddings_file} not found!")
            raise OSError(errno.ENOENT, os.strerror(errno.ENOENT),
                          self.embeddings_file)

        #log.info(f'Indexing file {self.embeddings_file} ...')
        print(f'Indexing file {self.embeddings_file} ...')
        # create the 2D array, which will be used for initializing
        # the Embedding layer of a NN.
        # We reserve the first row (idx=0), as the word embedding,
        # which will be used for zero padding (word with id = 0).
        word2idx, idx2word, embeddings = self.augment_embeddings(
            {}, {}, [], self.extra_tokens.PAD.value,
            emb=np.zeros(self.dim_))
        for token in self.extra_tokens:
            if token == self.extra_tokens.PAD:
                continue
            word2idx, idx2word, embeddings = self.augment_embeddings(
                word2idx, idx2word, embeddings, token.value)

        # read file, line by line
        with open(self.embeddings_file, "r") as f:
            index = len(embeddings)
            for index, line in enumerate(f):
                # skip the first row if it is a header
                if len(line.split()) < self.dim_ and index == 0:
                    continue
                if len(line.split()) < self.dim_ and index != 0:
                    print("line: {}".format(line))
                    raise (ValueError, "Found an embedding with wrong dim!")

                values = line.rstrip().split(" ")
                word = values[0]

                if word in word2idx:
                    continue

                vector = np.asarray(values[1:], dtype=np.float32)
                idx2word[index] = word
                word2idx[word] = index
                embeddings.append(vector)
                index += 1

        #log.info(f'Found {len(embeddings)} word vectors.')
        print(f'Found {len(embeddings)} word vectors.')
        embeddings = np.array(embeddings, dtype='float32')

        # write the data to a cache file
        self._dump_cache((word2idx, idx2word, embeddings))
        return word2idx, idx2word, embeddings


if __name__ == '__main__':
    loader = EmbeddingsLoader(
        './cache/glove.6B.50d.txt', 50)
    embeddings = loader.load()
