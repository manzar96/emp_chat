
def word2idx_from_dataset(wordcounts, most_freq=None, extra_tokens=None):
    word2idx = {}
    idx2word = {}
    counter = 0
    if extra_tokens is not None:
        for token in extra_tokens:
            word2idx[token.value] = counter
            idx2word[counter] = token.value
            counter += 1
    if most_freq is None:
        for word in wordcounts:
            if word not in word2idx:
                word2idx[word] = counter
                idx2word[counter] = word
                counter += 1
    else:
        sorted_voc = sorted(wordcounts.items(), key=lambda kv: kv[1])
        for word in sorted_voc[-most_freq:]:
            if word[0] not in word2idx:
                word2idx[word[0]] = counter
                idx2word[counter] = word[0]
                counter += 1
    return word2idx, idx2word


class Vocab:
    def __init__(self, word2idx, idx2word, pad_token=None, start_token=None,
                 eos_token=None):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab_size = len(self.word2idx)
        if pad_token:
            self.pad_idx = self.word2idx[pad_token]
        if start_token:
            self.pad_idx = self.word2idx[start_token]
        if eos_token:
            self.pad_idx = self.word2idx[eos_token]
