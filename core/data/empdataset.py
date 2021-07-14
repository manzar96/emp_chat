import os
import numpy as np
import pickle as pickle
# import pickle as pickle
import random
from core.utils.tensors import mktensor
from torch.utils.data import Dataset
import torch

class EmpatheticDataset(Dataset):

    def __init__(self, splitname, maxhistorylen, tokenizer_hist=None,
                 tokenizer_ans=None):
        self.csvfile = os.path.join("data/empatheticdialogues",
                                    f"{splitname}.csv")
        self.maxhistorylen = maxhistorylen

        self.data, self.ids = self.read_data()
        self.label2idx, self.idx2label = self.get_labels_dict()

        # self.change_labels()
        self.transforms = []
        # we use different tokenizers for context and answers in case its
        # needed!
        self.tokenizer_hist = tokenizer_hist
        self.tokenizer_ans = tokenizer_ans


    def read_data(self):
        data = []
        ids = []

        history = []
        lines = open(self.csvfile).readlines()

        for i in range(1, len(lines)):
            cparts = lines[i - 1].strip().split(",")
            sparts = lines[i].strip().split(",")
            if cparts[0] == sparts[0]:
                prevsent = cparts[5].replace("_comma_", ",")
                history.append(prevsent)
                utt_idx = int(sparts[1])

                if utt_idx%2 == 0:
                    # we have an answer from listener so we store!
                    prev_hist = " </s> ".join(history[-self.maxhistorylen:])
                    answer = sparts[5].replace("_comma_", ",")
                    emolabel = sparts[2]
                    data.append((prev_hist, answer, emolabel))
                    ids.append((sparts[0], sparts[1]))

            else:
                # we have new conversation so empty history
                history = []
        # dict_conv={}
        # for index,sample in enumerate(data):
        #     dict_conv[index] = sample[0].split("</s>")
        # import ipdb;ipdb.set_trace()
        # convfile = open("./data/empatheticdialogues/conversation.pkl","wb")
        # pickle.dump(dict_conv,convfile)
        return data, ids

    def get_labels_dict(self):
        label2idx = {}
        idx2label = {}
        counter = 0
        for data in self.data:
            history, ans, label = data
            if label not in label2idx:
                label2idx[label] = counter
                idx2label[counter] = label
                counter += 1
        return label2idx, idx2label

    def change_labels(self):
        positive = ['surprised', 'excited', 'proud', 'grateful', 'impressed',
                    'hopeful',
                    'confident', 'joyful', 'content', 'caring', 'trusting',
                    'faithful',
                    'prepared', 'sentimental', 'anticipating']
        negative = ['angry', 'sad', 'annoyed', 'lonely', 'afraid', 'terrified',
                    'guilty',
                    'disgusted', 'furious', 'anxious', 'nostalgic',
                    'disappointed',
                    'jealous', 'devastated', 'embarrassed', 'ashamed',
                    'apprehensive']
        for key in self.label2idx.keys():
            if key in positive:
                self.label2idx[key] = 1
                self.idx2label[1] = key
            elif key in negative:
                self.label2idx[key] = 0
                self.idx2label[0] = key
            else:
                raise ValueError

    def bert_transform_data(self, tokenize):
        alldata = []
        for data in self.data:
            history, answer, label = data
            new_hist = tokenize(history)
            new_ans = tokenize(answer)
            label_idx = self.label2idx[label]
            alldata.append((new_hist, new_ans, label_idx))
        self.data = alldata

    def map(self, t):
        self.transforms.append(t)
        return self

    def word_counts(self, tokenizer=None):
        voc_counts = {}
        for question, answer, label in self.data:
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
        hist, ans, label = self.data[index]
        if self.transforms == []:
            hist = self.tokenizer_hist(hist)
            ans = self.tokenizer_ans(ans)
            hist = mktensor(hist['input_ids'],dtype=torch.long)
            ans = mktensor(ans['input_ids'],dtype=torch.long)
        else:
            for t in self.transforms:
                hist = t(hist)
                ans = t(ans)
        label = mktensor(self.label2idx[label],dtype=torch.long)
        return hist, ans, label

    def getid(self, index):
        return self.ids[index]

class PreEmpatheticDatasetNeg(Dataset):

    def __init__(self, splitname, maxhistorylen):
        self.csvfile = os.path.join("data/empatheticdialogues",
                                    f"{splitname}.csv")
        self.maxhistorylen = maxhistorylen

        self.data, self.ids = self.read_data()
        self.label2idx, self.idx2label = self.get_labels_dict()
        self.transforms = []

    def read_data(self):
        data = []
        ids = []

        history = []
        lines = open(self.csvfile).readlines()

        for i in range(1, len(lines)):
            cparts = lines[i - 1].strip().split(",")
            sparts = lines[i].strip().split(",")
            if cparts[0] == sparts[0]:
                prevsent = cparts[5].replace("_comma_", ",")
                history.append(prevsent)
                utt_idx = int(sparts[1])

                if utt_idx%2 == 0:
                    # we have an answer from listener so we store!
                    prev_hist = " </s> ".join(history[-self.maxhistorylen:])
                    answer = sparts[5].replace("_comma_", ",")
                    emolabel = sparts[2]
                    data.append((prev_hist, answer, emolabel))
                    ids.append((sparts[0], sparts[1]))

            else:
                # we have new conversation so empty history
                history = []
        return data, ids

    def get_labels_dict(self):
        label2idx = {}
        idx2label = {}
        counter = 0
        for data in self.data:
            history, ans, label = data
            if label not in label2idx:
                label2idx[label] = counter
                idx2label[counter] = label
                counter += 1
        return label2idx, idx2label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        hist, ans, label = self.data[index]
        return hist, ans, label

    def getid(self, index):
        return self.ids[index]


class EmpatheticDatasetNeg(Dataset):

    def __init__(self, splitname, maxhistorylen, tokenizer_hist=None,
                 tokenizer_ans=None):
        self.csvfile = os.path.join("data/empatheticdialogues",
                                    f"{splitname}.csv")
        self.maxhistorylen = maxhistorylen

        self.data = self.read_data()
        self.label2idx, self.idx2label = self.get_labels_dict()
        self.change_labels()
        self.transforms = []
        # we use different tokenizers for context and answers in case its
        # needed!
        self.tokenizer_hist = tokenizer_hist
        self.tokenizer_ans = tokenizer_ans

    def read_data(self):
        dataset = pickle.load(open(
            './data/empatheticdialogues/train_neg_dataset.pkl',
                                 'rb'))
        return dataset

    def get_labels_dict(self):
        label2idx = {}
        idx2label = {}
        counter = 0
        for data in self.data:
            history, ans, label, neg_samples = data
            if label not in label2idx:
                label2idx[label] = counter
                idx2label[counter] = label
                counter += 1
        return label2idx, idx2label

    def change_labels(self):
        positive = ['surprised', 'excited', 'proud', 'grateful', 'impressed',
                    'hopeful',
                    'confident', 'joyful', 'content', 'caring', 'trusting',
                    'faithful',
                    'prepared', 'sentimental', 'anticipating']
        negative = ['angry', 'sad', 'annoyed', 'lonely', 'afraid', 'terrified',
                    'guilty',
                    'disgusted', 'furious', 'anxious', 'nostalgic',
                    'disappointed',
                    'jealous', 'devastated', 'embarrassed', 'ashamed',
                    'apprehensive']
        for key in self.label2idx.keys():
            if key in positive:
                self.label2idx[key] = 1
                self.idx2label[1] = key
            elif key in negative:
                self.label2idx[key] = 0
                self.idx2label[0] = key
            else:
                raise ValueError


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        hist, ans, label, neg_samples = self.data[index]
        if self.transforms == []:
            hist = self.tokenizer_hist(hist)
            ans = self.tokenizer_ans(ans)
            neg_samples = [self.tokenizer_ans(ans) for hist,ans,label in
                           neg_samples]

            hist = mktensor(hist['input_ids'],dtype=torch.long)
            ans = mktensor(ans['input_ids'],dtype=torch.long)
            neg = [mktensor(item['input_ids'],dtype=torch.long) for item in neg_samples]
        else:
            for t in self.transforms:
                hist = t(hist)
                ans = t(ans)
        label = mktensor(self.label2idx[label],dtype=torch.long)
        return hist, ans, label,neg


    def getid(self, index):
        return self.ids[index]


class EmpatheticDatasetPosNeg(Dataset):

    def __init__(self, splitname, maxhistorylen, tokenizer_hist=None,
                 tokenizer_ans=None):
        if splitname == "test":
            raise NotImplementedError
        self.maxhistorylen = maxhistorylen

        self.data = self.read_data()
        self.label2idx, self.idx2label = self.get_labels_dict()
        self.change_labels()
        self.transforms = []
        # we use different tokenizers for context and answers in case its
        # needed!
        self.tokenizer_hist = tokenizer_hist
        self.tokenizer_ans = tokenizer_ans

    def read_data(self):
        pos_samples = pickle.load(open('./data/final_pos.pickle','rb'))
        # fix pos samples
        fixed_pos_samples = []
        for sample,sample_pos,sample_neg in pos_samples:
            fixed_pos_samples.append([sample,sample_pos,sample_neg[0]])
        neg_samples = pickle.load(open('./data/final_neg.pickle', 'rb'))
        data = fixed_pos_samples+neg_samples
        random.shuffle(data)
        return data

    def get_labels_dict(self):
        label2idx = {}
        idx2label = {}
        counter = 0
        for data in self.data:
            sample,sample_same,sample_wrong = data
            label = sample[-1]
            if label not in label2idx:
                label2idx[label] = counter
                idx2label[counter] = label
                counter += 1
        return label2idx, idx2label

    def change_labels(self):
        positive = ['surprised', 'excited', 'proud', 'grateful', 'impressed',
                    'hopeful',
                    'confident', 'joyful', 'content', 'caring', 'trusting',
                    'faithful',
                    'prepared', 'sentimental', 'anticipating']
        negative = ['angry', 'sad', 'annoyed', 'lonely', 'afraid', 'terrified',
                    'guilty',
                    'disgusted', 'furious', 'anxious', 'nostalgic',
                    'disappointed',
                    'jealous', 'devastated', 'embarrassed', 'ashamed',
                    'apprehensive']
        for key in self.label2idx.keys():
            if key in positive:
                self.label2idx[key] = 1
                self.idx2label[1] = key
            elif key in negative:
                self.label2idx[key] = 0
                self.idx2label[0] = key
            else:
                raise ValueError


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample, sample_same, sample_wrong = self.data[index]
        hist, ans, label = sample
        if self.transforms == []:
            hist = self.tokenizer_hist(hist)
            ans = self.tokenizer_ans(ans)
            ans_same = sample_same[0]
            ans_wrong = sample_wrong[0]
            ans_same = self.tokenizer_ans(ans_same)
            ans_wrong = self.tokenizer_ans(ans_wrong)

            hist = mktensor(hist['input_ids'],dtype=torch.long)
            ans = mktensor(ans['input_ids'],dtype=torch.long)
            ans_same = mktensor(ans_same['input_ids'],dtype=torch.long)
            ans_wrong = mktensor(ans_wrong['input_ids'],dtype=torch.long)

        else:
            for t in self.transforms:
                hist = t(hist)
                ans = t(ans)
        label = mktensor(self.label2idx[label],dtype=torch.long)
        return hist, ans, label, ans_same, ans_wrong


if __name__ == "__main__":

    test_dataset = EmpatheticDataset('test', 4)
