import random
import pickle
from core.data.empdataset import PreEmpatheticDatasetNeg

train_dataset = PreEmpatheticDatasetNeg('train', 10)

positive = ['surprised','excited','proud','grateful','impressed','hopeful',
            'confident','joyful','content','caring','trusting','faithful',
            'prepared','sentimental','anticipating']
negative= ['angry','sad','annoyed','lonely','afraid','terrified','guilty',
           'disgusted','furious','anxious','nostalgic','disappointed',
           'jealous','devastated','embarrassed','ashamed','apprehensive']
dict_pos={}
dict_neg ={}

for sample in train_dataset:
    hist, ans, label = sample
    if label in positive and (not label in dict_pos.keys()):
        dict_pos[label] = [[hist,ans,label]]
    elif label in positive:
        dict_pos[label].append([hist, ans, label])
    elif label in negative and (not label in dict_neg.keys()):
        dict_neg[label] = [[hist,ans,label]]
    elif label in negative:
        dict_neg[label].append([hist,ans,label])
    else:
        raise ValueError


dataset = []
for index,sample in enumerate(train_dataset):
    hist, ans, label = sample
    if label in dict_pos.keys():
        # then we select samples from negative dataset:
        negkeys = dict_neg.keys()
        emos = random.sample(negkeys,10)
        neg_samples = [random.sample(dict_neg[emo],3) for emo in emos]
    else:
        poskeys = dict_pos.keys()
        emos = random.sample(poskeys,10)
        neg_samples = [random.sample(dict_pos[emo],3) for emo in emos]


    neg_samples = [item for sublist in neg_samples for item in sublist]
    random.shuffle(neg_samples)
    dataset.append([hist,ans,label,neg_samples])

pickle.dump(dataset,open('./data/empatheticdialogues/train_neg_dataset.pkl',
                             'wb'))
