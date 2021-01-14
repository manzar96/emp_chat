import random
import pickle5 as pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from core.data.empdataset import PreEmpatheticDatasetNeg

train_dataset = PreEmpatheticDatasetNeg('train', 10)

positive = ['surprised','excited','proud','grateful','impressed','hopeful',
            'confident','joyful','content','caring','trusting','faithful',
            'prepared','sentimental','anticipating']
negative= ['angry','sad','annoyed','lonely','afraid','terrified','guilty',
           'disgusted','furious','anxious','nostalgic','disappointed',
           'jealous','devastated','embarrassed','ashamed','apprehensive']


# replies = [sample[1] for sample in train_dataset]
# model = SentenceTransformer('stsb-roberta-base')
# sentence_embeddings = model.encode(replies)
# with open('./data/replies_roberta_emb.pickle', 'wb') as handle:
#     pickle.dump(sentence_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/replies_roberta_emb.pickle', 'rb') as handle:
    replies_emb = pickle.load(handle)

dict_pos={}
dict_pos_emb={}
dict_neg ={}
dict_neg_emb ={}

for index,sample in enumerate(train_dataset):
    if sample[2] in positive:
        dict_pos[index] = sample
        dict_pos_emb[index] = replies_emb[index]
    else:
        dict_neg[index] = sample
        dict_neg_emb[index] = replies_emb[index]

dict_pos_final = {}
for key in dict_pos.keys():
    emb1 = dict_pos_emb[key].reshape(1,-1)
    emb2 = [dict_pos_emb[key1].reshape(1,-1) for key1 in dict_pos.keys()]
    pos_similarities = [cosine_similarity(emb1, emb2[i])[0][0] for i in
                        range(len(emb2))]
    emb2 = [dict_neg_emb[key1].reshape(1, -1) for key1 in dict_neg.keys()]
    neg_similarities = [cosine_similarity(emb1, emb2[i])[0][0] for i in range(len(
        emb2))]
    # merge similarities with the corresponding sentence
    pos_samples = [ [dict_pos[key][1],similarity] for key,similarity in
                    zip(dict_pos.keys(),pos_similarities)]
    del pos_samples[key]
    pos_samples.sort(key=lambda x: x[1])

    neg_samples = [ [dict_neg[key][1],similarity] for key,similarity in
                    zip(dict_neg.keys(),neg_similarities)]
    neg_samples.sort(key=lambda x: x[1])
    sample = dict_pos[key]
    dict_pos_final[key] = [sample,pos_samples,neg_samples]

dict_neg_final = {}
for key in dict_neg.keys():
    emb1 = dict_neg_emb[key].reshape(1,-1)
    emb2 = [dict_neg_emb[key1].reshape(1,-1) for key1 in dict_neg.keys()]
    neg_similarities = [cosine_similarity(emb1, emb2[i])[0][0] for i in
                        range(len(emb2))]
    emb2 = [dict_pos_emb[key1].reshape(1, -1) for key1 in dict_pos.keys()]
    pos_similarities = [cosine_similarity(emb1, emb2[i])[0][0] for i in range(
        len(emb2))]
    # merge similarities with the corresponding sentence
    neg_samples = [[dict_neg[key][1],similarity] for key,similarity in
                    zip(dict_neg.keys(),neg_similarities)]
    del neg_samples[key]
    neg_samples.sort(key=lambda x: x[1])

    pos_samples = [ [dict_pos[key][1],similarity] for key,similarity in
                    zip(dict_pos.keys(),pos_similarities)]
    pos_samples.sort(key=lambda x: x[1])
    sample = dict_neg[key]
    dict_neg_final[key] = [sample,neg_samples,pos_samples]


with open('./data/dict_neg.pickle', 'wb') as handle:
    pickle.dump(dict_neg_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./data/dict_pos.pickle', 'wb') as handle:
    pickle.dump(dict_pos_final, handle, protocol=pickle.HIGHEST_PROTOCOL)