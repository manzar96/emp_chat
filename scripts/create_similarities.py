import random
import pickle5 as pickle
import numpy as np
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
# import ipdb;ipdb.set_trace()
# for key in dict_pos.keys():
#     print(key)
#     emb1 = dict_pos_emb[key].reshape(1,-1)
#     emb2 = [dict_pos_emb[key1].reshape(1,-1) for key1 in dict_pos.keys()]
#     pos_similarities = [cosine_similarity(emb1, emb2[i])[0][0] for i in
#                         range(len(emb2))]
#     emb2 = [dict_neg_emb[key1].reshape(1, -1) for key1 in dict_neg.keys()]
#     neg_similarities = [cosine_similarity(emb1, emb2[i])[0][0] for i in range(len(
#         emb2))]
#     # merge similarities with the corresponding sentence
#     pos_samples = [ [dict_pos[key][1],similarity] for key,similarity in
#                     zip(dict_pos.keys(),pos_similarities)]
#     del pos_samples[key]
#     pos_samples.sort(key=lambda x: x[1])
#
#     neg_samples = [ [dict_neg[key][1],similarity] for key,similarity in
#                     zip(dict_neg.keys(),neg_similarities)]
#     neg_samples.sort(key=lambda x: x[1])
#     sample = dict_pos[key]
#     dict_pos_final[key] = [sample,pos_samples,neg_samples]
#
# with open('./data/dict_pos.pickle', 'wb') as handle:
#     pickle.dump(dict_pos_final, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# dict_neg_final = {}
# for key in dict_neg.keys():
#     emb1 = dict_neg_emb[key].reshape(1,-1)
#     emb2 = [dict_neg_emb[key1].reshape(1,-1) for key1 in dict_neg.keys()]
#     neg_similarities = [cosine_similarity(emb1, emb2[i])[0][0] for i in
#                         range(len(emb2))]
#     emb2 = [dict_pos_emb[key1].reshape(1, -1) for key1 in dict_pos.keys()]
#     pos_similarities = [cosine_similarity(emb1, emb2[i])[0][0] for i in range(
#         len(emb2))]
#     # merge similarities with the corresponding sentence
#     neg_samples = [[dict_neg[key][1],similarity] for key,similarity in
#                     zip(dict_neg.keys(),neg_similarities)]
#     del neg_samples[key]
#     neg_samples.sort(key=lambda x: x[1])
#
#     pos_samples = [ [dict_pos[key][1],similarity] for key,similarity in
#                     zip(dict_pos.keys(),pos_similarities)]
#     pos_samples.sort(key=lambda x: x[1])
#     sample = dict_neg[key]
#     dict_neg_final[key] = [sample,neg_samples,pos_samples]
#
#
# with open('./data/dict_neg.pickle', 'wb') as handle:
#     pickle.dump(dict_neg_final, handle, protocol=pickle.HIGHEST_PROTOCOL)

# pos_emb = [dict_pos_emb[key1] for key1 in dict_pos.keys()]
# pos_similarities = cosine_similarity(pos_emb, pos_emb)
# sort_ind_pos = np.argsort(pos_similarities, axis=1)
# new_pos_samples = np.array(list(dict_pos.values()))
# final_pos = []
#
# neg_emb = [dict_neg_emb[key1] for key1 in dict_neg.keys()]
# neg_similarities = cosine_similarity(pos_emb, neg_emb)
# sort_ind_neg = np.argsort(neg_similarities, axis=1)
# new_neg_samples = np.array(list(dict_neg.values()))
#
#
#
#
# for index,key in enumerate(dict_pos.keys()):
#     print(key)
#     sample=dict_pos[key]
#     sorted_ind_pos = sort_ind_pos[index]
#     similarities = pos_similarities[index][sorted_ind_pos]
#     sorted_pos_samples = new_pos_samples[sorted_ind_pos]
#     #from the sorted_pos_samples we take the 2nd from the end! (and only
#     # the replie)
#     sim1 = similarities[-2]
#     pos_reply = sorted_pos_samples[-2][1]
#
#     sorted_ind_neg = sort_ind_neg[index]
#     similarities_neg = neg_similarities[index][sorted_ind_neg]
#     sorted_neg_samples = new_neg_samples[sorted_ind_neg]
#     sim2 = similarities_neg[-1]
#     neg_reply = sorted_neg_samples[-2][1]
#     final_pos.append([sample,[pos_reply,sim1],[[neg_reply,sim2]]])
#
# with open('./data/final_pos.pickle', 'wb') as handle:
#     pickle.dump(final_pos, handle, protocol=pickle.HIGHEST_PROTOCOL)



neg_emb = [dict_neg_emb[key1] for key1 in dict_neg.keys()]
neg_similarities = cosine_similarity(neg_emb, neg_emb)
sort_ind_neg = np.argsort(neg_similarities, axis=1)
new_neg_samples = np.array(list(dict_neg.values()))
final_neg = []

pos_emb = [dict_pos_emb[key1] for key1 in dict_pos.keys()]
pos_similarities = cosine_similarity(neg_emb, pos_emb)
sort_ind_pos = np.argsort(pos_similarities, axis=1)
new_pos_samples = np.array(list(dict_pos.values()))




for index,key in enumerate(dict_neg.keys()):
    print(key)
    sample=dict_neg[key]
    sorted_ind_neg = sort_ind_neg[index]
    similarities = neg_similarities[index][sorted_ind_neg]
    sorted_neg_samples = new_neg_samples[sorted_ind_neg]
    #from the sorted_pos_samples we take the 2nd from the end! (and only
    # the replie)
    sim1 = similarities[-2]
    neg_reply = sorted_neg_samples[-2][1]

    sorted_ind_pos = sort_ind_pos[index]
    similarities_pos = pos_similarities[index][sorted_ind_pos]
    sorted_pos_samples = new_pos_samples[sorted_ind_pos]
    sim2 = similarities_pos[-1]
    pos_reply = sorted_pos_samples[-2][1]
    final_neg.append([sample,[neg_reply,sim1],[[pos_reply,sim2]]])

with open('./data/final_neg.pickle', 'wb') as handle:
    pickle.dump(final_neg, handle, protocol=pickle.HIGHEST_PROTOCOL)