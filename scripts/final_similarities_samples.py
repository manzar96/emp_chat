import pickle

pos_samples = pickle.load(open('./data/final_pos.pickle', 'rb'))
neg_pos_samples = pickle.load(open('./data/final_neg_pos.pickle', 'rb'))
neg_neg_samples = pickle.load(open('./data/final_neg_neg.pickle', 'rb'))

neg_samples=[]
for sample_neg_pos,sample_neg_neg in zip(neg_pos_samples,neg_neg_samples):
    sample = sample_neg_pos[0]
    neg_response = sample_neg_pos[1]
    pos_response = sample_neg_neg[1]
    neg_samples.append([sample,neg_response,pos_response])

with open('./data/final_neg.pickle', 'wb') as handle:
    pickle.dump(neg_samples, handle, protocol=pickle.HIGHEST_PROTOCOL)