from lightfm import LightFM
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k, reciprocal_rank
from lightfm import LightFM
import re
import threading
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pickle
import json
import sys
from lightfm.data import Dataset



def main():
    f = 'new_sample.csv'    
    print(f)
    user_indicies = np.load('user_indicies.npy')
    print(max(user_indicies))
    post_indicies = np.load('post_indicies.npy')
    post_mappings = pd.read_csv('post_mappings.csv')
    print(post_mappings.columns)
    post_mappings.columns = [ 'ParentId', 'post_indicies']
    post_mappings.index = post_mappings['ParentId']
    post_mappings = post_mappings['post_indicies']
    print(post_mappings.index[:10])
    post_ind = lambda x: post_mappings.loc[x]
    print(max(post_indicies))
    model = pickle.load(open("savefile.pickle", "rb"))
    dataset = Dataset()
    dataset.fit((x for x in user_indicies),
            (x for x in post_indicies))
    dummies = range(max(user_indicies) + 1, max(user_indicies)+100)
    dataset.fit_partial((x for x in dummies))  
    print(dataset.interactions_shape())
    new = pd.read_csv(f)
    print(new.columns)
    print(new.dtypes)
    new['post_indicies'] = new['ParentId'].apply(post_ind)
    print(new.columns)
    new_user_indicies = dict()
    for i in range(len(new.OwnerUserId.unique())):
        new_user_indicies[new.OwnerUserId.unique()[i]] = dummies[i]
    new['user_indicies'] = new.OwnerUserId.apply(lambda x: new_user_indicies[x])
    new = new[['user_indicies','post_indicies', 'Score', 'OwnerUserId', 'ParentId']]
    print(new['user_indicies'].values)
    dataset.fit_partial((x for x in new.user_indicies.values),
             (x for x in new.post_indicies.values))
    (new_interactions, new_weights) = dataset.build_interactions(((x[0], x[1], x[2]) for x in new.values))
    #interactions = sparse.load_npz("interactions.npz")
    item_features = sparse.load_npz("item_features.npz")
    for i in new.user_indicies.unique():
          print(i, 'mean user embedding before refitting :', np.mean(model.user_embeddings[i]))
    print(new_interactions.shape)
    model = model.fit_partial(new_interactions, item_features = item_features, sample_weight = new_weights,
         epochs=10, num_threads=8, verbose=True)      
    for i in new.user_indicies.unique():
          print(i, 'mean user embedding after refitting:', np.mean(model.user_embeddings[i]))      
    
    with open('savefile.pickle', 'wb') as fle:
        pickle.dump(model, fle, protocol=pickle.HIGHEST_PROTOCOL)
        #item_dict ={}
#     df = filtered_q.sort_values('post_indicies').reset_index()
#     for i in range(df.shape[0]):
#         item_dict[(df.loc[i,'post_indicies'])] = df.loc[i,'Id']
#     mean_recommendation_user(model, interactions,  item_features, 3, 
#                                item_dict, threshold = 0,nrec_items = 50, show = True)
    
    
    
if __name__ == "__main__": 
    main()

    
    
    