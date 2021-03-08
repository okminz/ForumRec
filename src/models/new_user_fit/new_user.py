from lightfm import LightFM
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k, reciprocal_rank
from lightfm import LightFM
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pickle
import json
import sys
from lightfm.data import Dataset



def main():
#     n = len(sys.argv)
#     if n > 0:
#         f = sys.argv[0]
#     else:
#         f = 'new_sample.csv'
    f = 'new_sample.csv'    
    print(f)
    user_indicies = np.load('user_indicies.npy')
    print(max(user_indicies))
    post_indicies = np.load('post_indicies.npy')
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
    new = new[['Score', 'Body', 'OwnerUserId', 'ParentId', 'Id', 'user_indicies',
       'post_indicies']]
    print(new.columns)
    print(new[['Score','OwnerUserId', 'ParentId', 'Id', 'user_indicies',
       'post_indicies']].values[0])
    new_user_indicies = dict()
    for i in range(len(new.OwnerUserId.unique())):
        new_user_indicies[new.OwnerUserId.unique()[i]] = dummies[i]
    new['user_indicies'] = new.OwnerUserId.apply(lambda x: new_user_indicies[x])
    print(new['user_indicies'].values)
    dataset.fit_partial((x for x in new.user_indicies.values),
             (x for x in new.post_indicies.values))
    (new_interactions, new_weights) = dataset.build_interactions(((x[5], x[6], x[0]) for x in new.values))
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

    
    
    