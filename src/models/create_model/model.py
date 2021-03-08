from lightfm import LightFM
import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from lightfm.cross_validation import random_train_test_split
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k, reciprocal_rank
import re
from ValidTestData import create_datasets
from ValidTestData import get_df
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pickle
import json



def mean_recommendation_user(model, interactions,item_features, user_id,  
                               item_dict,threshold = 0,nrec_items = 25, show = True):
    
    n_users, n_items = interactions.shape
    user_x = user_id #user_dict[str(user_id)]    print(user_x)
    scores = pd.Series(model.predict(user_x,np.arange(n_items), item_features=item_features, num_threads=4))
    scores.index = range(n_items)
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    scores = [x for x in scores]
    return_score_list = scores[0:nrec_items]
    
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    if show == True:
        print ("User: " + str(user_id))
        print("\n Recommended Items:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + str(i))
            counter+=1
    recs = pd.Series(return_score_list).apply(lambda x: item_dict[x])
    recs.to_csv('recs.csv')
    return scores

def main():
    new = pd.read_csv('20210222233206.csv')
    model = pickle.load(open("savefile.pickle", "rb"))
#     model.fit_partial(interactions, user_features=None, item_features=None, sample_weight=None, epochs=1, num_threads=1, verbose=False)
    print(model)
    interactions = sparse.load_npz("interactions.npz")
    item_features = sparse.load_npz("item_features.npz")
    filtered_a = pd.read_csv('filtered_a.csv')
    filtered_q = pd.read_csv('filtered_q.csv')
    filtered_a.Score = filtered_a.Score.apply(lambda x: x + 1)
    item_dict ={}
    df = filtered_q.sort_values('post_indicies').reset_index()
    for i in range(df.shape[0]):
        item_dict[(df.loc[i,'post_indicies'])] = df.loc[i,'Id']
    mean_recommendation_user(model, interactions,  item_features, 3, 
                               item_dict, threshold = 0,nrec_items = 50, show = True)
    
    
    
if __name__ == "__main__": 
    main()

    
    
    