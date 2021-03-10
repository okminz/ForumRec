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
from ValidTestData import create_datasets
from ValidTestData import get_df
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import pickle
import json
from lightfm.data import Dataset






def main():
    FILE_DIRECTORY = ''
    FILE = 'Posts.xml'
    SPLIT_DATE = pd.datetime(2018, 1, 1)
    train_q, train_a, valid_q, valid_a, test_q, test_a = create_datasets(FILE_DIRECTORY,FILE , SPLIT_DATE)
    train_a = pd.concat([valid_a, test_a])
    train_q = pd.concat([valid_q, test_q])
    train_a['Id'] = train_a.index
    train_q['Id'] = train_q.index
    train_a = train_a.sort_values('Score', ascending=False)
    
    grouped_a = train_a.groupby('OwnerUserId').count()
    print(grouped_a.shape)
    grouped_a = grouped_a[grouped_a.Id > 9]
    print(grouped_a.shape)
    grouped_a['OwnerUserId'] = grouped_a.index
    users = list(grouped_a.OwnerUserId)
    print(len(users))
    print('original answers data shape: ',train_a.shape)
    filtered_a = train_a[train_a.OwnerUserId.isin(users)]
    print('filtered answers data shape: ', filtered_a.shape)
    print('original questions data shape: ', train_q.shape)
    filtered_q = train_q[train_q.Id.isin(filtered_a.ParentId)]
    print('filtered questions data shape: ', filtered_q.shape)
    
    clean = re.compile('<.*?>')
    clean_text = lambda x: re.sub(clean, '', x)
    clean_tags = lambda x: x.replace('<', ' ').replace('>', ' ')
    filtered_q['Tags'] = filtered_q['Tags'].fillna('').apply(clean_tags)
    filtered_q['Body'] = filtered_q['Body'].fillna('').apply(clean_text)
    
    filtered_a.index = range(len(filtered_a.Id))
    filtered_q.index = range(len(filtered_q.Id))
    user_indices = pd.Series(range(len(filtered_a['OwnerUserId'].unique())), index=filtered_a['OwnerUserId'].unique()).drop_duplicates()
    
    user_dict = dict(user_indices)
    user_id = list(user_indices.values)
    post_indices = pd.Series(range(len(filtered_a['ParentId'].unique())), index=filtered_a['ParentId'].unique()).drop_duplicates()
    user_ind = lambda x: user_indices.loc[x]
    post_ind = lambda x: post_indices.loc[x]


    filtered_a['user_indicies'] = filtered_a['OwnerUserId'].apply(user_ind)
    filtered_a['post_indicies'] = filtered_a['ParentId'].apply(post_ind)
    filtered_q['post_indicies'] = filtered_q['Id'].apply(post_ind)
    filtered_q = filtered_q.sort_values('post_indicies', ascending=True)
    
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    #Replace NaN with an empty string
    filtered_q['Body'] = filtered_q['Body'].fillna('')
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(filtered_q['Body'])
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    #Replace NaN with an empty string
    filtered_q['Tags'] = filtered_q['Tags'].fillna('')
    #valid_filtered_q['Body'] = valid_filtered_q['Body'].fillna('')
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tags_matrix = tfidf.fit_transform(filtered_q['Tags'])
    item_dict ={}
    df = filtered_q.sort_values('post_indicies').reset_index()
    for i in range(df.shape[0]):
        item_dict[(df.loc[i,'post_indicies'])] = df.loc[i,'Id']
    # convert to csr matrix
    books_metadata_csr = csr_matrix(tfidf_matrix)
    tags_csr = csr_matrix(tags_matrix)
    books_metadata_csr = sparse.hstack((books_metadata_csr, tags_csr), format='csr')
    filtered_a.to_csv('filtered_a.csv')
    filtered_q.to_csv('filtered_q.csv')
    dataset = Dataset()
    dataset.fit((x for x in filtered_a.user_indicies.values),
                (x for x in filtered_a.post_indicies.values))
    dummies = range(max(filtered_a.user_indicies.values)+1, max(filtered_a.user_indicies.values)+100)
    dataset.fit_partial((x for x in dummies))   
    np.save('user_indicies.npy', filtered_a.user_indicies.values)
    np.save('post_indicies.npy', filtered_a.post_indicies.values)
    print(filtered_a.columns)
    print(filtered_a.dtypes)
    filtered_a.Score = filtered_a.Score.apply(int)
    (interactions, weights) = dataset.build_interactions(((x[0], x[1], x[2])
                                    for x in filtered_a[['user_indicies', 'post_indicies', 'Score']].values))
    sparse.save_npz("interactions.npz", interactions)
    sparse.save_npz("weights.npz", weights)
    sparse.save_npz("item_features.npz", books_metadata_csr)
    print('interections matrix shape: ', dataset.interactions_shape())
    model = LightFM(loss='warp',
                random_state=2016,
                learning_rate=0.90,
                no_components=150,
                #user_alpha=0.000005,
                item_alpha=0.000005
               )
    train, test = random_train_test_split(interactions, test_percentage=0.3, random_state = 1)
    train_weight, test_weight = random_train_test_split(weights, test_percentage=0.3, random_state = 1)
    print('Fitting Model:')
    model = model.fit(train, item_features = books_metadata_csr, sample_weight = train_weight,
         epochs=100, num_threads=8, verbose=True)
    with open('savefile.pickle', 'wb') as fle:
        pickle.dump(model, fle, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__": 
    main()