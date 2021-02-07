import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from ValidTestData import get_df
from ValidTestData import create_datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.metrics.pairwise import linear_kernel


def get_recommendations(title):
    # Get the index of the movie that matches the title
    idx = indices[title]
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()))
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:101]
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # Return the top 10 most similar questions
    return train_a[train_a['ParentId'].isin(train_q['Id'].iloc[movie_indices].values)]['OwnerUserId']






def main(): 
    FILE = '/Posts.xml'
    SPLIT_DATE = pd.datetime(2019, 9, 1)
    train_q, train_a, valid_q, valid_a, test_q, test_a = create_datasets('', 'Posts.xml', SPLIT_DATE)
    train_a = train_a.sort_values('Score', ascending=False)
    train_q['Id'] = train_q.index
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')
    #Replace NaN with an empty string
    train_q['Body'] = train_q['Body'].fillna('')
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(train_q['Body'])
    #Output the shape of tfidf_matrix
    tfidf_matrix.shape
    indices = pd.Series(train_q.index, index=train_q['Id']).drop_duplicates()
    acc = []
    for i in train_q.sample(frac = 0.002).Id.values:
        answers = set(train_a[train_a['ParentId'] == i]['OwnerUserId'].values)
        print('a: ', len(answers))
        rec = set(get_recommendations(i))
        print('r: ', len(rec))
        if len(rec.intersection(answers)) > 0:
            acc.append(1)
            print(np.mean(acc))
        else:
            acc.append(0)
    print(np.mean(acc))
    
if __name__ == "__main__":
    main()

