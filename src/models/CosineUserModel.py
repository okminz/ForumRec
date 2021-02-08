import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn.metrics.pairwise import linear_kernel


def user_recommendations(title, filtered_a, filtered_q, tfidf_matrix):
    # Get the index of the movie that matches the title
    indices = pd.Series(filtered_q.index, index=filtered_q['Id']).drop_duplicates()
    filtered_a = filtered_a[filtered_a.ParentId != title]
    idx = indices[title]
    sim_scores = list(enumerate(linear_kernel(tfidf_matrix[idx], tfidf_matrix)))
    sim_idx = [x[0] for x in sim_scores]
    sim_vals = [x[1] for x in sim_scores]
    a = pd.Series(sim_vals, index=sim_idx)
  
    def find_sim(x):
        return a[indices[x]]
    
    sim_col = filtered_a['ParentId'].apply(find_sim)
    filtered_a['sims'] = sim_col
    #df = filtered_a[filtered_a.ParentId != title][['OwnerUserId', 'sims']]
    df = filtered_a[['OwnerUserId', 'sims']]
    grouped_sim = df.groupby('OwnerUserId').mean()
    res = grouped_sim.sort_values(ascending=False, by = 'sims')
    #print(list(res.sims)[:5])
    return list(grouped_sim.sort_values(ascending=False, by = 'sims').index[:100])



def main(configs):
    OUTPUT = configs['output']
    train_q = pd.read_csv(OUTPUT + '/TrainQuestions.csv')
    train_a = pd.read_csv(OUTPUT + '/TrainAnswers.csv')
    valid_a = pd.read_csv(OUTPUT + '/ValidQuestions.csv')
    valid_q = pd.read_csv(OUTPUT + '/ValidAnswers.csv')
    test_a = pd.read_csv(OUTPUT + '/TestQuestions.csv')
    test_q = pd.read_csv(OUTPUT + '/TestAnswers.csv')
    
    train_a = train_a.sort_values('Score', ascending=False)
    train_q['Id'] = train_q.index
    train_a['Id'] = train_a.index
    grouped_a = train_a.groupby('OwnerUserId').count()
    print(grouped_a.shape)
    grouped_a = grouped_a[grouped_a.Id > 49]
    print(grouped_a.shape)
    grouped_a['OwnerUserId'] = grouped_a.index
    users = list(grouped_a.OwnerUserId)
    print(len(users))

    print('original answers data shape: ',train_a.shape)
    filtered_a = train_a[train_a.OwnerUserId.isin(users)]
    print('filtered answers shape: ', filtered_a.shape)

    print('original questions shape: ', train_q.shape)
    filtered_q = train_q[train_q.Id.isin(filtered_a.ParentId)]
    print('filtered questions shape: ', filtered_q.shape)
    
    #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')
    #Replace NaN with an empty string
    filtered_q['Body'] = filtered_q['Body'].fillna('')
    #Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(filtered_q['Body'])
    #Output the shape of tfidf_matrix
    print(tfidf_matrix.shape)
    #indices = pd.Series(train_q.index, index=train_q['Id']).drop_duplicates()
    acc = []
    for i in filtered_q.sample(frac = 0.002).Id.values[:100]:
        answers = set(filtered_a[filtered_a['ParentId'] == i]['OwnerUserId'].values)
        rec = set(user_recommendations(i, filtered_a, filtered_q, tfidf_matrix))
        #print(answers)
        if len(acc)%10 == 0 and len(acc) > 0:
            print(len(acc), 'acc: ', np.mean(acc))
        if len(rec.intersection(answers)) > 0:
            acc.append(1)
            print(np.mean(acc))
            print(len(rec.intersection(answers)))
        else:
            acc.append(0)
    print('Accuracy: ', np.mean(acc))
    
if __name__ == "__main__":
    main(sys.argv)

