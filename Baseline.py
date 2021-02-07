import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from ValidTestData import get_df
from ValidTestData import create_datasets




def baseline(q, a):
    popular = set(list(a.OwnerUserId.value_counts().index)[:100])
    accr = []
    for i in list(q.index):
        answers = set(a[a.ParentId == i].OwnerUserId.values)
        if len(answers.intersection(popular)) > 0:
            accr.append(1)
        else:
            accr.append(0)
    return np.mean(accr)


def main(): 
    FILE = '/Posts.xml'
    SPLIT_DATE = pd.datetime(2019, 9, 1)
    train_q, train_a, valid_q, valid_a, test_q, test_a = create_datasets('', 'Posts.xml', SPLIT_DATE)
    popular = set(list(train_a.OwnerUserId.value_counts().index)[:100])
    baseline_scores = dict()
    valid_accr = baseline(valid_q, valid_a)
    print('Baseline validation accuracy: ', valid_accr)
    baseline_scores['validation accuracy'] = valid_accr
    test_accr = baseline(test_q, test_a)
    print('Baseline test accuracy: ', test_accr)
    baseline_scores['test accuracy'] = test_accr
    train_accr = baseline(train_q, train_a)
    print('train accuracy: ', train_accr)
    baseline_scores['train accuracy'] = train_accr
    scores = pd.Series(data = d)
    scores.to_csv('BaselineScores.csv')
    
if __name__ == "__main__":
    main()

    