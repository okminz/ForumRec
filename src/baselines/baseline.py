import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET




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


def main(configs):
    OUTPUT = configs['output']
    train_q = pd.read_csv(OUTPUT + '/TrainQuestions.csv')
    train_a = pd.read_csv(OUTPUT + '/TrainAnswers.csv')
    valid_a = pd.read_csv(OUTPUT + '/ValidQuestions.csv')
    valid_q = pd.read_csv(OUTPUT + '/ValidAnswers.csv')
    test_a = pd.read_csv(OUTPUT + '/TestQuestions.csv')
    test_q = pd.read_csv(OUTPUT + '/TestAnswers.csv')
    
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
    
if __name__ == "__main__":
    main()

    