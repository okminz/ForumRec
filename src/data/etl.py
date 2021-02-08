import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

def get_df(file_path):
    parsed = ET.parse(file_path)
    root = parsed.getroot()
    
    data = []

    for i, child in enumerate(root):
        data.append(child.attrib)
        
    dfItem = pd.DataFrame.from_records(data).set_index('Id')
    return dfItem

def create_datasets(file, split_date):
    # Get data from file
    posts = get_df(file)
    split_date = eval(split_date)
    
    # Use only needed data and convert to right format
    relevant = posts[['PostTypeId', 'CreationDate', 'Score', 'Body', 'Tags', 'OwnerUserId', 'AnswerCount', 'ParentId']]
    relevant['CreationDate'] = pd.to_datetime(relevant['CreationDate'])
    answered_data = relevant[relevant['AnswerCount'] != '0']
    
    # Split into training and evaluating data
    train_data = answered_data[answered_data['CreationDate'] < split_date]
    evaluation_data = answered_data[answered_data['CreationDate'] >= split_date]
    
    # Get questions from evaluation data and split in half for validation and testing
    question_ids = np.array(evaluation_data[evaluation_data['PostTypeId'] == '1'].index)
    np.random.shuffle(question_ids)
    valid_questions_index = question_ids[:question_ids.size//2]
    test_questions_index = question_ids[question_ids.size//2:]
    
    # Split validation and test datasets into questions and the answers to those questions
    valid_questions = evaluation_data.loc[valid_questions_index]
    valid_answers = evaluation_data[evaluation_data['ParentId'].isin(valid_questions_index)]
    test_questions = evaluation_data.loc[test_questions_index]
    test_answers = evaluation_data[evaluation_data['ParentId'].isin(test_questions_index)]
    
    # Split training data, and clean up training, validation, and testing datasets
    train_q = train_data[train_data['PostTypeId'] == '1'].drop(['PostTypeId', 'ParentId', 'AnswerCount'], axis=1)
    train_a = train_data[train_data['PostTypeId'] == '2'].drop(['PostTypeId', 'AnswerCount', 'Tags'], axis=1)
    valid_q  = valid_questions.drop(['CreationDate', 'PostTypeId', 'ParentId', 'AnswerCount'], axis=1)
    valid_a = valid_answers.drop(['CreationDate', 'PostTypeId', 'AnswerCount', 'Tags'], axis=1)
    test_q = test_questions.drop(['CreationDate', 'PostTypeId', 'ParentId', 'AnswerCount'], axis=1)
    test_a = test_answers.drop(['CreationDate', 'PostTypeId', 'AnswerCount', 'Tags'], axis=1)
    
    # Return train, validation, and test datasets
    return train_q, train_a, valid_q, valid_a, test_q, test_a

def main(configs):
    FILE = configs['file']
    SPLIT_DATE = configs['split_date']
    OUTPUT = configs['output']
    train_q, train_a, valid_q, valid_a, test_q, test_a = create_datasets(FILE, SPLIT_DATE)
    train_q.to_csv(OUTPUT + '/TrainQuestions.csv')
    train_a.to_csv(OUTPUT + '/TrainAnswers.csv')
    valid_a.to_csv(OUTPUT + '/ValidQuestions.csv')
    valid_q.to_csv(OUTPUT + '/ValidAnswers.csv')
    test_a.to_csv(OUTPUT + '/TestQuestions.csv')
    test_q.to_csv(OUTPUT + '/TestAnswers.csv')
    print('########### Data Created ###########')
    
    
if __name__ == "__main__":
    main(sys.argv)
    