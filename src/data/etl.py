import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET

def get_df(file_path):
    """Parses xml file and converts data to pandas file."""
    parsed = ET.parse(file_path)
    root = parsed.getroot()
    
    data = []

    for i, child in enumerate(root):
        data.append(child.attrib)
        
    # Turn into pandas DataFrame and set index to Id
    dfItem = pd.DataFrame.from_records(data).set_index('Id')
    return dfItem

def create_datasets(file, split_date):
    # Get data from file
    posts = get_df(file)
    split_date = eval(split_date)
    
    # Use only needed data and convert to right format
    relevant = posts[['PostTypeId', 'CreationDate', 'Title', 'Body', 'Tags', 'OwnerUserId', 'AnswerCount', 'ParentId']]
    relevant['CreationDate'] = pd.to_datetime(relevant['CreationDate'])
    
    # Split the data to include only answered data after a certain date
    useful_data = relevant[(relevant['AnswerCount'] != '0') & (relevant['CreationDate'] >= pd.datetime(2018, 1, 1))]
    
    # Split data into questions and answers
    data_q = useful_data[useful_data['PostTypeId'] == '1'].drop(['PostTypeId', 'ParentId', 'AnswerCount'], axis=1)
    data_a = useful_data[useful_data['PostTypeId'] == '2'].drop(['PostTypeId', 'AnswerCount', 'Tags'], axis=1)
    
    # Return train, validation, and test datasets
    return data_q, data_a

def main(configs):
    FILE = configs['file']
    SPLIT_DATE = configs['split_date']
    QUESTIONS = configs['questions_file']
    ANSWERS = configs['answers_file']
    data_q, data_a = create_datasets(FILE, SPLIT_DATE)
    data_q.to_csv(QUESTIONS)
    data_a.to_csv(ANSWERS)
    print('########### Data Created ###########')
    
    
if __name__ == "__main__":
    main(sys.argv)
    

    
