from stackapi import StackAPI
import pandas as pd
import numpy as np
import time
import datetime
import json

def get_id(x):
    if type(x) == dict:
        return x['user_id']
    else:
        return x
    
# Open configs file  
config_file = '../data/APIData/api_params'
with open(config_file) as conf:
    configs = json.load(conf)
    
# Apply config file paramaters to create SITE
last_request = configs['last_request']
SITE = StackAPI('superuser', key="nIFln5DrNi7grh*o22xAIw((")
SITE.page_size = configs['page_size']
SITE.max_pages = configs['max_pages']
api_filter = configs['filter']

# Get the posts
request_time = int(time.time())
posts = SITE.fetch('posts', filter = api_filter, fromdate = last_request, todate = request_time)

# Transform post data to usable csv
full = pd.DataFrame(posts)
items = pd.DataFrame(full['items'].tolist())
items['owner'] = items['owner'].apply(get_id)
items['creation_date'] = pd.to_datetime(items['creation_date'], unit='s')
answered_questions = items[(items['post_type'] == "answer") & (items['score'] > 2)].post_id
items = items[~items['post_id'].isin(answered_questions)]

# Get tag data and merge with post data
tags = SITE.fetch('questions', filter='!BHMIb2uwAoY2iaeLXw*o8d5gWzG74D', fromdate = last_request, todate = request_time)
tags_data = pd.DataFrame(pd.DataFrame(tags)['items'].tolist())
api_data = items.merge(tags_data, how='left', left_on='post_id', right_on='question_id').drop('question_id', axis=1)

# Create timestamps for today and set data to that csv 
today = datetime.datetime.now()
unique_second = str(today.year) + str(today.month).zfill(2) + str(today.day) + str(today.hour).zfill(2) + str(today.minute).zfill(2) + str(today.second).zfill(2)
api_data.to_csv("../data/APIData/" + unique_second + ".csv")

# Update configs file
new_file = {
    "page_size": 100,
    "max_pages": 100,
    "last_request": request_time,
    "filter": "!0S2DU7n2**TqdL2snKOmnaNHL"
}
with open(config_file, 'w') as conv: 
     conv.write(json.dumps(new_file))
