#!/usr/bin/env python

from os import listdir, path, makedirs
import sys
import json
from src.data import etl
from src.models import CosinePostModel, CosineUserModel
from src.baselines import baseline


def main(targets):
    """ Runs data pipeline to parse all the data into these folders and turn movie title data into a csv"""

    if targets == 'test':
        filepath = 'config/test_params.json'
        with open(filepath) as file:
            configs = json.load(file)

        # etl.main(configs)
        print('########### Post Based Model ###########')
        CosinePostModel.main(configs)
        print('')
        print('########### User Based Model ###########')
        CosineUserModel.main(configs)
        
        print("####################")
        baseline.main(configs)
        print("####################")

    if targets == 'data' or targets == 'all':
        filepath = 'config/etl_params.json'
        with open(filepath) as file:
            configs = json.load(file)
        
        etl.main(configs)
        
    if targets == 'models' or targets == 'all':
        filepath = 'config/etl_params.json'
        with open(filepath) as file:
            configs = json.load(file)
            
        print('########### Post Based Model ###########')
        CosinePostModel.main(configs)
        print('')
        print('########### User Based Model ###########')
        CosineUserModel.main(configs)
        
        
    if targets == 'baselines' or targets == 'all':
        filepath = 'config/etl_params.json'
        with open(filepath) as file:
            configs = json.load(file)        
        
        print('####################')
        baseline.main(configs)
        print('####################')

    return None


if __name__ == '__main__':
    targets = sys.argv[1]
    main(targets)