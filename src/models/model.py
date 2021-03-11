import pandas as pd
import numpy as np
import pickle
import os
import sys
from scipy import sparse
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k, reciprocal_rank

def mean_recommendation_user(model, interactions, item_features, user_id,  
                               item_dict, threshold=0, nrec_items=25, show=True):
    """ Generate recommendations for inputted user """
    # Get size of interactions object
    n_users, n_items = interactions.shape
    user_x = user_id 
    
    # If item features cause not baseline
    if item_features is not None:
        scores = pd.Series(model.predict(user_x, np.arange(n_items), item_features=item_features))
    else:
        scores = pd.Series(model.predict(user_x, np.arange(n_items)))
    
    # Get scores, sort them and return top nrec_items amount of recommendations
    scores.index = range(n_items)
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    scores = [x for x in scores]
    return_score_list = scores[0:nrec_items]
    
    # Filter through item dict
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    
    # Show recommendations
    if show == True:
        print ("User: " + str(user_id))
        print("\n Recommended Items:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + str(i))
            counter+=1
    
    # Store recommendations
    recs = pd.Series(return_score_list).apply(lambda x: item_dict[x])
 
    return scores, recs


def main(configs, baseline=False):
    # Get API data
    new = pd.read_csv(configs["api_data"])
    
    # Load in input from their file locations
    interactions = sparse.load_npz(configs["interactions"])
    user_id = configs["user_id"]
    threshold = configs["recommendation_threshold"]
    nrec = configs["num_recommendations"]

    
    # If running baselines during test:
    if baseline:
        # Load model and item dict and run mean_recommendation_user without item features
        item_features = None
        model = pickle.load(open(configs["baseline_model"], "rb"))
        print(model)
        item_dict = pickle.load(open(configs["baseline_item_dict"], 'rb'))
    else:
        # Load model and item dict and run mean_recommendation_user with item features
        item_features = sparse.load_npz(configs["item_features"])
        model = pickle.load(open(configs["model"], "rb"))
        print(model)
        item_dict = pickle.load(open(configs["item_dict"],"rb"))

        
    # Generate Recommendations for user
    scores, recs = mean_recommendation_user(model, interactions,  item_features, user_id, 
                               item_dict, threshold=threshold,nrec_items=nrec, show=True)
    
    # Save Recommendations
    recs.to_csv(configs["recommendations"] + str(user_id) + ".csv")
    
    
if __name__ == "__main__":
    main(sys.argv)

    
    
    