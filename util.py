
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model

def read_data():
    training_data = pd.read_csv('/Users/zhilianggong/Code/numerai/data/numerai_training_data.csv', header = 0)
    tour_data = pd.read_csv('/Users/zhilianggong/Code/numerai/data/numerai_tournament_data.csv', header = 0)

    features = [f for f in list(training_data) if 'feature' in f]
    X = training_data[features]
    Y = training_data['target']
    x_pred = tour_data[features]
    ids = tour_data['id']

    return X, Y, x_pred, ids

def write_prediction_csv(pred_array, ids):
    results_df = pd.DataFrame(data = { 'probability': pred_array })
    joined = pd.DataFrame(ids).join(results_df)
    joined.to_csv('/Users/zhilianggong/Code/numerai/predictions/predictions.csv', index = False)
