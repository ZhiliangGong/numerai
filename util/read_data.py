
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model

def read_data():
    print('loading data...')
    training_data = pd.read_csv('/Users/zhilianggong/Code/numerai/data/numerai_training_data.csv', header = 0)
    tour_data = pd.read_csv('/Users/zhilianggong/Code/numerai/data/numerai_tournament_data.csv', header = 0)

    features = [f for f in list(training_data) if 'feature' in f]
    X = training_data[features]
    Y = training_data['target']
    x_pred = tour_data[features]
    ids = tour_data['id']

    return X, Y, x_pred, ids
