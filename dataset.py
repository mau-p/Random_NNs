import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.utils.np_utils import to_categorical

def _get_data():
    data = pd.read_csv('dataset.csv')
    return data

def _split_data(data):
    train, val, test = np.split(data.sample(frac=1, random_state=42), [int(.8*len(data)), int(.9*len(data))])

    train_x = train[train.columns[:-1]]
    train_y = to_categorical(train[train.columns[-1]])
    val_x = val[val.columns[:-1]]
    val_y = to_categorical(val[val.columns[-1]])
    test_x = test[test.columns[:-1]]
    test_y = to_categorical(test[test.columns[-1]])

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)

    data = {'train_x': train_x, 'train_y': train_y, 'val_x': val_x, 'val_y': val_y, 'test_x': test_x, 'test_y': test_y}

    return data

def data_pipeline():
    data = _get_data()
    return _split_data(data)
