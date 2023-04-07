import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

def _get_data():
    data = pd.read_csv('winequality-white.csv', sep='\t', header=None, index_col=False)
    parsed_data = []
    skip = True
    for row in data.iterrows():
        if skip:
            skip = False
            continue
        vector = row[1].tolist()[0]
        vector = vector.split(';')
        vector = [float(x) for x in vector]
        label = vector[11]

        if label <= 5:
            label = 0
        else:
            label = 1
            
        vector = vector[0:10]
        parsed_data.append([vector, label])

    random.shuffle(parsed_data)
    return parsed_data

def _split_data(data):
    train = data[0:int(len(data)*0.8)]
    val = data[int((len(data)*0.8)):int(len(data)*0.9)]
    test = data[int((len(data)*0.9)):]

    train_x = [item[0] for item in train]
    train_y = np.array([int(item[1]) for item in train])
    val_x = [item[0] for item in val]
    val_y = np.array([int(item[1]) for item in val])
    test_x = [item[0] for item in test]
    test_y = np.array([int(item[1]) for item in test])

    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)

    data = {'train_x': train_x, 'train_y': train_y, 'val_x': val_x, 'val_y': val_y, 'test_x': test_x, 'test_y': test_y}

    return data

def data_pipeline():
    data = _get_data()
    return _split_data(data)
