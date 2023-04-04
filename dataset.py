import pandas as pd
import random
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

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
        if (label <= 5):
            label = 1
        else:
            label = 2
        vector = vector[0:10]
        parsed_data.append([vector, label])

    random.shuffle(parsed_data)
    return parsed_data

def _split_data(data):
    train = data[0:int(len(data)*0.8)]
    val = data[int((len(data)*0.8)):int(len(data)*0.9)]
    test = data[int((len(data)*0.9)):]

    train_x = [item[0] for item in train]
    train_y = [int(item[1]) for item in train]
    val_x = [item[0] for item in val]
    val_y = [int(item[1]) for item in val]
    test_x = [item[0] for item in test]
    test_y = [int(item[1]) for item in test]
    
    count = [train_y.count(i) for i in range(1, 3)]
    print(f'Count of scores before resampling: {count}')

    sm = SMOTE(random_state=42)
    train_x, train_y = sm.fit_resample(train_x, train_y)

    #TODO: tweak the sampling strategies, perhaps with combination of SMOTE and undersampling?

    count = [train_y.count(i) for i in range(1, 3)]
    print(f'Count of scores after resampling: {count}')

    train_y = [_one_hot_encoding(item) for item in train_y]
    val_y = [_one_hot_encoding(item) for item in val_y]

    data = {'train_x': train_x, 'train_y': train_y, 'val_x': val_x, 'val_y': val_y, 'test_x': test_x, 'test_y': test_y}

    return data

def _one_hot_encoding(item):
    encoding = [0] * 2
    encoding[item-1] = 1
    return encoding


def data_pipeline():
    data = _get_data()
    return _split_data(data)
