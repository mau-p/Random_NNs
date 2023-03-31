import pandas as pd
import random


def _get_data():
    data = pd.read_csv('wine.data', sep='\t', header=None, index_col=False)
    parsed_data = []
    for row in data.iterrows():
        vector = row[1].tolist()[0]
        vector = vector.split(',')
        vector = [float(x) for x in vector]
        label = vector[0]
        vector = vector[1:]
        parsed_data.append([vector, label])

    random.shuffle(parsed_data)
    return parsed_data

def _split_data(data):
    train = data[0:int(len(data)*0.7)]
    val = data[int((len(data)*0.7)):int(len(data)*0.9)]
    test = data[int((len(data)*0.9)):]

    train_x = [item[0] for item in train]
    train_y = [int(item[1]) for item in train]
    val_x = [item[0] for item in val]
    val_y = [int(item[1]) for item in val]
    test_x = [item[0] for item in test]
    test_y = [int(item[1]) for item in test]

    train_y = [_one_hot_encoding(item) for item in train_y]
    val_y = [_one_hot_encoding(item) for item in val_y]
    print(test_y)

    data = {'train_x': train_x, 'train_y': train_y, 'val_x': val_x, 'val_y': val_y, 'test_x': test_x, 'test_y': test_y}

    return data

def _one_hot_encoding(item):
    encoding = [0] * 3
    encoding[item-1] = 1
    return encoding


def data_pipeline():
    data = _get_data()
    return _split_data(data)
