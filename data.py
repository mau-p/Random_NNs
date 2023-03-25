import pandas as pd
import numpy as np


def _get_data():
    '''
    Get data from csv, drop last row and suffle
    '''
    data = pd.read_csv('winequality.csv')
    data.drop(data.tail(1).index, inplace=True)
    data = data.sample(frac=1, random_state=50).reset_index(drop=True)

    return data

def _split_data(data: pd.DataFrame):
    '''
    Obtain an 70, 20, 10 split for train, val, test
    '''

    train = data.iloc[0:int(len(data)*0.7)]
    val = data.iloc[int((len(data)*0.7)):int(len(data)*0.9)]
    test = data.iloc[int((len(data)*0.9)):]

    return train, val, test


def data_pipeline():
    data = _get_data()
    train,val,test = _split_data(data)

