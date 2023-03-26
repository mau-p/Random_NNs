import random
from keras.layers import Dense, Input, Dropout, Conv1D, Flatten, AveragePooling1D, MaxPooling1D
from keras import Sequential, callbacks

class Model:
    def __init__(self) -> None:
        self.rand_hidden_layers = random.randrange(2)
        self.dropout_rate = random.uniform(0,0.5)
        self.generate_random_model()
        self.results = None

    def generate_random_model(self):
        self.model = Sequential()
        self.model.add(Input(shape=(13,1)))
        for _ in range(self.rand_hidden_layers):
            self.model.add(Dense(units=15, activation='sigmoid'))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Flatten())
        self.model.add(Dense(units=3, activation='softmax'))
