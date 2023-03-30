import random

# Turn off annoying Tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.layers import Dense, Input, Dropout, Conv1D, Flatten, AveragePooling1D, MaxPooling1D
from keras import Sequential, callbacks

class Model:
    def __init__(self) -> None:
        self.activations = ['sigmoid', 'relu', 'tanh']

        # Select a random number of hidden layers
        self.rand_hidden_layers = random.randrange(1, 20, 1)
        print(f"rand_hidden_layers: {self.rand_hidden_layers}")
        self.dropout_rate = random.uniform(0,0.5)
        self.generate_random_model()
        self.results = None

    def generate_random_model(self):
        self.model = Sequential()
        self.model.add(Input(shape=(13,1)))

        # Select a random activation function
        act = random.choice(self.activations)
        for _ in range(self.rand_hidden_layers):
            self.model.add(Dense(units=13, activation=act))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Flatten())
        self.model.add(Dense(units=3, activation='softmax'))
