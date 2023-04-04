import random

# Turn off annoying Tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.layers import Dense, Input, Dropout, Flatten
from keras.regularizers import l2
from keras import Sequential

class Model:
    def __init__(self) -> None:
        # Select a random number of hidden layers
        self.rand_hidden_layers = random.randrange(1, 5, 1)
        print(f"rand_hidden_layers: {self.rand_hidden_layers}")
        self.dropout_rate = random.uniform(0,0.5)
        self.generate_random_model()
        self.results = None

    def generate_random_model(self):
        self.model = Sequential()
        self.model.add(Dense(units=10, input_shape=(10,1), activation='relu'))

        for _ in range(self.rand_hidden_layers):
            self.model.add(Dense(units=10, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
            
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Flatten())
        self.model.add(Dense(units=10, activation='softmax'))
