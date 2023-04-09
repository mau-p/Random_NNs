import model
import random
from tqdm import tqdm

# Turn off annoying Tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np

class Ensemble():
    def __init__(self, data, nn_to_train) -> None:
        self.data = data
        self.n_of_networks = nn_to_train
        self.create_networks()
        self.train_all_NNs()

    def create_networks(self):
        self.neural_networks = []

        for _ in range(self.n_of_networks):
            new_model = model.Model()
            self.neural_networks.append(new_model)
    
    def train_all_NNs(self):
        self.trained_models = []
        for network in self.neural_networks:
            print(f'--------Training Network: #{self.neural_networks.index(network)+1}')
            trained_model = network.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            trained_model = network.model.fit(x= self.data['train_x'], y=self.data['train_y'], validation_data=(self.data['val_x'], self.data['val_y']), batch_size=8, epochs=50)
            self.trained_models.append(trained_model)

    def make_prediction(self, x, voting_rule):
        preferences = []

        for network in self.trained_models:
            x = np.asarray(x)
            x = x.reshape(1,10,1)
            prediction = network.model.predict(x)[0]
            preference = self.prediction_to_profile(prediction)
            preferences.append(preference)
        return voting_rule(preferences)

    def get_accuracy(self, data, voting_rule):
        accuracy  = 0
        for vector, label in zip(tqdm(data['test_x']), data['test_y']):
            result = self.make_prediction(vector, voting_rule)
            if result == label:
                accuracy += 1

        accuracy /= len(data['test_x'])
        return accuracy
    
    def prediction_to_profile(self, prediction):
        preferences = []
        if prediction > 0.5:
            preferences = [1,0]
        elif prediction < 0.5:
            preferences = [0,1]
        else: # If the prediction is exactly 0.5, choose a random preference
            preferences = random.choice([[0,1], [1,0]])

        return preferences
