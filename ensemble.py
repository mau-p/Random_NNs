import model

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
            trained_model = network.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Accuracy()])
            trained_model = network.model.fit(x= self.data['train_x'], y=self.data['train_y'], validation_data=(self.data['val_x'], self.data['val_y']), batch_size= 4)
            self.trained_models.append(trained_model)

    def make_prediction(self, x, voting_rule):
        preferences = []

        for network in self.trained_models:
            x = np.asarray(x)
            x = x.reshape(1,13,1)
            prediction = network.model.predict(x)[0]
            preference = self.prediction_to_profile(prediction)
            print(f"preferences for network {self.trained_models.index(network)+1}: {preference}")
            preferences.append(preference)

        return voting_rule(preferences)

    def get_accuracy(self, data, voting_rule):
        accuracy  = 0
        for vector, label in zip(data['test_x'], data['test_y']):
            result = self.make_prediction(vector, voting_rule)
            if result == label:
                accuracy += 1

        accuracy /= len(data['test_x'])
        return accuracy
    
    def prediction_to_profile(self, prediction):
        preferences = []
        for _ in prediction:
            best_vote = np.argmax(prediction)
            preferences.append(best_vote)
            prediction[best_vote] = -1

        return [x+1 for x in preferences]
