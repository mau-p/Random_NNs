import model
import random
from tqdm import tqdm
from keras.utils.np_utils import to_categorical
from collections import Counter
from operator import itemgetter

# Turn off annoying Tensorflow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0' 
import tensorflow as tf
import numpy as np

class Ensemble():
    def __init__(self, data, nn_to_train) -> None:
        self.data = data
        self.n_of_networks = nn_to_train
        self.trained_models = []
        self.create_networks()
        self.train_all_NNs()

    def create_networks(self):
        self.neural_networks = []

        for _ in range(self.n_of_networks):
            new_model = model.Model()
            self.neural_networks.append(new_model)
    
    def train_all_NNs(self):
        for network in self.neural_networks:
            print(f'--------Training Network: #{self.neural_networks.index(network)+1}')
            trained_model = network.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            trained_model = network.model.fit(x=self.data['train_x'], y=self.data['train_y'], validation_data=(self.data['val_x'], self.data['val_y']), batch_size=10, epochs=50)
            self.trained_models.append(trained_model)

    def make_prediction(self, x):
        preferences = []

        for network in self.trained_models:
            x = np.asarray(x)
            x = x.reshape(1,10,1)
            prediction = network.model.predict(x)[0]
            preferences.append(self.prediction_to_profile(prediction))
        
        return preferences

    def get_accuracy(self, data, voting_rules):
        accuracies = Counter()
        for vector, label in zip(data['test_x'], data['test_y']):
            profile = self.make_prediction(vector)
            for rule in voting_rules:
                result = rule(profile)
                if result == np.argmax(label):
                    accuracies.update({rule.__name__: 1})

        for rule in accuracies:
            accuracies[rule] /= len(data['test_x'])

        return accuracies
    
    def prediction_to_profile(self, prediction):
        preference = np.argsort(prediction)[::-1]
        return preference
