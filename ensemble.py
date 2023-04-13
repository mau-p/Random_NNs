import model
from keras.models import load_model
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Ensemble():
    def __init__(self, data=None, nn_to_train=0, models_to_load=[], batch_size=32) -> None:
        self.batch_size = batch_size
        self.data = data
        self.n_of_networks = nn_to_train
        self.trained_models = []
        if data:
            self.create_networks()
            self.train_all_NNs()
        elif models_to_load:
            self.set_trained_models(models_to_load)


    def set_trained_models(self, models_to_load):
        for model in models_to_load:
            print(f"--------Loading model: #{model.split('/')[-1].split('_')[-1]}")
            loaded_model = load_model(model)
            self.trained_models.append(loaded_model)


    def create_networks(self):
        self.neural_networks = []

        for _ in range(self.n_of_networks):
            new_model = model.Model()
            self.neural_networks.append(new_model)


    def train_all_NNs(self):
        for network in self.neural_networks:
            print(f'--------Training Network: #{self.neural_networks.index(network)+1}')
            trained_model = network.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            trained_model = network.model.fit(x=self.data['train_x'], y=self.data['train_y'], validation_data=(self.data['val_x'], self.data['val_y']), batch_size=self.batch_size, epochs=50)
            self.trained_models.append(trained_model)
        for network in self.trained_models:
            print(f'Saving model: #{self.trained_models.index(network)+1}')
            network.model.save(f'./models/model_{self.trained_models.index(network)+1}')
            self._plot_model(network.history, self.trained_models.index(network)+1)


    def make_profiles(self, data_x):
        profiles = [[] for _ in range(len(data_x))]

        for network in self.trained_models:
            if self.n_of_networks == 0:
                predictions = network.predict(data_x, batch_size=self.batch_size)
            else:
                predictions = network.model.predict(data_x, batch_size=self.batch_size)
            preferences = self.predictions_to_preferences(predictions)
            for i in range(len(preferences)):
                profiles[i].append(preferences[i])
        return profiles


    def get_accuracy(self, voting_rules, data=None):
        accuracies = Counter()
        if data:
            profiles = pd.DataFrame({"label": data['test_y'].argmax(axis=1), "profile": self.make_profiles(data['test_x'])})
            profiles.to_hdf('profiles.h5', key='df', mode='w')
        else:
            profiles = pd.read_hdf('profiles.h5', key='df')

        for rule in voting_rules:
            profiles[rule.__name__] = profiles.apply(lambda x: rule(x['profile']), axis=1)
            accuracy = (profiles[rule.__name__] == profiles['label']).astype(int).sum()
            accuracies.update({rule.__name__: (accuracy/len(profiles))})
        results = profiles.drop(columns=['profile'], axis=1)
        results.to_csv('results.csv', index=False)

        return accuracies
    
    
    def predictions_to_preferences(self, predictions):
        preferences = [list(np.argsort(prediction)[::-1]) for prediction in predictions]
        return preferences


    def _plot_model(self, network_history, index):
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5))
        fig.suptitle(f'Model {index}')
        ax1.plot(network_history['accuracy'], color='blue', label='train')
        ax1.plot(network_history['val_accuracy'], color='red', label='val')
        ax1.set_title('model accuracy')
        ax1.set(ylabel='accuracy', xlabel='epoch')
        ax1.legend(loc='upper right')

        ax2.plot(network_history['loss'], color='blue', label='train')
        ax2.plot(network_history['val_loss'], color='red', label='val')
        ax2.set_title('model loss')
        ax2.set(ylabel='loss', xlabel='epoch')
        ax2.legend(loc='upper right')

        fig.savefig(f'./models/model_{index}_acc_loss.png')
        fig.clf()