import model
import tensorflow as tf
import social_choice

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
            print(f'--------Training Network: #{self.neural_networks.index(network)}')
            trained_model = network.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=[tf.keras.metrics.Accuracy()])
            trained_model = network.model.fit(x= self.data['train_x'], y=self.data['train_y'], validation_data=(self.data['val_x'], self.data['val_y']))
            self.trained_models.append(trained_model)

    
    def make_prediction(self, x):
        predictions = []

        for network in self.trained_networks:
            prediction = network.model.predict(x)
            print(prediction)