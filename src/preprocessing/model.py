import tensorflow as tf
from tensorflow.keras import layers


class SimpleLSTMModel(tf.keras.Model):
    def __init__(self, n_lstm_units = 64, n_dense_units = 32, n_features=None, n_target_variables=None):
        super(SimpleLSTMModel, self).__init__()

        self.lstm = layers.LSTM(n_lstm_units, activation='relu', return_sequences=False)
        self.dense1 = layers.Dense(n_dense_units, activation='relu')
        self.dense2 = layers.Dense(n_target_variables)

        self.input_shape = None

        def call(self, inputs):
            x = self.lstm(inputs)
            x = self.dense1(x)
            return self.dense2(x)

        def build(self,input_shape):
            self.input_shape = input_shape
            super(SimpleLSTMModel, self).build(input_shape)