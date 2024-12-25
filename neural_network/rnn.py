import tensorflow as tf
from tensorflow.keras import layers, Model

class RNN:
    def __init__(self, input_shape, lstm_units=64, dense_units=32, activation='relu'):
        """
        Initialize the Recurrent Neural Network.
        :param input_shape: Shape of the sequential input (timesteps, features).
        :param lstm_units: Number of LSTM units.
        :param dense_units: Number of dense layer units after LSTM.
        :param activation: Activation function for the dense layer.
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.activation = activation
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build the RNN model.
        """
        inputs = layers.Input(shape=self.input_shape)
        x = layers.LSTM(self.lstm_units, return_sequences=False)(inputs)
        x = layers.Dense(self.dense_units, activation=self.activation)(x)
        return Model(inputs=inputs, outputs=x)
    
    def get_model(self):
        """
        Return the built RNN model.
        """
        return self.model
