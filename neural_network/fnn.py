import tensorflow as tf
from tensorflow.keras import layers, Model

class FNN:
    def __init__(self, input_dim, hidden_units=[64, 32], activation='relu'):
        """
        Initialize the Feedforward Neural Network.
        :param input_dim: Dimension of the input features.
        :param hidden_units: List of hidden layer units.
        :param activation: Activation function for the hidden layers.
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.activation = activation
        self.model = self._build_model()
    
    def _build_model(self):
        """
        Build the FNN model.
        """
        inputs = layers.Input(shape=(self.input_dim,))
        x = inputs
        for units in self.hidden_units:
            x = layers.Dense(units, activation=self.activation)(x)
        return Model(inputs=inputs, outputs=x)
    
    def get_model(self):
        """
        Return the built FNN model.
        """
        return self.model
