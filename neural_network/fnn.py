import tensorflow as tf
from keras import layers, models

class FNN:
    def __init__(self, input_dim, hidden_layers=[64, 64, 32], output_dim=1, activation='relu', output_activation='sigmoid'):
        """
        Initializes the FNN model.

        Args:
        - input_dim: Number of input features.
        - hidden_layers: List specifying the number of neurons in each hidden layer.
        - output_dim: Number of output neurons.
        - activation: Activation function for hidden layers.
        - output_activation: Activation function for the output layer.
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.activation = activation
        self.output_activation = output_activation
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds the FNN model.
        """
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(self.input_dim,)))
        # Add hidden layers
        for units in self.hidden_layers:
            model.add(layers.Dense(units, activation=self.activation))
        # Add output layer
        model.add(layers.Dense(self.output_dim, activation=self.output_activation))
        return model

    def compile(self, optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']):
        """
        Compiles the FNN model.

        Args:
        - optimizer: Optimization algorithm.
        - loss: Loss function.
        - metrics: List of metrics to evaluate during training.
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train(self, X_train, y_train, batch_size=32, epochs=50, validation_split=0.2):
        """
        Trains the FNN model.

        Args:
        - X_train: Training data.
        - y_train: Labels for the training data.
        - batch_size: Batch size.
        - epochs: Number of epochs.
        - validation_split: Fraction of training data to use for validation.
        """
        return self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the FNN model on test data.

        Args:
        - X_test: Test data.
        - y_test: Labels for the test data.
        """
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        """
        Makes predictions using the trained FNN model.

        Args:
        - X: Input data.
        """
        return self.model.predict(X)

    def summary(self):
        """
        Prints a summary of the FNN model.
        """
        self.model.summary()
