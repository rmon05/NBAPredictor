import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fnn import FNN # change import path as needed

from keras import layers, models, optimizers


def process_and_train(file_path):
    """
    Processes the data from a single large CSV file and trains the FNN model.
    """
    # Load data
    data = pd.read_csv(file_path, header=0)
    X = data.iloc[:, 1:].values # Features
    y = data.iloc[:, 0].values # Label (won)

    # Convert all data to numeric, coercing errors to NaN
    X = pd.DataFrame(X)
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')


    # Split and normalize
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit and transform training data
    X_test = scaler.transform(X_test)        # Only transform the test data


    
    # MODEL IS NOT EFFECTIVE RIGHT NOW, FIGURE OUT WHY AND FIX IT

    # # Initialize and train the FNN model
    # input_shape = X_train.shape[1]  # The number of features
    # fnn_model = FNN(input_dim=input_shape)  # Initialize the FNN model
    # fnn_model.compile()  # Compiles the model with the default settings (e.g., Adam optimizer)

    # # Train the model
    # fnn_model.train(X_train, y_train, epochs=10, batch_size=32)  # Training with only training data

    # # Evaluate the model
    # loss, accuracy = fnn_model.evaluate(X_test, y_test)  # Evaluate on test data
    # print(f"Model Evaluation -> Loss: {loss}, Accuracy: {accuracy}")


    # TEST FNN
    ann = models.Sequential()
    ann.add(layers.InputLayer(shape=(X_train.shape[1],)))
    ann.add(layers.Dense(units=16, activation="relu"))
    ann.add(layers.Dense(units=8, activation="relu"))
    ann.add(layers.Dense(units=1, activation="sigmoid"))

    optimizer = optimizers.Adam(learning_rate=0.001, clipvalue=1.0)
    ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    ann.fit(X_train, y_train, batch_size=32, epochs=15)

    loss, accuracy = ann.evaluate(X_test, y_test)
    print(f"Model Evaluation on Test -> Loss: {loss}, Accuracy: {accuracy}")


if __name__ == "__main__":
    file_path = "../data/processed/all_processed_data.csv" 
    process_and_train(file_path)
