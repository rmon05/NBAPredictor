import pandas as pd
from sklearn.preprocessing import StandardScaler

def normalize(X_train, X_test, y_train, y_test, home_train, home_test):
    # Z scale everything else besides Home which is binary feature
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Add Home back unscaled
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_train_scaled["Home"] = home_train.values
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    X_test_scaled["Home"] = home_test.values

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1))
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1,1))
    y_train_scaled = pd.DataFrame(y_train_scaled, columns=["Result"])
    y_test_scaled  = pd.DataFrame(y_test_scaled, columns=["Result"])

    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
