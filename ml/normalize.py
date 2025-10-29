import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

data_dir = Path(__file__).parent / "../data"
input_path_parquet = data_dir / f"rolling/parquet/gamesRolling.parquet"
output_path_train_X_parquet = data_dir / f"normalized/parquet/gamesTrainX.parquet"
output_path_train_X_csv = data_dir / f"normalized/csv/gamesTrainX.csv"
output_path_test_X_parquet = data_dir / f"normalized/parquet/gamesTestX.parquet"
output_path_test_X_csv = data_dir / f"normalized/csv/gamesTestX.csv"
output_path_train_y_parquet = data_dir / f"normalized/parquet/gamesTrainY.parquet"
output_path_train_y_csv = data_dir / f"normalized/csv/gamesTrainY.csv"
output_path_test_y_parquet = data_dir / f"normalized/parquet/gamesTestY.parquet"
output_path_test_y_csv = data_dir / f"normalized/csv/gamesTestY.csv"

output_path_train_X_parquet.parent.mkdir(parents=True, exist_ok=True)
output_path_train_X_csv.parent.mkdir(parents=True, exist_ok=True)
output_path_test_X_parquet.parent.mkdir(parents=True, exist_ok=True)
output_path_test_X_csv.parent.mkdir(parents=True, exist_ok=True)
output_path_train_y_parquet.parent.mkdir(parents=True, exist_ok=True)
output_path_train_y_csv.parent.mkdir(parents=True, exist_ok=True)
output_path_test_y_parquet.parent.mkdir(parents=True, exist_ok=True)
output_path_test_y_csv.parent.mkdir(parents=True, exist_ok=True)

def normalize():    
    df = pd.read_parquet(input_path_parquet)
    home_col = df["Home"]

    # Split into train and test
    X = df.copy()
    X = X.drop(columns=["Home"])
    X = X.drop(columns=["Result"])
    y = df["Result"]
    X_train, X_test, y_train, y_test, home_train, home_test = train_test_split(
        X, y, home_col, test_size=0.2, random_state=42, shuffle=True
    )

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
    
    
    # Parquet
    X_train_scaled.to_parquet(output_path_train_X_parquet, index=False)
    X_test_scaled.to_parquet(output_path_test_X_parquet, index=False)
    X_train_scaled.to_csv(output_path_train_X_csv, index=False)
    X_test_scaled.to_csv(output_path_test_X_csv, index=False)

    y_train_scaled.to_parquet(output_path_train_y_parquet, index=False)
    y_test_scaled.to_parquet(output_path_test_y_parquet, index=False)
    y_train_scaled.to_csv(output_path_train_y_csv, index=False)
    y_test_scaled.to_csv(output_path_test_y_csv, index=False)




    

def main():
    total_start = time.time()

    normalize()

    total_end = time.time()
    print(f"Data normalized in {total_end - total_start:.2f}s")


if __name__ == "__main__":
    main()