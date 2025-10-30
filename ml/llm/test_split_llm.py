import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time

data_dir = Path(__file__).parent / "../data"
input_path_parquet = data_dir / f"rolling/parquet/gamesRolling.parquet"
output_path_train_X_parquet = data_dir / f"llm/parquet/gamesTrainX.parquet"
output_path_train_X_csv = data_dir / f"llm/csv/gamesTrainX.csv"
output_path_test_X_parquet = data_dir / f"llm/parquet/gamesTestX.parquet"
output_path_test_X_csv = data_dir / f"llm/csv/gamesTestX.csv"
output_path_train_y_parquet = data_dir / f"llm/parquet/gamesTrainY.parquet"
output_path_train_y_csv = data_dir / f"llm/csv/gamesTrainY.csv"
output_path_test_y_parquet = data_dir / f"llm/parquet/gamesTestY.parquet"
output_path_test_y_csv = data_dir / f"llm/csv/gamesTestY.csv"

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
    X = X.drop(columns=["Result"])
    y = df["Result"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    

    X_train_scaled = X_train
    X_test_scaled = X_test
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    y_train_scaled = y_train
    y_test_scaled = y_test
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
    print(f"Data split for llm in {total_end - total_start:.2f}s")


if __name__ == "__main__":
    main()