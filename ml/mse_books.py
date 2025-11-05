from sklearn.metrics import mean_squared_error
import pandas as pd

from pathlib import Path
import time

data_dir = Path(__file__).parent / "../data"
input_path_book_parquet = data_dir / f"books/parquet/books.parquet"

        

def eval_books():
    df = pd.read_parquet(input_path_book_parquet)

    books = [
        "Killersports",
        "bet365",
        "pinnacle"
    ]
    mse = {}
    actual_mean = df["Spread_Actual"].mean()
    mean_baseline = [actual_mean] * len(df["Spread_Actual"])
    mse["baseline"] = mean_squared_error(df["Spread_Actual"], mean_baseline)
    for book in books:
        col = f"Spread_{book}"
        df_nonnull = df[["Spread_Actual", col]].dropna()
        mse[col] = mean_squared_error(df_nonnull["Spread_Actual"], df_nonnull[col])

    for measurement in mse.keys():
        print(f"{measurement} unscaled mse: {mse[measurement]}")
        print(f"{measurement} unscaled rmse: {mse[measurement]**(0.5)}\n")

if __name__ == "__main__":
    total_start = time.time()

    eval_books()

    total_end = time.time()
    print(f"Completed in {total_end - total_start:.2f}s")
