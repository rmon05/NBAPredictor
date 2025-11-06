from sklearn.metrics import mean_squared_error
import numpy as np
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


def eval_edge():
    df = pd.read_parquet(input_path_book_parquet)

    # Check placer edge on book
    placer = "Killersports"
    book = "bet365"
    bankroll = float(1000)
    post_vig_return = 1.85
    for _, row in df.iterrows():
        if row[f"Spread_{placer}"] is None or row[f"Spread_{book}"] is None:
            continue

        if float(row[f"Spread_{placer}"]) == row[f"Spread_{book}"]:
            continue
        
        unit_size = bankroll/float(100)
        placer_diff = abs(row[f"Spread_{placer}"]-row["Spread_Actual"])
        book_diff = abs(row[f"Spread_{book}"]-row["Spread_Actual"])
        bankroll -= unit_size
        if placer_diff < book_diff:
            bankroll += post_vig_return*unit_size
    print(f"Taking {placer} spread against book {book} yields accumulated bankroll {bankroll:.3f}")

if __name__ == "__main__":
    total_start = time.time()

    eval_books()
    eval_edge()

    total_end = time.time()
    print(f"Completed in {total_end - total_start:.2f}s")
