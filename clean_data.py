
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/customers.csv")
OUTPUT_PATH = Path("data/customers_cleaned.csv")


def clean_data():
    print("Starting data cleaning...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    print(f"Original dataset shape: {df.shape}")
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Only use "None" if missing values mean no internet service.
    df["internet_service"] = df["internet_service"].fillna("None")

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nCleaned file saved to: {OUTPUT_PATH.resolve()}")
    print("Data cleaning completed!")


if __name__ == "__main__":
    clean_data()
