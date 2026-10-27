
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/customers_cleaned.csv")

REQUIRED_COLUMNS = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract",
    "internet_service",
    "churn",
]


def validate_data(file_path=DATA_PATH):
    file_path = Path(file_path)

    # 1. Check whether the dataset exists
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    # 2. Load the dataset
    df = pd.read_csv(file_path, keep_default_na=False)
    
    # Reject actual missing values and blank cells before numeric checks.
    has_missing_values = df.isna().any().any()

    has_blank_values = (
        df.astype("string")
        .apply(lambda column: column.str.strip().eq(""))
        .any()
        .any()
    )

    if has_missing_values or has_blank_values:
        raise ValueError("Data contains missing values")


    # 3. Check whether the dataset is empty
    if df.empty:
        raise ValueError("Dataset is empty")

    # 4. Check required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # 5. Check for missing values and display affected columns
    missing_values = df[REQUIRED_COLUMNS].isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        print("Missing values found in the following columns:")
        print(missing_values)

        # Display rows containing missing values
        print("\nRows containing missing values:")
        print(df.loc[ df[REQUIRED_COLUMNS].isnull().any(axis=1) ])

        raise ValueError(
            "Dataset contains missing values. "
            "Please clean the data before validation."
        )

    # 6. Validate churn values
    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Churn must contain only 0 and 1")

    # 7. Validate tenure
    if (df["tenure"] < 0).any():
        raise ValueError("Tenure cannot be negative")

    # 8. Validate monthly charges
    if (df["monthly_charges"] < 0).any():
        raise ValueError("Monthly charges cannot be negative")

    # 9. Validate support calls
    if (df["support_calls"] < 0).any():
        raise ValueError("Support calls cannot be negative")

    # 10. Validate contract types
    allowed_contracts = {
        "Monthly",
        "One Year",
        "Two Year",
    }

    if not set(df["contract"].unique()).issubset(
        allowed_contracts
    ):
        raise ValueError("Unexpected contract type")

    # 11. Validate internet service types
    allowed_internet_services = {
        "DSL",
        "Fiber",
        "None",
    }

    if not set(df["internet_service"].unique()).issubset(
        allowed_internet_services
    ):
        raise ValueError("Unexpected internet service")

    # 12. Ensure both churn classes exist
    if df["churn"].nunique() != 2:
        raise ValueError(
            "Target must contain both churn classes (0 and 1)"
        )

    # 13. Validation successful
    print(f"Data validation passed: {len(df)} rows")

    return df


if __name__ == "__main__":
    validate_data()
