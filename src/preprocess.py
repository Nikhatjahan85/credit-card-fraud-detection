import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(path)
    return df


def split_xy(df: pd.DataFrame):
    """
    Split dataset into features (X) and target (y)
    """
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain 'Class' column")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    return X, y


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic cleaning (optional but good practice)
    """
    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values (if any)
    df = df.fillna(0)

    return df