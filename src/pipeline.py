from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def get_pipeline(X):
    """
    Create preprocessing pipeline for numeric features
    """

    numeric_features = X.columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features)
        ],
        remainder="drop"
    )

    return preprocessor