import os
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class SAPManpowerEstimator:
    """
    Chained manpower estimation using predicted duration.
    """

    def __init__(self):
        # -----------------------------
        # Path handling (robust)
        # -----------------------------
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATASET_PATH = os.path.join(
            BASE_DIR, "..", "..", "datasets", "sap_effort_dataset.csv"
        )

        # -----------------------------
        # Column definitions
        # -----------------------------
        self.CATEGORICAL_COLS = [
            "operation_type",
            "equipment",
            "plant"
        ]

        self.NUMERICAL_COLS = [
            "planned_work_quantity",
            "predicted_duration"
        ]

        self.TARGET_COL = "actual_work_quantity"

        # -----------------------------
        # Run pipeline
        # -----------------------------
        self.load_data()
        self.build_pipeline()
        self.run_cross_validation()

    def load_data(self):
        print("\nLoading SAP dataset for manpower estimation...")
        df = pd.read_csv(self.DATASET_PATH)

        # Simulate predicted duration (as output of Step 5 model)
        # In real system, this comes from the time estimator
        df["predicted_duration"] = (
            df["planned_duration"]
            * np.random.uniform(0.9, 1.1, size=len(df))
        )

        self.X = df[
            self.CATEGORICAL_COLS +
            self.NUMERICAL_COLS
        ]

        self.y = df[self.TARGET_COL]

        print("Dataset shape:", df.shape)
        print(df.head())

    def build_pipeline(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"),
                 self.CATEGORICAL_COLS),
                ("num", "passthrough",
                 self.NUMERICAL_COLS),
            ]
        )

        self.model = Pipeline(
            steps=[
                ("preprocess", self.preprocessor),
                ("regressor", LinearRegression())
            ]
        )

    def run_cross_validation(self):
        print("\nRunning manpower estimation (10-fold CV)\n")

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        maes, rmses, r2s = [], [], []

        for train_idx, test_idx in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)

            maes.append(mean_absolute_error(y_test, preds))
            rmses.append(np.sqrt(mean_squared_error(y_test, preds)))
            r2s.append(r2_score(y_test, preds))

        print(f"MAE : {np.mean(maes):.3f}")
        print(f"RMSE: {np.mean(rmses):.3f}")
        print(f"R²  : {np.mean(r2s):.3f}")


if __name__ == "__main__":
    SAPManpowerEstimator()
