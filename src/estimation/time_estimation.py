import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class SAPEffortEstimator:
    """
    SAP Task List Effort Estimation
    Target: actual_duration
    """

    def __init__(self):
        # -----------------------------
        # 1. Configuration
        # -----------------------------
        import os

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.DATASET_PATH = os.path.join(
            BASE_DIR, "..", "..", "data", "effort_training_dataset.csv"
        )

        self.CATEGORICAL_COLS = [
            "operation_type",
            "equipment",
            "plant"
        ]

        self.NUMERICAL_COLS = [
            "planned_duration",
            "planned_work_quantity",
            "planned_material_qty"
        ]

        self.TARGET_COL = "actual_duration"

        self.models = {
            "LinearRegression": LinearRegression(),
            "Ridge": RidgeCV(alphas=[0.1, 1.0, 10.0]),
            "Lasso": LassoCV(cv=5),
            "ElasticNet": ElasticNetCV(cv=5),
            "RandomForest": RandomForestRegressor(
                n_estimators=300,
                random_state=42
            )
        }

        # -----------------------------
        # 2. Load & prepare data
        # -----------------------------
        self.load_data()
        self.build_pipeline()
        self.run_cross_validation()

    # -----------------------------
    # Data handling
    # -----------------------------
    def load_data(self):
        print("\nLoading SAP effort dataset...")
        self.df = pd.read_csv(self.DATASET_PATH)

        required_cols = (
            self.CATEGORICAL_COLS +
            self.NUMERICAL_COLS +
            [self.TARGET_COL]
        )

        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

        print("Dataset shape:", self.df.shape)
        print(self.df.head())

        self.X = self.df[self.CATEGORICAL_COLS + self.NUMERICAL_COLS]
        self.y = self.df[self.TARGET_COL]

    # -----------------------------
    # ML pipeline
    # -----------------------------
    def build_pipeline(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"),
                 self.CATEGORICAL_COLS),
                ("num", "passthrough", self.NUMERICAL_COLS),
            ]
        )

    # -----------------------------
    # Evaluation
    # -----------------------------
    def run_cross_validation(self):
        print("\nRunning 10-fold cross-validation...\n")

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        for model_name, model in self.models.items():
            rmses, maes, r2s = [], [], []

            pipeline = Pipeline(
                steps=[
                    ("preprocess", self.preprocessor),
                    ("model", model)
                ]
            )

            for train_idx, test_idx in kf.split(self.X):
                X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_test)

                rmses.append(
                    np.sqrt(mean_squared_error(y_test, preds))
                )
                maes.append(
                    mean_absolute_error(y_test, preds)
                )
                r2s.append(
                    r2_score(y_test, preds)
                )

            print(f"Model: {model_name}")
            print(f"  RMSE: {np.mean(rmses):.3f}")
            print(f"  MAE : {np.mean(maes):.3f}")
            print(f"  R²  : {np.mean(r2s):.3f}")
            print("-" * 40)


if __name__ == "__main__":
    SAPEffortEstimator()
