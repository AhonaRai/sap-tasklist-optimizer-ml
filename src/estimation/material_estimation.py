import os
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


class SAPMaterialEstimator:
    """
    Stage 7: Material requirement estimation (hybrid).
    """

    def __init__(self):
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
            "predicted_duration",
            "predicted_manpower"
        ]

        self.load_data()
        self.build_pipeline()
        self.run_cross_validation()

    # -----------------------------
    # Data preparation
    # -----------------------------
    def load_data(self):
        print("\nLoading SAP dataset for material estimation...")
        df = pd.read_csv(self.DATASET_PATH)

        # Simulate upstream predictions
        df["predicted_duration"] = (
            df["planned_duration"] * np.random.uniform(0.9, 1.1, len(df))
        )

        df["predicted_manpower"] = (
            df["planned_work_quantity"]
            + np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
        )

        # Binary target: material required or not
        df["material_required"] = (df["actual_material_qty"] > 0).astype(int)

        self.X = df[self.CATEGORICAL_COLS + self.NUMERICAL_COLS]
        self.y = df["material_required"]

        print("Dataset shape:", df.shape)
        print(df.head())

    # -----------------------------
    # ML pipeline
    # -----------------------------
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
                ("classifier", LogisticRegression(max_iter=1000))
            ]
        )

    # -----------------------------
    # Evaluation
    # -----------------------------
    def run_cross_validation(self):
        print("\nRunning material requirement classification (10-fold CV)\n")

        kf = KFold(n_splits=10, shuffle=True, random_state=42)

        accs, precs, recalls = [], [], []

        for train_idx, test_idx in kf.split(self.X):
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_test)

            accs.append(accuracy_score(y_test, preds))
            precs.append(precision_score(y_test, preds))
            recalls.append(recall_score(y_test, preds))

        print(f"Accuracy : {np.mean(accs):.3f}")
        print(f"Precision: {np.mean(precs):.3f}")
        print(f"Recall   : {np.mean(recalls):.3f}")


if __name__ == "__main__":
    SAPMaterialEstimator()
