import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


class SAPMaterialEstimator:
    """
    End-to-End Material Requirement Estimator
    Predicts: Whether material is required (0/1)
    """

    def __init__(self):

        # -----------------------------
        # Paths
        # -----------------------------
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")

        self.BASESET = os.path.join(
            self.DATA_DIR,
            "effort_training_dataset.csv"
        )

        self.TIME_PRED = os.path.join(
            self.DATA_DIR,
            "predicted_time.csv"
        )

        self.MAN_PRED = os.path.join(
            self.DATA_DIR,
            "predicted_manpower.csv"
        )

        self.OUTPUT = os.path.join(
            self.DATA_DIR,
            "predicted_material.csv"
        )

        self.MODEL = os.path.join(
            BASE_DIR, "..", "..", "models", "material_model.pkl"
        )

        os.makedirs(os.path.dirname(self.MODEL), exist_ok=True)

        # -----------------------------
        # Features
        # -----------------------------
        self.CATEGORICAL = [
            "operation_type",
            "equipment",
            "plant"
        ]

        self.NUMERICAL = [
            "predicted_duration",
            "predicted_manpower"
        ]

        self.TARGET = "material_required"

        # -----------------------------
        # Run
        # -----------------------------
        self.load_data()
        self.build_pipeline()
        self.evaluate()
        self.train_final()
        self.export_predictions()

    # -----------------------------
    # Load & Merge Data
    # -----------------------------
    def load_data(self):

        print("\nLoading material dataset...")

        base = pd.read_csv(self.BASESET)
        time = pd.read_csv(self.TIME_PRED)
        man = pd.read_csv(self.MAN_PRED)

        # Merge predictions
        df = base.merge(time, on="operation_id")
        df = df.merge(man, on="operation_id")

        # Binary target
        df["material_required"] = (
            df["actual_material_qty"] > 0
        ).astype(int)

        self.df = df

        self.X = df[self.CATEGORICAL + self.NUMERICAL]
        self.y = df[self.TARGET]

        print("Rows:", len(df))
        print(df.head())

    # -----------------------------
    # Build Pipeline
    # -----------------------------
    def build_pipeline(self):

        self.preprocessor = ColumnTransformer(
            [
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    self.CATEGORICAL
                ),
                (
                    "num",
                    "passthrough",
                    self.NUMERICAL
                )
            ]
        )

        self.pipeline = Pipeline(
            [
                ("prep", self.preprocessor),
                ("clf", LogisticRegression(max_iter=1000))
            ]
        )

    # -----------------------------
    # Cross Validation
    # -----------------------------
    def evaluate(self):

        print("\nEvaluating material model (10-fold CV)\n")

        kf = KFold(10, shuffle=True, random_state=42)

        accs, precs, recalls = [], [], []

        for tr, te in kf.split(self.X):

            Xtr = self.X.iloc[tr]
            Xte = self.X.iloc[te]

            ytr = self.y.iloc[tr]
            yte = self.y.iloc[te]

            self.pipeline.fit(Xtr, ytr)

            preds = self.pipeline.predict(Xte)

            accs.append(accuracy_score(yte, preds))
            precs.append(precision_score(yte, preds))
            recalls.append(recall_score(yte, preds))

        print(f"Accuracy : {np.mean(accs):.3f}")
        print(f"Precision: {np.mean(precs):.3f}")
        print(f"Recall   : {np.mean(recalls):.3f}")

    # -----------------------------
    # Final Training
    # -----------------------------
    def train_final(self):

        print("\nTraining final material model...")

        self.pipeline.fit(self.X, self.y)

        joblib.dump(self.pipeline, self.MODEL)

        print("Saved model →", self.MODEL)

    # -----------------------------
    # Export Predictions
    # -----------------------------
    def export_predictions(self):

        print("\nExporting material predictions...")

        preds = self.pipeline.predict(self.X)

        out = self.df[["operation_id"]].copy()

        out["predicted_material_required"] = preds

        out.to_csv(self.OUTPUT, index=False)

        print("Saved predictions →", self.OUTPUT)


if __name__ == "__main__":

    SAPMaterialEstimator()
