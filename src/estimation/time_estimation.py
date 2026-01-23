import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class SAPTimeEstimator:
    """
    End-to-End SAP Time Estimation
    Target: actual_duration
    Outputs: predicted_time.csv
    """

    def __init__(self):

        # -----------------------------
        # Paths
        # -----------------------------
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.BASE_DATA = os.path.join(BASE_DIR, "..", "..", "data")

        self.DATASET_PATH = os.path.join(
            self.BASE_DATA,
            "effort_training_dataset.csv"
        )

        self.OUTPUT_PATH = os.path.join(
            self.BASE_DATA,
            "predicted_time.csv"
        )

        self.MODEL_PATH = os.path.join(
            BASE_DIR, "..", "..", "models", "time_model.pkl"
        )

        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)

        # -----------------------------
        # Features
        # -----------------------------
        self.CATEGORICAL = [
            "operation_type",
            "equipment",
            "plant"
        ]

        self.NUMERICAL = [
            "planned_duration",
            "planned_work_quantity",
            "planned_material_qty"
        ]

        self.TARGET = "actual_duration"

        # -----------------------------
        # Models
        # -----------------------------
        self.models = {
            "Linear": LinearRegression(),
            "Ridge": RidgeCV(alphas=[0.1, 1, 10]),
            "Lasso": LassoCV(cv=5),
            "Elastic": ElasticNetCV(cv=5),
            "RF": RandomForestRegressor(
                n_estimators=300,
                random_state=42
            )
        }

        # -----------------------------
        # Run
        # -----------------------------
        self.load_data()
        self.build_preprocessor()
        self.evaluate_models()
        self.train_best()
        self.export_predictions()

    # -----------------------------
    # Load
    # -----------------------------
    def load_data(self):

        print("\nLoading time dataset...")

        df = pd.read_csv(self.DATASET_PATH)

        if "operation_id" not in df.columns:
            raise ValueError("Missing operation_id")

        self.df = df

        self.X = df[self.CATEGORICAL + self.NUMERICAL]
        self.y = df[self.TARGET]

        print("Rows:", len(df))
        print(df.head())

    # -----------------------------
    # Preprocess
    # -----------------------------
    def build_preprocessor(self):

        self.preprocessor = ColumnTransformer(
            [
                ("cat",
                 OneHotEncoder(handle_unknown="ignore"),
                 self.CATEGORICAL),

                ("num",
                 "passthrough",
                 self.NUMERICAL),
            ]
        )

    # -----------------------------
    # CV
    # -----------------------------
    def evaluate_models(self):

        print("\nEvaluating time models...\n")

        kf = KFold(10, shuffle=True, random_state=42)

        self.scores = {}

        for name, model in self.models.items():

            rmses = []

            pipe = Pipeline([
                ("prep", self.preprocessor),
                ("model", model)
            ])

            for tr, te in kf.split(self.X):

                Xtr, Xte = self.X.iloc[tr], self.X.iloc[te]
                ytr, yte = self.y.iloc[tr], self.y.iloc[te]

                pipe.fit(Xtr, ytr)
                p = pipe.predict(Xte)

                rmses.append(
                    np.sqrt(mean_squared_error(yte, p))
                )

            self.scores[name] = np.mean(rmses)

            print(name, "RMSE:", round(self.scores[name], 3))

    # -----------------------------
    # Train Best
    # -----------------------------
    def train_best(self):

        self.best = min(self.scores, key=self.scores.get)

        print("\nBest time model:", self.best)

        self.final_model = Pipeline([
            ("prep", self.preprocessor),
            ("model", self.models[self.best])
        ])

        self.final_model.fit(self.X, self.y)

        joblib.dump(self.final_model, self.MODEL_PATH)

        print("Saved:", self.MODEL_PATH)

    # -----------------------------
    # Export
    # -----------------------------
    def export_predictions(self):

        print("\nExporting time predictions...")

        preds = self.final_model.predict(self.X)

        out = self.df.copy()

        out["predicted_duration"] = preds

        out = out[[
            "operation_id",
            "predicted_duration"
        ]]

        out.to_csv(self.OUTPUT_PATH, index=False)

        print("Saved:", self.OUTPUT_PATH)


if __name__ == "__main__":
    SAPTimeEstimator()
