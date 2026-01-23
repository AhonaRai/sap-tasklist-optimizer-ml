import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SAPManpowerEstimator:
    """
    Chained Manpower Estimator
    Uses predicted_duration
    Target: actual_work_quantity
    """

    def __init__(self):

        # -----------------------------
        # Paths
        # -----------------------------
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.BASE_DATA = os.path.join(BASE_DIR, "..", "..", "data")

        self.BASESET = os.path.join(
            self.BASE_DATA,
            "effort_training_dataset.csv"
        )

        self.TIME_PRED = os.path.join(
            self.BASE_DATA,
            "predicted_time.csv"
        )

        self.OUT = os.path.join(
            self.BASE_DATA,
            "predicted_manpower.csv"
        )

        self.MODEL = os.path.join(
            BASE_DIR, "..", "..", "models", "manpower_model.pkl"
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
            "planned_work_quantity",
            "predicted_duration"
        ]

        self.TARGET = "actual_work_quantity"

        # -----------------------------
        # Models
        # -----------------------------
        self.models = {
            "Linear": LinearRegression(),
            "Ridge": RidgeCV(alphas=[0.1, 1, 10]),
            "Lasso": LassoCV(cv=5),
            "RF": RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        }

        # -----------------------------
        # Run
        # -----------------------------
        self.load()
        self.prep()
        self.eval()
        self.train()
        self.export()

    # -----------------------------
    # Load
    # -----------------------------
    def load(self):

        print("\nLoading manpower dataset...")

        base = pd.read_csv(self.BASESET)
        time = pd.read_csv(self.TIME_PRED)

        if "operation_id" not in base:
            raise ValueError("Missing operation_id in base")

        if "operation_id" not in time:
            raise ValueError("Missing operation_id in time")

        if "predicted_duration" not in time:
            raise ValueError("Missing predicted_duration")

        time = time[["operation_id", "predicted_duration"]]

        df = base.merge(
            time,
            on="operation_id",
            validate="one_to_one"
        )

        if df["predicted_duration"].isna().any():
            raise ValueError("Run time model first")

        self.df = df

        self.X = df[self.CATEGORICAL + self.NUMERICAL]
        self.y = df[self.TARGET]

        print("Rows:", len(df))
        print(df.head())

    # -----------------------------
    # Prep
    # -----------------------------
    def prep(self):

        self.prep_pipe = ColumnTransformer(
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
    def eval(self):

        print("\nEvaluating manpower models...\n")

        kf = KFold(10, shuffle=True, random_state=42)

        self.scores = {}

        for name, model in self.models.items():

            rmses = []

            pipe = Pipeline([
                ("prep", self.prep_pipe),
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
    # Train
    # -----------------------------
    def train(self):

        best = min(self.scores, key=self.scores.get)

        print("\nBest manpower model:", best)

        self.final = Pipeline([
            ("prep", self.prep_pipe),
            ("model", self.models[best])
        ])

        self.final.fit(self.X, self.y)

        joblib.dump(self.final, self.MODEL)

        print("Saved:", self.MODEL)

    # -----------------------------
    # Export
    # -----------------------------
    def export(self):

        print("\nExporting manpower predictions...")

        p = self.final.predict(self.X)

        out = self.df.copy()

        out["predicted_manpower"] = np.maximum(
            1,
            np.round(p)
        ).astype(int)

        out = out[[
            "operation_id",
            "predicted_manpower"
        ]]

        out.to_csv(self.OUT, index=False)

        print("Saved:", self.OUT)


if __name__ == "__main__":
    SAPManpowerEstimator()
