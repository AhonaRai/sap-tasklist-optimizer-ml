import pandas as pd
import os


class DataLoader:
    """
    Loads and merges all pipeline outputs
    """

    def __init__(self):

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.DATA_DIR = os.path.join(
            BASE_DIR, "..", "..", "data"
        )

    def load(self):

        print("\nLoading all datasets...")

        # -----------------------------
        # Load Files
        # -----------------------------

        planned = pd.read_csv(
            os.path.join(
                self.DATA_DIR,
                "planned_tasklist_operations.csv"
            )
        )

        actual = pd.read_csv(
            os.path.join(
                self.DATA_DIR,
                "actual_maintenance_operations.csv"
            )
        )

        time = pd.read_csv(
            os.path.join(
                self.DATA_DIR,
                "predicted_time.csv"
            )
        )

        manpower = pd.read_csv(
            os.path.join(
                self.DATA_DIR,
                "predicted_manpower.csv"
            )
        )

        material = pd.read_csv(
            os.path.join(
                self.DATA_DIR,
                "predicted_material.csv"
            )
        )

        # -----------------------------
        # Safe Merging
        # -----------------------------

        df = planned.merge(
            actual,
            on="operation_id",
            validate="one_to_one"
        )

        df = df.merge(
            time,
            on="operation_id",
            validate="one_to_one"
        )

        df = df.merge(
            manpower,
            on="operation_id",
            validate="one_to_one"
        )

        df = df.merge(
            material,
            on="operation_id",
            validate="one_to_one"
        )

        # -----------------------------
        # Final Check
        # -----------------------------

        print("Unified rows:", len(df))
        print("Unified columns:", df.columns.tolist())

        return df
