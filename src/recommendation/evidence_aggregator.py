import pandas as pd


class EvidenceAggregator:
    """
    Aggregates order-level evidence into operation-level patterns
    """

    def aggregate(self, df):

        print("\nAggregating evidence...\n")

        # -----------------------------
        # Safety Check
        # -----------------------------
        required = [
            "operation_id",
            "order_id",

            "time_gap",
            "manpower_gap",
            "material_gap",

            "time_overrun",
            "manpower_overrun",
            "material_overuse",
        ]

        missing = set(required) - set(df.columns)

        if missing:
            raise ValueError(f"Missing columns in evidence: {missing}")

        # -----------------------------
        # Group by Operation
        # -----------------------------
        grouped = df.groupby("operation_id")

        summary = grouped.agg({

            # Magnitudes (how big are deviations)
            "time_gap": "mean",
            "manpower_gap": "mean",
            "material_gap": "mean",

            # Rates (how often problems happen)
            "time_overrun": "mean",
            "manpower_overrun": "mean",
            "material_overuse": "mean",

            # Frequency (how many orders affected)
            "order_id": "count"

        }).reset_index()

        # -----------------------------
        # Rename for Clarity
        # -----------------------------
        summary.rename(columns={

            "time_gap": "avg_time_gap",
            "manpower_gap": "avg_manpower_gap",
            "material_gap": "avg_material_gap",

            "time_overrun": "time_overrun_rate",
            "manpower_overrun": "manpower_overrun_rate",
            "material_overuse": "material_overuse_rate",

            "order_id": "frequency"

        }, inplace=True)

        # Save for inspection
        summary.to_csv("data/aggregated_evidence.csv", index=False)

        print("Aggregated evidence sample:")
        print(summary.head())

        return summary
