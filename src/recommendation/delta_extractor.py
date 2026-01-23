import pandas as pd


class DeltaExtractor:
    """
    Extracts per-order behavioral deltas
    """

    def extract(self, df):

        print("\nExtracting behavioral evidence...\n")

        evidence_rows = []

        for _, row in df.iterrows():

            time_gap = row["actual_duration"] - row["predicted_duration"]

            manpower_gap = (
                row["actual_work_quantity"]
                - row["predicted_manpower"]
            )

            material_gap = (
                row["actual_material_qty"]
                - row["predicted_material_required"]
            )

            record = {

                # -----------------------------
                # IDENTIFIERS
                # -----------------------------
                "order_id": row["order_id"],
                "operation_id": row["operation_id"],

                # -----------------------------
                # CONTEXT
                # -----------------------------
                "operation_type": row["operation_type"],
                "equipment": row["equipment"],
                "plant": row["plant"],

                # -----------------------------
                # TIME
                # -----------------------------
                "planned_duration": row["planned_duration"],
                "actual_duration": row["actual_duration"],
                "predicted_duration": row["predicted_duration"],
                "time_gap": round(time_gap, 2),

                # -----------------------------
                # MANPOWER
                # -----------------------------
                "planned_work_quantity": row["planned_work_quantity"],
                "actual_work_quantity": row["actual_work_quantity"],
                "predicted_manpower": row["predicted_manpower"],
                "manpower_gap": manpower_gap,

                # -----------------------------
                # MATERIAL
                # -----------------------------
                "planned_material_qty": row["planned_material_qty"],
                "actual_material_qty": row["actual_material_qty"],
                "predicted_material_required": row["predicted_material_required"],
                "material_gap": material_gap,

                # -----------------------------
                # FLAGS
                # -----------------------------
                "time_overrun": int(time_gap > 0.5),
                "manpower_overrun": int(manpower_gap > 0),
                "material_overuse": int(material_gap > 0),
            }

            evidence_rows.append(record)

        evidence_df = pd.DataFrame(evidence_rows)

        evidence_df.to_csv("data/order_level_analysis.csv", index=False)

        print("Evidence sample:")
        print(evidence_df.head())

        return evidence_df
