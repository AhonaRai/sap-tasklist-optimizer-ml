import pandas as pd


class ProposalGenerator:
    """
    Generates explainable task list change proposals
    from aggregated behavioral evidence
    """

    def generate(self, summary_df):

        proposals = []

        for _, row in summary_df.iterrows():

            rec = {

                # Identifier
                "operation_id": row["operation_id"],

                # Suggested updates
                "time_update": None,
                "manpower_update": None,
                "material_update": None,

                # Confidence
                "confidence": None,

                # Evidence
                "frequency": int(row["frequency"]),
                "time_overrun_rate": round(row["time_overrun_rate"], 2),
                "manpower_overrun_rate": round(row["manpower_overrun_rate"], 2),
                "material_overuse_rate": round(row["material_overuse_rate"], 2),

                # Explanation
                "reasons": []
            }

            strength = 0   # for confidence scoring


            # -----------------------------
            # TIME
            # -----------------------------
            if abs(row["avg_time_gap"]) > 0.5:

                rec["time_update"] = round(row["avg_time_gap"], 2)

                rec["reasons"].append(
                    f"Avg time gap {round(row['avg_time_gap'],2)} hrs"
                )

                strength += row["time_overrun_rate"]


            # -----------------------------
            # MANPOWER
            # -----------------------------
            if abs(row["avg_manpower_gap"]) >= 1:

                rec["manpower_update"] = int(
                    round(row["avg_manpower_gap"])
                )

                rec["reasons"].append(
                    f"Avg manpower gap {round(row['avg_manpower_gap'],2)}"
                )

                strength += row["manpower_overrun_rate"]


            # -----------------------------
            # MATERIAL
            # -----------------------------
            if abs(row["avg_material_gap"]) >= 1:

                rec["material_update"] = int(
                    round(row["avg_material_gap"])
                )

                rec["reasons"].append(
                    f"Avg material gap {round(row['avg_material_gap'],2)}"
                )

                strength += row["material_overuse_rate"]


            # -----------------------------
            # Confidence Scoring
            # -----------------------------
            if strength >= 1.5 and row["frequency"] >= 20:
                rec["confidence"] = "HIGH"

            elif strength >= 0.8 and row["frequency"] >= 10:
                rec["confidence"] = "MEDIUM"

            else:
                rec["confidence"] = "LOW"


            # -----------------------------
            # Keep Only Useful Proposals
            # -----------------------------
            if rec["reasons"]:

                rec["reasons"] = "; ".join(rec["reasons"])
                proposals.append(rec)


        return pd.DataFrame(proposals)
