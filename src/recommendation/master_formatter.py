import pandas as pd
import json
import os


class MasterProposalFormatter:
    """
    Converts change_proposals.csv into SAP-style master_change_proposals.json
    """

    def __init__(self):

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.DATA_DIR = os.path.join(BASE_DIR, "..", "..", "data")

        self.PROPOSALS_PATH = os.path.join(
            self.DATA_DIR,
            "change_proposals.csv"
        )

        self.EVIDENCE_PATH = os.path.join(
            self.DATA_DIR,
            "order_level_analysis.csv"
        )

        self.OUTPUT_PATH = os.path.join(
            self.DATA_DIR,
            "master_change_proposals.json"
        )

        # Fake product mapping (for demo)
        self.PRODUCT_MAP = {
            "PUMP": "SEAL_X12",
            "VALVE": "GREASE_A",
            "MOTOR": "CLEANER_B"
        }

    # -----------------------------
    # Confidence Logic
    # -----------------------------
    def get_confidence(self, freq, rate):

        if freq >= 10 and rate >= 0.7:
            return "HIGH"

        if freq >= 5 and rate >= 0.4:
            return "MEDIUM"

        return "LOW"

    # -----------------------------
    # Load Data
    # -----------------------------
    def load_data(self):

        self.proposals = pd.read_csv(self.PROPOSALS_PATH)

        self.evidence = pd.read_csv(self.EVIDENCE_PATH)

        print("Loaded proposals:", len(self.proposals))
        print("Loaded evidence:", len(self.evidence))

    # -----------------------------
    # Build Order-Level Analysis
    # -----------------------------
    def build_order_analysis(self):

        order_analysis = {}

        for _, row in self.evidence.iterrows():

            oid = row["order_id"]

            if oid not in order_analysis:
                order_analysis[oid] = {
                    "time_deltas": [],
                    "manpower_deltas": [],
                    "material_deltas": []
                }

            order_analysis[oid]["time_deltas"].append(
                {
                    "operation_id": row["operation_id"],
                    "delta": row["time_gap"]
                }
            )

            order_analysis[oid]["manpower_deltas"].append(
                {
                    "operation_id": row["operation_id"],
                    "delta": row["manpower_gap"]
                }
            )

            order_analysis[oid]["material_deltas"].append(
                {
                    "operation_id": row["operation_id"],
                    "delta": row["material_gap"]
                }
            )

        return order_analysis

    # -----------------------------
    # Build Master Proposals
    # -----------------------------
    def build_master_proposals(self):

        master = []

        for _, row in self.proposals.iterrows():

            op_id = row["operation_id"]

            confidence = row["confidence"]

            # ---------------- TIME ----------------
            if not pd.isna(row["time_update"]):

                master.append({
                    "type": "UPDATE_DURATION",
                    "TaskListOperationInternalId": op_id,
                    "confidence": confidence,
                    "suggested": {
                        "hours": round(float(row["time_update"]), 2)
                    },
                    "evidence": {
                        "frequency": int(row["frequency"]),
                        "time_overrun_rate": float(row["time_overrun_rate"])
                    }
                })

            # ---------------- MANPOWER ----------------
            if not pd.isna(row["manpower_update"]):

                master.append({
                    "type": "UPDATE_MANPOWER",
                    "TaskListOperationInternalId": op_id,
                    "confidence": confidence,
                    "suggested": {
                        "quantity": int(row["manpower_update"]),
                        "unit": "EA"
                    },
                    "evidence": {
                        "frequency": int(row["frequency"]),
                        "manpower_overrun_rate": float(row["manpower_overrun_rate"])
                    }
                })

            # ---------------- MATERIAL ----------------
            if not pd.isna(row["material_update"]):

                master.append({
                    "type": "UPDATE_COMPONENT_QUANTITY",
                    "TaskListOperationInternalId": op_id,
                    "confidence": confidence,
                    "suggested": {
                        "quantity": int(row["material_update"]),
                        "unit": "EA"
                    },
                    "evidence": {
                        "frequency": int(row["frequency"]),
                        "material_overuse_rate": float(row["material_overuse_rate"])
                    }
                })

        return master

    # -----------------------------
    # Run Formatter
    # -----------------------------
    def run(self):

        print("\nFormatting master proposals...\n")

        self.load_data()

        order_analysis = self.build_order_analysis()

        master = self.build_master_proposals()

        final = {
            "master_change_proposals": master,
            "order_level_analysis": order_analysis
        }

        with open(self.OUTPUT_PATH, "w") as f:
            json.dump(final, f, indent=4)

        print("Saved →", self.OUTPUT_PATH)
        print("Total proposals:", len(master))


if __name__ == "__main__":

    formatter = MasterProposalFormatter()

    formatter.run()
