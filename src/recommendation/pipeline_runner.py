import os
import subprocess
import pandas as pd

from delta_extractor import DeltaExtractor
from evidence_aggregator import EvidenceAggregator
from proposal_generator import ProposalGenerator


class PipelineRunner:
    """
    End-to-End SAP Tasklist Optimization Pipeline
    """

    def __init__(self):

        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        self.ROOT = os.path.join(self.BASE_DIR, "..", "..")

        self.DATA = os.path.join(self.ROOT, "data")

    # -----------------------------
    # Step 1: Generate Data
    # -----------------------------
    def generate_data(self):

        print("\n[1] Generating synthetic data...\n")

        subprocess.run(
            ["python3", "src/data_generation/generate_planned_data.py"],
            cwd=self.ROOT,
            check=True
        )

        subprocess.run(
            ["python3", "src/data_generation/generate_actual_data.py"],
            cwd=self.ROOT,
            check=True
        )

        subprocess.run(
            ["python3", "src/data_generation/merge_planned_actual.py"],
            cwd=self.ROOT,
            check=True
        )

    # -----------------------------
    # Step 2: Time Estimation
    # -----------------------------
    def run_time_model(self):

        print("\n[2] Running Time Estimation...\n")

        subprocess.run(
            ["python3", "src/estimation/time_estimation.py"],
            cwd=self.ROOT,
            check=True
        )

    # -----------------------------
    # Step 3: Manpower Estimation
    # -----------------------------
    def run_manpower_model(self):

        print("\n[3] Running Manpower Estimation...\n")

        subprocess.run(
            ["python3", "src/estimation/manpower_estimation.py"],
            cwd=self.ROOT,
            check=True
        )

    # -----------------------------
    # Step 4: Material Estimation
    # -----------------------------
    def run_material_model(self):

        print("\n[4] Running Material Estimation...\n")

        subprocess.run(
            ["python3", "src/estimation/material_estimation.py"],
            cwd=self.ROOT,
            check=True
        )

    # -----------------------------
    # Step 5: Load Unified Dataset
    # -----------------------------
    def load_final_dataset(self):

        print("\n[5] Preparing unified dataset...\n")

        base = pd.read_csv(
            os.path.join(self.DATA, "effort_training_dataset.csv")
        )

        time = pd.read_csv(
            os.path.join(self.DATA, "predicted_time.csv")
        )

        man = pd.read_csv(
            os.path.join(self.DATA, "predicted_manpower.csv")
        )

        material = pd.read_csv(
            os.path.join(self.DATA, "predicted_material.csv")
        )

        # Merge step-by-step
        df = base.merge(time, on="operation_id")
        df = df.merge(man, on="operation_id")
        df = df.merge(material, on="operation_id")

        print("Unified dataset shape:", df.shape)

        return df

    # -----------------------------
    # Step 6: Evidence Extraction
    # -----------------------------
    def extract_evidence(self, df):

        print("\n[6] Extracting evidence...\n")

        extractor = DeltaExtractor()

        return extractor.extract(df)

    # -----------------------------
    # Step 7: Aggregate Evidence
    # -----------------------------
    def aggregate_evidence(self, evidence):

        print("\n[7] Aggregating evidence...\n")

        agg = EvidenceAggregator()

        return agg.aggregate(evidence)

    # -----------------------------
    # Step 8: Generate Proposals
    # -----------------------------
    def generate_proposals(self, summary):

        print("\n[8] Generating proposals...\n")

        gen = ProposalGenerator()

        proposals = gen.generate(summary)

        out = os.path.join(self.DATA, "change_proposals.csv")

        proposals.to_csv(out, index=False)

        print("Saved proposals →", out)

    # -----------------------------
    # Run All
    # -----------------------------
    def run(self):

        print("\nStarting SAP Optimization Pipeline...\n")

        # Data
        self.generate_data()

        # ML Models
        self.run_time_model()
        self.run_manpower_model()
        self.run_material_model()

        # Analytics
        final_df = self.load_final_dataset()

        evidence = self.extract_evidence(final_df)

        summary = self.aggregate_evidence(evidence)

        self.generate_proposals(summary)

        print("\nPipeline completed successfully ✅\n")


if __name__ == "__main__":

    runner = PipelineRunner()

    runner.run()
