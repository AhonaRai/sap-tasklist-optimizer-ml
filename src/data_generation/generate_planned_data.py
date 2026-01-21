import pandas as pd
import numpy as np
import random
# --- SAP-like controlled domains ---

TASKLIST_GROUPS = ["TL_1001", "TL_1002", "TL_1003"]
TASKLIST_TYPES = ["A", "B"]  # A = Equipment, B = Functional Location

OPERATION_TYPES = [
    "INSPECTION",
    "REPAIR",
    "REPLACEMENT"
]

PLANTS = ["PL01", "PL02"]
WORK_CENTERS = ["WC_MECH", "WC_ELEC"]
EQUIPMENTS = ["PUMP", "MOTOR", "VALVE"]

def planned_effort_by_operation(operation_type):
    """
    Business-driven planned effort assumptions.
    These simulate how task lists are authored in SAP.
    """

    if operation_type == "INSPECTION":
        planned_duration = round(random.uniform(0.5, 2.0), 1)
        planned_work_qty = 1
        planned_material_qty = random.choice([0, 1])

    elif operation_type == "REPAIR":
        planned_duration = round(random.uniform(2.0, 5.0), 1)
        planned_work_qty = random.choice([1, 2])
        planned_material_qty = random.choice([1, 2])

    else:  # REPLACEMENT
        planned_duration = round(random.uniform(3.0, 8.0), 1)
        planned_work_qty = random.choice([2, 3])
        planned_material_qty = random.choice([1, 3])

    return planned_duration, planned_work_qty, planned_material_qty

def generate_planned_operations(n_rows=200, seed=42):
    random.seed(seed)
    rows = []

    for i in range(n_rows):
        op_type = random.choice(OPERATION_TYPES)
        duration, work_qty, material_qty = planned_effort_by_operation(op_type)

        row = {
            "tasklist_group": random.choice(TASKLIST_GROUPS),
            "tasklist_type": random.choice(TASKLIST_TYPES),
            "operation_id": f"OP_{1000 + i}",
            "operation_type": op_type,
            "plant": random.choice(PLANTS),
            "work_center": random.choice(WORK_CENTERS),
            "equipment": random.choice(EQUIPMENTS),
            "planned_duration": duration,
            "planned_work_quantity": work_qty,
            "planned_material_qty": material_qty
        }

        rows.append(row)

    return pd.DataFrame(rows)
def validate_planned_data(df):
    assert df.isnull().sum().sum() == 0, "Null values found"
    assert (df["planned_duration"] > 0).all(), "Invalid durations"
    assert (df["planned_work_quantity"] >= 1).all(), "Invalid manpower"
    assert (df["planned_material_qty"] >= 0).all(), "Invalid material qty"

if __name__ == "__main__":
    planned_df = generate_planned_operations(n_rows=200)
    validate_planned_data(planned_df)

    planned_df.to_csv(
        "data/planned_tasklist_operations.csv",
        index=False
    )

    print("Planned synthetic data generated.")
    print(planned_df.head())
