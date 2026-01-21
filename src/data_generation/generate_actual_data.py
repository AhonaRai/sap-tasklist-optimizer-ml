import pandas as pd
import numpy as np
import random

PLANNED_DATA_PATH = "data/planned_tasklist_operations.csv"

planned_df = pd.read_csv(PLANNED_DATA_PATH)

def actual_duration_from_planned(planned_duration, operation_type):
    """
    Controlled execution variance by operation type
    """

    if operation_type == "INSPECTION":
        factor = random.uniform(0.9, 1.1)
    elif operation_type == "REPAIR":
        factor = random.uniform(0.85, 1.25)
    else:  # REPLACEMENT
        factor = random.uniform(0.8, 1.3)

    return round(planned_duration * factor, 1)

def actual_work_quantity_from_planned(planned_qty, operation_type):
    """
    Manpower usually matches planned, occasionally increases.
    """
    if operation_type == "INSPECTION":
        return planned_qty

    if random.random() < 0.2:  # 20% cases need extra help
        return planned_qty + 1

    return planned_qty

def actual_material_from_planned(planned_material, operation_type):
    """
    Material usage is event-based, not duration-based.
    """
    if operation_type == "INSPECTION":
        return random.choice([0, planned_material])

    if random.random() < 0.15:
        return planned_material + 1

    return planned_material

def generate_actual_operations(planned_df):
    rows = []

    for idx, row in planned_df.iterrows():
        actual_duration = actual_duration_from_planned(
            row["planned_duration"],
            row["operation_type"]
        )

        actual_work_qty = actual_work_quantity_from_planned(
            row["planned_work_quantity"],
            row["operation_type"]
        )

        actual_material_qty = actual_material_from_planned(
            row["planned_material_qty"],
            row["operation_type"]
        )

        rows.append({
            "order_id": f"MO_{2000 + idx}",
            "tasklist_group": row["tasklist_group"],
            "operation_id": row["operation_id"],
            "actual_duration": actual_duration,
            "actual_work_quantity": actual_work_qty,
            "actual_material_qty": actual_material_qty
        })

    return pd.DataFrame(rows)

def validate_actual_data(df):
    assert df.isnull().sum().sum() == 0, "Nulls found"
    assert (df["actual_duration"] > 0).all(), "Invalid duration"
    assert (df["actual_work_quantity"] >= 1).all(), "Invalid manpower"
    assert (df["actual_material_qty"] >= 0).all(), "Invalid material qty"

if __name__ == "__main__":
    actual_df = generate_actual_operations(planned_df)
    validate_actual_data(actual_df)

    actual_df.to_csv(
        "data/actual_maintenance_operations.csv",
        index=False
    )

    print("Actual synthetic data generated.")
    print(actual_df.head())

