import pandas as pd
import random

PLANNED_PATH = "data/planned_tasklist_operations.csv"


def actual_duration(planned, op_type):

    if op_type == "INSPECTION":
        factor = random.uniform(0.9, 1.1)
    elif op_type == "REPAIR":
        factor = random.uniform(0.85, 1.25)
    else:
        factor = random.uniform(0.8, 1.3)

    return round(planned * factor, 1)


def actual_work(planned, op_type):

    if op_type == "INSPECTION":
        return planned

    if random.random() < 0.2:
        return planned + 1

    return planned


def actual_material(planned, op_type):

    if op_type == "INSPECTION":
        return random.choice([0, planned])

    if random.random() < 0.15:
        return planned + 1

    return planned


def generate_actual(planned_df):

    rows = []

    for i, row in planned_df.iterrows():

        rows.append({

            # ✅ REUSE KEYS
            "order_id": row["order_id"],
            "operation_id": row["operation_id"],  # KEEP KEY
           

            "tasklist_group": row["tasklist_group"],

            "actual_duration": actual_duration(
                row["planned_duration"],
                row["operation_type"]
            ),

            "actual_work_quantity": actual_work(
                row["planned_work_quantity"],
                row["operation_type"]
            ),

            "actual_material_qty": actual_material(
                row["planned_material_qty"],
                row["operation_type"]
            )
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":

    planned = pd.read_csv(PLANNED_PATH)

    actual = generate_actual(planned)

    actual.to_csv(
        "data/actual_maintenance_operations.csv",
        index=False
    )

    print("Actual data generated")
    print(actual.head())
