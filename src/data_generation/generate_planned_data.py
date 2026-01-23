import pandas as pd
import random

# --- Domains ---

TASKLIST_GROUPS = ["TL_1001", "TL_1002", "TL_1003"]
TASKLIST_TYPES = ["A", "B"]

OPERATION_TYPES = ["INSPECTION", "REPAIR", "REPLACEMENT"]

PLANTS = ["PL01", "PL02"]
WORK_CENTERS = ["WC_MECH", "WC_ELEC"]
EQUIPMENTS = ["PUMP", "MOTOR", "VALVE"]


def planned_effort_by_operation(operation_type):

    if operation_type == "INSPECTION":
        return round(random.uniform(0.5, 2.0), 1), 1, random.choice([0, 1])

    elif operation_type == "REPAIR":
        return round(random.uniform(2.0, 5.0), 1), random.choice([1, 2]), random.choice([1, 2])

    else:  # REPLACEMENT
        return round(random.uniform(3.0, 8.0), 1), random.choice([2, 3]), random.choice([1, 3])


def generate_planned_operations(n_rows=200, seed=42):

    random.seed(seed)
    rows = []

    for i in range(n_rows):

        op_type = random.choice(OPERATION_TYPES)
        duration, work_qty, material_qty = planned_effort_by_operation(op_type)

        row = {
            # ✅ PRIMARY BUSINESS KEY
            "order_id": f"MO_{2000+i}",
            "operation_id": f"OP_{1000+i}",   # PRIMARY KEY
            "tasklist_group": random.choice(TASKLIST_GROUPS),
            "tasklist_type": random.choice(TASKLIST_TYPES),

            "operation_type": op_type,
            "plant": random.choice(PLANTS),
            "work_center": random.choice(WORK_CENTERS),
            "equipment": random.choice(EQUIPMENTS),

            "planned_duration": duration,
            "planned_work_quantity": work_qty,
            "planned_material_qty": material_qty,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def validate(df):

    assert "order_id" in df.columns
    assert "operation_id" in df.columns

    assert df["order_id"].is_unique
    assert df["operation_id"].is_unique



if __name__ == "__main__":

    df = generate_planned_operations(200)

    validate(df)

    df.to_csv(
        "data/planned_tasklist_operations.csv",
        index=False
    )

    print("Planned data generated")
    print(df.head())
