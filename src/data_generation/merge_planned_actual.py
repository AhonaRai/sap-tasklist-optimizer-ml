import pandas as pd

planned = pd.read_csv("data/planned_tasklist_operations.csv")
actual = pd.read_csv("data/actual_maintenance_operations.csv")

merged = planned.merge(
    actual,
    on="operation_id",
    how="inner",
    suffixes=("_plan", "_act")
)

# Safety check
assert len(merged) == len(planned)

# Unify order_id (they should be identical)
merged["order_id"] = merged["order_id_plan"]

effort_df = merged[[
    "order_id",
    "operation_id",

    "operation_type",
    "equipment",
    "plant",

    "planned_duration",
    "planned_work_quantity",
    "planned_material_qty",

    "actual_duration",
    "actual_work_quantity",
    "actual_material_qty"
]]

effort_df.to_csv(
    "data/effort_training_dataset.csv",
    index=False
)

print("Merged dataset created")
print(effort_df.head())
print(len(effort_df))