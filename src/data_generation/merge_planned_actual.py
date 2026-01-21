import pandas as pd
planned_df = pd.read_csv("data/planned_tasklist_operations.csv")
actual_df = pd.read_csv("data/actual_maintenance_operations.csv")
merged_df = planned_df.merge(
    actual_df,
    on=["tasklist_group", "operation_id"],
    how="inner"
)

assert len(merged_df) == len(planned_df), "Row mismatch after merge"

effort_df = merged_df[[
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

print("Effort estimation dataset created.")
print(effort_df.head())

