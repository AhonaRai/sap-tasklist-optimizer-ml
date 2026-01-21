# SAP Task List Optimizer (ML Prototype)

This repository contains a staged machine learning prototype for optimizing SAP maintenance task lists using effort estimation.

The goal is to learn from **planned vs actual maintenance operations** and recommend improved task list parameters such as duration, manpower, and material usage.

---

## 🔍 Problem Statement
SAP maintenance task lists are typically static and do not evolve based on execution feedback. This often leads to inaccurate planning, inefficient resource allocation, and execution delays.

---

## 🧠 Approach (Staged Pipeline)

The system is designed as a dependency-aware learning pipeline:

1. **Synthetic Data Generation**
   - Generate planned task list operations
   - Generate actual maintenance execution data
   - Merge into a supervised training dataset

2. **Time Estimation**
   - Predict actual execution duration from planned attributes

3. **Manpower Estimation**
   - Predict required manpower using predicted duration (chained model)

4. **Material Estimation**
   - Hybrid approach:
     - Rule-based feasibility constraints
     - ML-based material requirement classification

---

## 📂 Repository Structure

sap-tasklist-optimizer-ml/
├── data/
│ ├── planned_tasklist_operations.csv
│ ├── actual_maintenance_operations.csv
│ └── effort_training_dataset.csv
│
├── src/
│ ├── data_generation/
│ │ ├── generate_planned_data.py
│ │ ├── generate_actual_data.py
│ │ └── merge_planned_actual.py
│ │
│ └── estimation/
│ ├── time_estimation.py
│ ├── manpower_estimation.py
│ └── material_estimation.py
│
├── docs/
├── requirements.txt
└── README.md


---

## ▶️ How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt


Run estimation stages:

python src/estimation/time_estimation.py
python src/estimation/manpower_estimation.py
python src/estimation/material_estimation.py

📌 Notes

This project uses schema-driven synthetic data

Designed to be extended with real SAP maintenance data

Focuses on explainability and enterprise-aligned modeling

🚀 Future Work

Gap analysis between planned and predicted values

Automated task list update recommendations

Integration with real SAP maintenance APIs


