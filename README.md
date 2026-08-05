# MLB Pitcher Injury Risk Prediction

This repository contains a machine learning pipeline for predicting elbow/UCL injury risk in Major League Baseball pitchers using Statcast pitch-level data and MLB injury-list records.

The training script downloads Statcast and injury-list data from the repository, engineers appearance-level and rolling workload features, labels appearances occurring within 60 days before an elbow/UCL injury, balances the dataset, and trains three machine learning models:

* Logistic Regression
* Random Forest
* XGBoost

Each model is evaluated using multiple performance metrics and feature importance analysis to compare its ability to identify pitchers at elevated injury risk.

---

# Requirements

* Python 3.10 or newer
* Git

---

# 1. Clone the repository

```bash
git clone https://github.com/zachjf9/PitcherInjuryRiskAnalysis.git
cd PitcherInjuryRiskAnalysis
```

If you already have the repository cloned:

```bash
git pull origin main
```

---

# 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

# 4. Run the ML training script

For Machine Learning Models Run:

```bash
python MachineLearningModels.py
python3 MachineLearningModels.py
```

---

# Data Processing Pipeline

The pipeline performs the following steps:

1. Downloads Statcast training, validation, and testing datasets.
2. Downloads MLB injury-list data from the 2015–2025 seasons (excluding 2020).
3. Filters injury records for elbow/UCL-related injuries using keyword matching.
4. Aggregates pitch-level Statcast data into appearance-level statistics.
5. Creates rolling performance and workload features.
6. Labels appearances that occurred within **60 days** before an elbow/UCL injury.
7. Balances the training data through undersampling and SMOTE oversampling.
8. Trains and evaluates three machine learning models.

---

# Common Issues

## PowerShell blocks virtual environment activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

## Missing Python packages

Reinstall the required dependencies:

```bash
pip install -r requirements.txt
```

## Unable to download datasets

The training script downloads all datasets directly from the GitHub repository. Verify that:

* You have an active internet connection.
* The repository is accessible.
* The dataset URLs have not changed.

---

# Deactivate the virtual environment

```bash
deactivate
```
