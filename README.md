# XGBoost Pitcher Injury Risk Model

This branch contains the XGBoost portion of the MLB pitcher injury-risk project. The training script processes Statcast pitch data, creates appearance-level and rolling workload features, labels appearances that occur within 30 days before an elbow/UCL injury, and trains an XGBoost classifier.

## Requirements

- Python 3.10 or newer
- Git
- The Statcast and injury-list CSV files included in this branch

The script expects the following files to be in the same folder as `train_xgboost.py`:

```text
statcast_train.csv
statcast_val.csv
statcast_test.csv
2015IL.csv
2016IL.csv
2017IL.csv
2018IL.csv
2019IL.csv
2021IL.csv
2022IL.csv
2023IL.csv
2024IL.csv
2025IL.csv
```

## 1. Clone the XGBoost branch

```bash
git clone --branch feature/Anas --single-branch https://github.com/zachjf9/PitcherInjuryRiskAnalysis.git
cd PitcherInjuryRiskAnalysis
```

If the repository has already been cloned, switch to the branch and update it:

```bash
git fetch origin
git checkout feature/Anas
git pull origin feature/Anas
```

## 2. Create a virtual environment

### PowerShell

```powershell
python -m venv anas
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\anas\Scripts\Activate.ps1
```

### Command Prompt

```cmd
python -m venv anas
anas\Scripts\activate.bat
```

### macOS or Linux

```bash
python3 -m venv anas
source anas/bin/activate
```

## 3. Install the required packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run the XGBoost model

From the repository root, run:

```bash
python train_xgboost.py
```

## Model settings

The current XGBoost configuration is:

```python
XGBClassifier(
    n_estimators=5000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight_value,
    random_state=42,
    eval_metric="logloss"
)
```

The model uses rolling statistics from the previous 5, 10, and 30 appearances. These rolling features are separate from the 30-day injury prediction window.

## Common issues

### PowerShell blocks virtual-environment activation (issue encountered during setup)

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\anas\Scripts\Activate.ps1
```

### A CSV file cannot be found

Make sure `train_xgboost.py`, the three Statcast files, and all injury-list files are in the repository root. Run the script from that same folder.

### A required package is missing

Install the dependencies again:

```bash
pip install -r requirements.txt
```

## Deactivate the environment

```bash
deactivate
```
