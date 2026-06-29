import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier, plot_importance

BASE_URL = "https://media.githubusercontent.com/media/zachjf9/PitcherInjuryRiskAnalysis/refs/heads/main/"
WINDOW_DAYS = 10
YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
MAJOR_PITCHES = ["FF", "SI", "SL", "CH", "CU", "FC", "KC", "FS", "ST"]
FEATURES = [
    "avg_velocity", "max_velocity", "velocity_std",
    "avg_spin", "spin_std", "pitch_count",
    "vel_last5", "spin_last5", "count_last5",
    "vel_change", "spin_change", "count_change",
    "FF", "SI", "SL", "CH", "CU", "FC", "KC", "FS", "ST"
]

def convert_name(name):
    parts = name.strip().split()
    if len(parts) < 2:
        return name
    return f"{parts[-1]}, {parts[0]}"

def extract_start_date(text):
    if pd.isna(text):
        return None
    match = re.search(r'(\d{1,2}/\d{1,2}/\d{2})', str(text))
    if match:
        return pd.to_datetime(match.group(1))
    return None

print("--- Processing Injury Data ---")

all_injuries = []

for year in YEARS:
    url = f"{BASE_URL}{year}IL.csv"
    try:
        df_year = pd.read_csv(url)
        df_year["statcast_name"] = df_year["Player"].apply(convert_name)
        df_year["injury_start"] = df_year["InjuryOne"].apply(extract_start_date)

        temp = df_year[["statcast_name", "injury_start"]].copy()
        temp["season"] = year
        all_injuries.append(temp)
        print(f"Loaded: {year}IL.csv")
    except Exception as e:
        print(f"Warning: Could not load {year}IL.csv from {url}. Error: {e}")

injuries_df = pd.concat(all_injuries, ignore_index=True)
injuries_df["injury_start"] = pd.to_datetime(injuries_df["injury_start"])
print(f"Total injury records loaded: {injuries_df.shape[0]}")

print("\n--- Feature Engineering ---")

def process_statcast_file(url_path):
    df = pd.read_csv(url_path)

    appearance_df = (
        df.groupby(["pitcher", "player_name", "game_date"])
        .agg(
            avg_velocity=("release_speed", "mean"),
            max_velocity=("release_speed", "max"),
            velocity_std=("release_speed", "std"),
            avg_spin=("release_spin_rate", "mean"),
            spin_std=("release_spin_rate", "std"),
            pitch_count=("pitch_type", "count")
        )
        .reset_index()
    )

    pitch_usage = pd.crosstab(
        [df["pitcher"], df["player_name"], df["game_date"]],
        df["pitch_type"],
        normalize="index"
    ) * 100
    pitch_usage = pitch_usage.reset_index()

    for pitch in MAJOR_PITCHES:
        if pitch not in pitch_usage.columns:
            pitch_usage[pitch] = 0

    pitch_usage = pitch_usage[["pitcher", "player_name", "game_date"] + MAJOR_PITCHES]
    appearance_df = appearance_df.merge(pitch_usage, on=["pitcher", "player_name", "game_date"], how="left")

    appearance_df["game_date"] = pd.to_datetime(appearance_df["game_date"])
    appearance_df = appearance_df.sort_values(["pitcher", "game_date"])

    # Historical rolling calculations with standard shift and relative delta differences
    appearance_df["vel_last5"] = (
        appearance_df.groupby("pitcher")["avg_velocity"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    appearance_df["spin_last5"] = (
        appearance_df.groupby("pitcher")["avg_spin"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    appearance_df["count_last5"] = (
        appearance_df.groupby("pitcher")["pitch_count"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    appearance_df["vel_change"] = appearance_df["avg_velocity"] - appearance_df["vel_last5"]
    appearance_df["spin_change"] = appearance_df["avg_spin"] - appearance_df["spin_last5"]
    appearance_df["count_change"] = appearance_df["pitch_count"] - appearance_df["count_last5"]

    return appearance_df

print("Streaming features...")
train_features = process_statcast_file("https://media.githubusercontent.com/media/zachjf9/PitcherInjuryRiskAnalysis/refs/heads/main/statcast_train.csv")
val_features   = process_statcast_file("https://media.githubusercontent.com/media/zachjf9/PitcherInjuryRiskAnalysis/refs/heads/main/statcast_val.csv")
test_features  = process_statcast_file("https://media.githubusercontent.com/media/zachjf9/PitcherInjuryRiskAnalysis/refs/heads/main/statcast_test.csv")

print("\n--- Labeling Appearances ---")

def label_dataset(df, injuries_lookup):
    df_labeled = df.copy()
    df_labeled["injury_label"] = 0

    for idx, row in df_labeled.iterrows():
        player = row["player_name"]
        game_date = row["game_date"]

        player_injuries = injuries_lookup[injuries_lookup["statcast_name"] == player]

        for injury_date in player_injuries["injury_start"]:
            days_until = (injury_date - game_date).days

            if 0 <= days_until <= WINDOW_DAYS:
                df_labeled.at[idx, "injury_label"] = 1
                break

    print(df_labeled["injury_label"].value_counts())
    return df_labeled

print("Labeling Train Set...")
train_labeled = label_dataset(train_features, injuries_df)

print("Labeling Validation Set...")
val_labeled = label_dataset(val_features, injuries_df)

print("Labeling Test Set...")
test_labeled = label_dataset(test_features, injuries_df)

# =====================================================================
# XG Boost Model
# =====================================================================

print("\n--- Training XGBoost Model ---")

X_train, y_train = train_labeled[FEATURES], train_labeled["injury_label"]
X_val, y_val     = val_labeled[FEATURES], val_labeled["injury_label"]

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=35,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Evaluation
preds = model.predict(X_val)
probs = model.predict_proba(X_val)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_val, preds))

print("=== ROC AUC ===")
print(f"{roc_auc_score(y_val, probs):.4f}")

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_val, preds))

# Feature Importance Output
importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n=== Feature Importance ===")
print(importance)

plot_importance(model, max_num_features=15)
plt.tight_layout()
plt.show()