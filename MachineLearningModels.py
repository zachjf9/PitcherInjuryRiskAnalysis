# Imports

from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")

# Data Filtering

BASE_URL = "https://media.githubusercontent.com/media/zachjf9/PitcherInjuryRiskAnalysis/refs/heads/main/"
WINDOW_DAYS = 60
YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
MAJOR_PITCHES = ["FF", "SI", "SL", "CH", "CU", "FC", "KC", "FS", "ST"]

FEATURES = [
    "avg_velocity", "max_velocity", "velocity_std", "avg_spin",
    "spin_std", "pitch_count", "ff_avg_velocity", "ff_avg_spin",
    "vel_last5", "spin_last5", "count_last5",
    "vel_last10", "spin_last10", "vel_change", "spin_change",
    "count_change", "vel_trend", "spin_trend", "vel_std_last5",
    "spin_std_last5", "FF", "SI", "SL", "CH", "CU", "FC", "KC", "FS", "ST",
    "relative_vel_season",
    "relative_spin_season",
    "days_since_last_appearance",
    "pitch_count_last10_games",
    "pitch_count_last30_games",
    "acute_workload",
    "chronic_workload",
    "acwr"
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


print("Processing Injury Data")

all_injuries = []
keywords = ["elbow", "ucl", "ulnar", "tommy john", "arm"]
keyword_regex = "|".join(keywords)

injury_cols = ["InjuryOne", "InjuryTwo", "InjuryThree",
               "InjuryFour", "InjuryFive", "InjurySix", "InjurySeven"]

for year in YEARS:
    url = f"{BASE_URL}{year}IL.csv"
    try:
        df_year = pd.read_csv(url)
        statcast_names = df_year["Player"].apply(convert_name)

        for col in injury_cols:
            if col in df_year.columns:
                is_elbow = df_year[col].astype(
                    str).str.lower().str.contains(keyword_regex, na=False)
                matched_rows = df_year[is_elbow].copy()
                if not matched_rows.empty:
                    dates = matched_rows[col].apply(extract_start_date)

                    temp = pd.DataFrame({
                        "statcast_name": statcast_names[is_elbow],
                        "injury_start": dates,
                        "injury_text": matched_rows[col],
                        "season": year
                    })

                    temp = temp.dropna(subset=["injury_start"])
                    all_injuries.append(temp)

        print(f"{year}IL.csv Loaded")
    except Exception as e:
        print(f"Warning: Could not load {year}IL.csv from {url}. Error: {e}")

injuries_df = pd.concat(all_injuries, ignore_index=True)
injuries_df["injury_start"] = pd.to_datetime(injuries_df["injury_start"])

print(f"\nTotal isolated Elbow/UCL injury events: {injuries_df.shape[0]}")

# Aggregation


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

    ff_metrics = df[df['pitch_type'] == 'FF'].groupby(['pitcher', 'game_date']).agg(
        ff_avg_velocity=('release_speed', 'mean'),
        ff_avg_spin=('release_spin_rate', 'mean')
    ).reset_index()
    appearance_df = appearance_df.merge(
        ff_metrics, on=['pitcher', 'game_date'], how='left')
    appearance_df['ff_avg_velocity'] = appearance_df['ff_avg_velocity'].fillna(
        0)
    appearance_df['ff_avg_spin'] = appearance_df['ff_avg_spin'].fillna(0)

    pitch_usage = pd.crosstab(
        [df["pitcher"], df["player_name"], df["game_date"]],
        df["pitch_type"],
        normalize="index"
    ) * 100
    pitch_usage = pitch_usage.reset_index()

    for pitch in MAJOR_PITCHES:
        if pitch not in pitch_usage.columns:
            pitch_usage[pitch] = 0

    pitch_usage = pitch_usage[["pitcher",
                               "player_name", "game_date"] + MAJOR_PITCHES]
    appearance_df = appearance_df.merge(
        pitch_usage, on=["pitcher", "player_name", "game_date"], how="left")

    appearance_df["game_date"] = pd.to_datetime(appearance_df["game_date"])
    appearance_df = appearance_df.sort_values(["pitcher", "game_date"])

    # Rolling Calculations
    appearance_df["vel_last5"] = appearance_df.groupby("pitcher")["avg_velocity"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    appearance_df["spin_last5"] = appearance_df.groupby("pitcher")["avg_spin"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    appearance_df["count_last5"] = appearance_df.groupby("pitcher")[
        "pitch_count"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    appearance_df["vel_last10"] = appearance_df.groupby("pitcher")["avg_velocity"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    appearance_df["spin_last10"] = appearance_df.groupby("pitcher")["avg_spin"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    appearance_df["vel_std_last5"] = appearance_df.groupby("pitcher")[
        "avg_velocity"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())
    appearance_df["spin_std_last5"] = appearance_df.groupby(
        "pitcher")["avg_spin"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())
    appearance_df[['vel_std_last5', 'spin_std_last5']] = appearance_df[[
        'vel_std_last5', 'spin_std_last5']].fillna(0)

    # Trend Features
    appearance_df["vel_change"] = appearance_df["avg_velocity"] - \
        appearance_df["vel_last5"]
    appearance_df["spin_change"] = appearance_df["avg_spin"] - \
        appearance_df["spin_last5"]
    appearance_df["count_change"] = appearance_df["pitch_count"] - \
        appearance_df["count_last5"]
    appearance_df["vel_trend"] = appearance_df["avg_velocity"] - \
        appearance_df["vel_last10"]
    appearance_df["spin_trend"] = appearance_df["avg_spin"] - \
        appearance_df["spin_last10"]

    # Relative Performance
    appearance_df['cum_v'] = appearance_df.groupby(
        'pitcher')['avg_velocity'].transform(lambda x: x.shift(1).expanding().mean())
    appearance_df['cum_s'] = appearance_df.groupby(
        'pitcher')['avg_spin'].transform(lambda x: x.shift(1).expanding().mean())
    appearance_df['relative_vel_season'] = (
        appearance_df['avg_velocity'] - appearance_df['cum_v']).fillna(0)
    appearance_df['relative_spin_season'] = (
        appearance_df['avg_spin'] - appearance_df['cum_s']).fillna(0)
    appearance_df.drop(columns=['cum_v', 'cum_s'], inplace=True)

    # Workload and Rest
    appearance_df['days_since_last_appearance'] = appearance_df.groupby(
        'pitcher')['game_date'].diff().dt.days.fillna(0)
    appearance_df['pitch_count_last10_games'] = appearance_df.groupby('pitcher')[
        'pitch_count'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(0)
    appearance_df['pitch_count_last30_games'] = appearance_df.groupby('pitcher')[
        'pitch_count'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean()).fillna(0)

    return appearance_df


def advanced_workload_features(df):

    df = df.sort_values(['pitcher', 'game_date']).copy()

    df['acute_workload'] = (
        df.groupby('pitcher')['pitch_count']
          .transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
    )

    df['chronic_workload'] = (
        df.groupby('pitcher')['pitch_count']
          .transform(lambda x: x.shift(1).rolling(28, min_periods=1).sum())
    )

    df['acute_workload'] = df['acute_workload'].fillna(0)
    df['chronic_workload'] = df['chronic_workload'].fillna(0)

    df['acwr'] = df['acute_workload'] / (
        (df['chronic_workload'] / 4) + 1e-6
    )

    df['acwr'] = (
        df['acwr']
        .replace([np.inf, -np.inf], 1.0)
        .fillna(1.0)
    )

    return df


def label_dataset_optimized(df, injuries_lookup, window_days=60):
    df_labeled = df.copy()

    merged = df_labeled.merge(
        injuries_lookup[['statcast_name', 'injury_start']],
        left_on='player_name',
        right_on='statcast_name',
        how='left'
    )

    merged['days_until'] = (merged['injury_start'] -
                            merged['game_date']).dt.days

    at_risk = merged[(merged['days_until'] >= 0) & (
        merged['days_until'] <= window_days)]

    injury_indices = at_risk[['pitcher', 'game_date']].drop_duplicates()
    df_labeled['injury_label'] = 0

    df_labeled = df_labeled.merge(
        injury_indices.assign(injury_label_new=1),
        on=['pitcher', 'game_date'],
        how='left'
    )
    df_labeled['injury_label'] = df_labeled['injury_label_new'].fillna(
        0).astype(int)
    df_labeled.drop(columns=['injury_label_new'], inplace=True)

    print(df_labeled['injury_label'].value_counts())
    return df_labeled


def undersample_dataset_by_pitcher(original_df, train_indices, features_list, every_nth_negative=3, random_state=42):
    train_subset = original_df.loc[train_indices].copy()

    positives_df = train_subset[train_subset["injury_label"] == 1].copy()
    negatives_df = train_subset[train_subset["injury_label"] == 0].copy()

    sampled_negatives_list = []
    for pitcher_id, group in negatives_df.groupby('pitcher'):
        group_sorted = group.sort_values('game_date').reset_index(drop=True)
        sampled_negatives_list.append(group_sorted.iloc[::every_nth_negative])

    if sampled_negatives_list:
        sampled_negatives_df = pd.concat(sampled_negatives_list)
    else:
        sampled_negatives_df = pd.DataFrame(columns=train_subset.columns)

    balanced_df = pd.concat([positives_df, sampled_negatives_df])

    balanced_df = balanced_df.sample(
        frac=1,
        random_state=random_state
    ).reset_index(drop=True)

    print(
        f"\nTraining set after undersampling - Every {every_nth_negative} negative appearance per pitcher:")
    print(balanced_df["injury_label"].value_counts())

    X_sampled = balanced_df[features_list]
    y_sampled = balanced_df["injury_label"]

    return X_sampled, y_sampled


print("Streaming Data:")
train_features = process_statcast_file(
    "https://media.githubusercontent.com/media/zachjf9/PitcherInjuryRiskAnalysis/refs/heads/main/statcast_train.csv")
val_features = process_statcast_file(
    "https://media.githubusercontent.com/media/zachjf9/PitcherInjuryRiskAnalysis/refs/heads/main/statcast_val.csv")
test_features = process_statcast_file(
    "https://media.githubusercontent.com/media/zachjf9/PitcherInjuryRiskAnalysis/refs/heads/main/statcast_test.csv")

train_labeled = advanced_workload_features(
    label_dataset_optimized(train_features, injuries_df, WINDOW_DAYS))
val_labeled = advanced_workload_features(
    label_dataset_optimized(val_features, injuries_df, WINDOW_DAYS))
test_labeled = advanced_workload_features(
    label_dataset_optimized(test_features, injuries_df, WINDOW_DAYS))

train_idx, val_idx = train_test_split(
    train_labeled.index,
    stratify=train_labeled["injury_label"],
    test_size=0.2,
    random_state=42
)

X_train_sampled, y_train_sampled = undersample_dataset_by_pitcher(
    train_labeled,
    train_idx,
    FEATURES,
    every_nth_negative=2,
    random_state=42
)

X_train_sampled = X_train_sampled.fillna(X_train_sampled.mean())

sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(
    X_train_sampled, y_train_sampled)

print("\nTraining set after SMOTE:")
print(y_train_smote.value_counts())

X_val_split = train_labeled.loc[val_idx, FEATURES]
y_val_split = train_labeled.loc[val_idx, "injury_label"]

X_val_split = X_val_split.fillna(X_train_smote.mean())

print("Finished data preparation")

comparison = train_labeled.groupby("injury_label")[FEATURES].mean().T

comparison["difference"] = (
    comparison[1] - comparison[0]
)

comparison["percent_change"] = (
    comparison["difference"] /
    comparison[0].replace(0, 1)
)

comparison = comparison.sort_values(
    "percent_change",
    ascending=False
)

print("\nFeature Differences: Injured vs Healthy")
print(comparison.head(20))

# Models and Evaluation

print("Logistic Regression")

custom_weights = {0: 1, 1: 1}

model = LogisticRegression(
    random_state=42,
    max_iter=5000,
    solver='liblinear',
    class_weight=custom_weights
)

model.fit(X_train_smote, y_train_smote)

probs = model.predict_proba(X_val_split)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val_split, probs)

threshold_results = pd.DataFrame({
    'Threshold': thresholds,
    'Precision': precisions[:-1],
    'Recall': recalls[:-1]
})

threshold_results['F1_Score'] = 2 * (threshold_results['Precision'] * threshold_results['Recall']) / (
    threshold_results['Precision'] + threshold_results['Recall'])
threshold_results = threshold_results.fillna(0)

# Recall Threshold
min_recall_target = 0.50
eligible_thresholds = threshold_results[threshold_results['Recall']
                                        >= min_recall_target]

if not eligible_thresholds.empty:
    chosen_threshold = eligible_thresholds.loc[eligible_thresholds['Precision'].idxmax(
    ), 'Threshold']
else:
    print(
        f"Warning: No threshold achieves a recall of at least {min_recall_target}. Reverting to F1-score maximization.")
    chosen_threshold = threshold_results.loc[threshold_results['F1_Score'].idxmax(
    ), 'Threshold']

print(
    f"Chosen Decision Threshold (Logistic Regression): {chosen_threshold:.4f}")

# Evaluation
lr_preds_at_threshold = (probs >= chosen_threshold).astype(int)

print("\nClassification Report")
print(classification_report(y_val_split, lr_preds_at_threshold))

print("ROC AUC")
print(f"{roc_auc_score(y_val_split, probs):.4f}")

print("\nConfusion Matrix")
cm_lr = confusion_matrix(y_val_split, lr_preds_at_threshold)
print(cm_lr)

importance = pd.DataFrame({
    "Feature": X_train_smote.columns,
    "Importance": model.coef_[0]
}).sort_values("Importance", ascending=False)

print("\nFeature Importance")
print(importance)
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature',
            data=importance.head(15), palette='viridis')
plt.title("Top Feature Importances (Logistic Regression)", fontsize=16)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Random Forest Model")

X_train, y_train = X_train_smote, y_train_smote
X_val, y_val = X_val_split, y_val_split

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

rf_probs = model.predict_proba(X_val)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_val, rf_probs)

threshold_results = pd.DataFrame({
    'Threshold': thresholds,
    'Precision': precisions[:-1],
    'Recall': recalls[:-1]
})

threshold_results['F1_Score'] = 2 * (threshold_results['Precision'] * threshold_results['Recall']) / (
    threshold_results['Precision'] + threshold_results['Recall'])
threshold_results = threshold_results.fillna(0)

# Recall Threshold
min_recall_target_rf = 0.7
eligible_thresholds_rf = threshold_results[threshold_results['Recall']
                                           >= min_recall_target_rf]

if not eligible_thresholds_rf.empty:
    chosen_threshold_rf = eligible_thresholds_rf.loc[eligible_thresholds_rf['F1_Score'].idxmax(
    ), 'Threshold']
else:
    print(
        f"Warning: No threshold achieves a recall of at least {min_recall_target_rf}. Reverting to global F1-score maximization.")
    chosen_threshold_rf = threshold_results.loc[threshold_results['F1_Score'].idxmax(
    ), 'Threshold']

print(f"Chosen Decision Threshold (Random Forest): {chosen_threshold_rf:.4f}")

# Evaluation
rf_preds_at_threshold = (rf_probs >= chosen_threshold_rf).astype(int)

print("Classification Report")
print(classification_report(y_val, rf_preds_at_threshold))

print("ROC AUC Score")
print(f"{roc_auc_score(y_val, rf_probs):.4f}\n")

print("Confusion Matrix")
print(confusion_matrix(y_val, rf_preds_at_threshold))

mportance = pd.DataFrame(
    {"Feature": X_train.columns, "Importance": model.feature_importances_}
).sort_values("Importance", ascending=False)

print("\nTop Feature Importances")
print(importance.head(15))
plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=importance.head(
    15), palette='viridis', hue='Feature', legend=False)
plt.title("Top Feature Importances (Random Forest)", fontsize=16)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("XGBoost Model")

X_train, y_train = X_train_smote, y_train_smote
X_val, y_val = X_val_split, y_val_split

count_class_0 = (y_train == 0).sum()
count_class_1 = (y_train == 1).sum()
scale_pos_weight_value = count_class_0 / count_class_1
print(f"Calculated scale_pos_weight: {scale_pos_weight_value:.2f}")

xgb_model = XGBClassifier(
    n_estimators=5000,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight_value,
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=50
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

xgb_probs = xgb_model.predict_proba(X_val)[:, 1]

precisions_xgb, recalls_xgb, thresholds_xgb = precision_recall_curve(
    y_val, xgb_probs)

threshold_results_xgb = pd.DataFrame({
    'Threshold': thresholds_xgb,
    'Precision': precisions_xgb[:-1],
    'Recall': recalls_xgb[:-1]
})

threshold_results_xgb['F1_Score'] = 2 * (threshold_results_xgb['Precision'] * threshold_results_xgb['Recall']) / (
    threshold_results_xgb['Precision'] + threshold_results_xgb['Recall'])
threshold_results_xgb = threshold_results_xgb.fillna(0)

# Recall Threshold
min_recall_target_xgb = 0.7
eligible_thresholds_xgb = threshold_results_xgb[threshold_results_xgb['Recall']
                                                >= min_recall_target_xgb]

if not eligible_thresholds_xgb.empty:
    chosen_threshold_xgb = eligible_thresholds_xgb.loc[eligible_thresholds_xgb['F1_Score'].idxmax(
    ), 'Threshold']
else:
    print(
        f"Warning: No threshold achieves a recall of at least {min_recall_target_xgb}. Reverting to global F1-score maximization.")
    chosen_threshold_xgb = threshold_results_xgb.loc[threshold_results_xgb['F1_Score'].idxmax(
    ), 'Threshold']

print(f"Chosen Decision Threshold (XGBoost): {chosen_threshold_xgb:.4f}")

# Evaluation
xgb_preds_at_threshold = (xgb_probs >= chosen_threshold_xgb).astype(int)

print("Classification Report")
print(classification_report(y_val, xgb_preds_at_threshold))

print("ROC AUC Score")
print(f"{roc_auc_score(y_val, xgb_probs):.4f}\n")

print("Confusion Matrix")
print(confusion_matrix(y_val, xgb_preds_at_threshold))

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTop Feature Importances")
print(importance.head(15))
plt.figure(figsize=(12, 8))
sns.barplot(x="Importance", y="Feature", data=importance.head(
    15), palette='viridis', hue='Feature', legend=False)
plt.title("Top Feature Importances (XGBoost)", fontsize=16)
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()