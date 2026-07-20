import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import warnings
warnings.filterwarnings("ignore")
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier, plot_importance

BASE_URL = "https://media.githubusercontent.com/media/zachjf9/PitcherInjuryRiskAnalysis/refs/heads/main/"
WINDOW_DAYS = 30
YEARS = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025]
MAJOR_PITCHES = ["FF", "SI", "SL", "CH", "CU", "FC", "KC", "FS", "ST"]

FEATURES = [
    "avg_velocity","max_velocity","velocity_std","avg_spin",
    "spin_std","pitch_count", "ff_avg_velocity","ff_avg_spin",
    "vel_last5","spin_last5","count_last5",
    "vel_last10","spin_last10","vel_change","spin_change",
    "count_change","vel_trend","spin_trend","vel_std_last5",
    "spin_std_last5","FF","SI","SL","CH","CU","FC","KC","FS","ST",
    "relative_vel_season",
    "relative_spin_season",
    "days_since_last_appearance",
    "pitch_count_last10_games",
    "pitch_count_last30_games"
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

injury_cols = ["InjuryOne", "InjuryTwo", "InjuryThree", "InjuryFour", "InjuryFive", "InjurySix", "InjurySeven"]

for year in YEARS:
    url = f"{BASE_URL}{year}IL.csv"
    try:
        df_year = pd.read_csv(url)
        statcast_names = df_year["Player"].apply(convert_name)

        for col in injury_cols:
            if col in df_year.columns:
                is_elbow = df_year[col].astype(str).str.lower().str.contains(keyword_regex, na=False)
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
    appearance_df = appearance_df.merge(ff_metrics, on=['pitcher', 'game_date'], how='left')
    appearance_df['ff_avg_velocity'] = appearance_df['ff_avg_velocity'].fillna(0)
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

    pitch_usage = pitch_usage[["pitcher", "player_name", "game_date"] + MAJOR_PITCHES]
    appearance_df = appearance_df.merge(pitch_usage, on=["pitcher", "player_name", "game_date"], how="left")

    appearance_df["game_date"] = pd.to_datetime(appearance_df["game_date"])
    appearance_df = appearance_df.sort_values(["pitcher", "game_date"])

    # Rolling Calculations
    appearance_df["vel_last5"] = appearance_df.groupby("pitcher")["avg_velocity"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    appearance_df["spin_last5"] = appearance_df.groupby("pitcher")["avg_spin"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    appearance_df["count_last5"] = appearance_df.groupby("pitcher")["pitch_count"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    appearance_df["vel_last10"] = appearance_df.groupby("pitcher")["avg_velocity"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    appearance_df["spin_last10"] = appearance_df.groupby("pitcher")["avg_spin"].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    appearance_df["vel_std_last5"] = appearance_df.groupby("pitcher")["avg_velocity"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())
    appearance_df["spin_std_last5"] = appearance_df.groupby("pitcher")["avg_spin"].transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())
    appearance_df[['vel_std_last5', 'spin_std_last5']] = appearance_df[['vel_std_last5', 'spin_std_last5']].fillna(0)

    # Trend Features
    appearance_df["vel_change"] = appearance_df["avg_velocity"] - appearance_df["vel_last5"]
    appearance_df["spin_change"] = appearance_df["avg_spin"] - appearance_df["spin_last5"]
    appearance_df["count_change"] = appearance_df["pitch_count"] - appearance_df["count_last5"]
    appearance_df["vel_trend"] = appearance_df["avg_velocity"] - appearance_df["vel_last10"]
    appearance_df["spin_trend"] = appearance_df["avg_spin"] - appearance_df["spin_last10"]

    # Relative Performance
    appearance_df['cum_v'] = appearance_df.groupby('pitcher')['avg_velocity'].transform(lambda x: x.shift(1).expanding().mean())
    appearance_df['cum_s'] = appearance_df.groupby('pitcher')['avg_spin'].transform(lambda x: x.shift(1).expanding().mean())
    appearance_df['relative_vel_season'] = (appearance_df['avg_velocity'] - appearance_df['cum_v']).fillna(0)
    appearance_df['relative_spin_season'] = (appearance_df['avg_spin'] - appearance_df['cum_s']).fillna(0)
    appearance_df.drop(columns=['cum_v', 'cum_s'], inplace=True)

    # Workload and Rest
    appearance_df['days_since_last_appearance'] = appearance_df.groupby('pitcher')['game_date'].diff().dt.days.fillna(0)
    appearance_df['pitch_count_last10_games'] = appearance_df.groupby('pitcher')['pitch_count'].transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean()).fillna(0)
    appearance_df['pitch_count_last30_games'] = appearance_df.groupby('pitcher')['pitch_count'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean()).fillna(0)

    return appearance_df

def add_advanced_features(df):
    df = df.sort_values(['pitcher', 'game_date'])

    df['acute_workload'] = df.groupby('pitcher')['pitch_count'].transform(lambda x: x.rolling(window=7, min_periods=1).sum())

    df['chronic_workload'] = df.groupby('pitcher')['pitch_count'].transform(lambda x: x.rolling(window=28, min_periods=1).sum())

    df['acwr'] = df['acute_workload'] / (df['chronic_workload'] / 4 + 1e-6)

    df['acwr'] = df['acwr'].fillna(1.0)
    return df

def label_dataset_optimized(df, injuries_lookup, window_days=30):
    df_labeled = df.copy()

    merged = df_labeled.merge(
        injuries_lookup[['statcast_name', 'injury_start']],
        left_on='player_name',
        right_on='statcast_name',
        how='left'
    )

    merged['days_until'] = (merged['injury_start'] - merged['game_date']).dt.days

    at_risk = merged[(merged['days_until'] >= 0) & (merged['days_until'] <= window_days)]

    injury_indices = at_risk[['pitcher', 'game_date']].drop_duplicates()
    df_labeled['injury_label'] = 0

    df_labeled = df_labeled.merge(
        injury_indices.assign(injury_label_new=1),
        on=['pitcher', 'game_date'],
        how='left'
    )
    df_labeled['injury_label'] = df_labeled['injury_label_new'].fillna(0).astype(int)
    df_labeled.drop(columns=['injury_label_new'], inplace=True)

    print(df_labeled['injury_label'].value_counts())
    return df_labeled

print("Streaming Data...")
train_features = process_statcast_file("https://media.githubusercontent.com/media/zachjf9/PitcherInjuryRiskAnalysis/refs/heads/main/statcast_train.csv")
val_features   = process_statcast_file("https://media.githubusercontent.com/media/zachjf9/PitcherInjuryRiskAnalysis/refs/heads/main/statcast_val.csv")
test_features  = process_statcast_file("https://media.githubusercontent.com/media/zachjf9/PitcherInjuryRiskAnalysis/refs/heads/main/statcast_test.csv")

train_labeled = add_advanced_features(label_dataset_optimized(train_features, injuries_df, WINDOW_DAYS))
val_labeled = add_advanced_features(label_dataset_optimized(val_features, injuries_df, WINDOW_DAYS))
test_labeled = add_advanced_features(label_dataset_optimized(test_features, injuries_df, WINDOW_DAYS))

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    train_labeled[FEATURES],
    train_labeled["injury_label"],
    test_size=0.2,
    random_state=42,
    stratify=train_labeled["injury_label"]
)

X_train_split = X_train_split.fillna(X_train_split.mean())
X_val_split = X_val_split.fillna(X_val_split.mean())

sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train_split, y_train_split)

print("Finished data preparation.")

print("Logistic Regression")

X_val_eval = val_labeled[FEATURES]
y_val_eval = val_labeled["injury_label"]

custom_weights = {0: 1, 1: 50}

model = LogisticRegression(
    random_state=42,
    max_iter=3000,
    solver='liblinear',
    class_weight=custom_weights
)

X_val_eval_filled = X_val_eval.fillna(X_train_split.mean())

model.fit(X_train_split, y_train_split)

# Evaluation
preds = model.predict(X_val_eval_filled)
probs = model.predict_proba(X_val_eval_filled)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_val_eval, preds))

print("=== ROC AUC ===")
print(f"{roc_auc_score(y_val_eval, probs):.4f}")

print("\n=== Confusion Matrix ===")
cm_lr = confusion_matrix(y_val_eval, preds)
print(cm_lr)

importance = pd.DataFrame({
    "Feature": X_train_split.columns,
    "Importance": model.coef_[0]
}).sort_values("Importance", ascending=False)

print("\n=== Feature Importance ===")
print(importance)
print("\n")

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance.head(15), legend=False)

print("Random Forest Model")

X_train, y_train = X_train_smote, y_train_smote
X_val, y_val = X_val_split, y_val_split

model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
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

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n=== Feature Importance ===")
print(importance)
print("\n")

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance.head(15), legend=False)

print("XGBoost Model")

X_train, y_train = X_train_split, y_train_split
X_val, y_val = X_val_split, y_val_split

count_class_0 = y_train.value_counts()[0]
count_class_1 = y_train.value_counts()[1]
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
    eval_metric="logloss"
)

xgb_model.fit(X_train, y_train)

# Evaluation
preds = xgb_model.predict(X_val)
probs = xgb_model.predict_proba(X_val)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_val, preds))

print("=== ROC AUC ===")
print(f"{roc_auc_score(y_val, probs):.4f}")

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_val, preds))

importance = pd.DataFrame({
    "Feature": X_train.columns,
    "Importance": xgb_model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n=== Feature Importance ===")
print(importance)
print("\n")
plot_importance(xgb_model, max_num_features=15)
plt.tight_layout()
plt.show()