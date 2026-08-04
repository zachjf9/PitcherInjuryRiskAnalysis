import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_features(df, label_col="injury_label"):
    df = df.copy()
    df = df.sort_values(["pitcher", "season", "game_date"]).reset_index(drop = True)

    #Target label
    y = df[label_col].copy()

    #Drop columns that should not be model inputs
    drop_cols = ["game_date", "pitcher", "player_name", "name_clean", "season", label_col]

    X = df.drop(columns = drop_cols, errors = "ignore")

    #Keep only numeric columns
    X = X.select_dtypes(include = ["number"])

    #Handle bad values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    feature_cols = X.columns.tolist()

    return X, y, feature_cols

def scale_features(train_X, val_X, test_X):
    #Initialize scaler
    scaler = StandardScaler()

    #Learn scaling parameters from training set
    train_scaled = scaler.fit_transform(train_X)

    #Apply same transformation
    val_scaled = scaler.transform(val_X)
    test_scaled = scaler.transform(test_X)

    return train_scaled, val_scaled, test_scaled, scaler

def create_sequences(df, feature_cols, label_col="injury_label", window = 5):
    X_sequences = []
    y_labels = []

    df = df.copy()

    #Ensure games are processed in chronological order
    df = df.sort_values(["pitcher", "season", "game_date"]).reset_index(drop = True)

    #Create sequences separately for each pitcher and season
    for (_, _), group in df.groupby(["pitcher", "season"], sort = False):
        group = group.sort_values("game_date")

        #Extract feature matrix and labels
        X = group[feature_cols].to_numpy(dtype = np.float32)
        y = group[label_col].to_numpy(dtype = np.int32)

        if len(group) <= window:
            continue

        #Create rolling windows of previous games
        for i in range(window, len(group)):
            X_sequences.append(X[i - window:i])
            y_labels.append(y[i])

    return np.array(X_sequences, dtype = np.float32), np.array(y_labels, dtype = np.int32)