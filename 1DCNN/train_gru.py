import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from cnn_dataset import (prepare_features, scale_features, create_cnn_sequences)

#Rolling window
WINDOW = 10
LABEL_COL = "injury_label"

#Load train, validation, and test features
def load_data():
    train_df = pd.read_csv("../pitcher_game_features_train.csv")
    val_df = pd.read_csv("../pitcher_game_features_val.csv")
    test_df = pd.read_csv("../pitcher_game_features_test.csv")

    #Convert game dates from strings to datetime objects
    train_df["game_date"] = pd.to_datetime(train_df["game_date"])
    val_df["game_date"] = pd.to_datetime(val_df["game_date"])
    test_df["game_date"] = pd.to_datetime(test_df["game_date"])

    return train_df, val_df, test_df

#Prepare features, scale data, and create CNN sequences
def prepare_datasets(train_df, val_df, test_df):
    #Separate predictors and labels
    train_X, train_y, feature_cols = prepare_features(train_df, LABEL_COL)
    val_X, val_y, _ = prepare_features(val_df, LABEL_COL)
    test_X, test_y, _ = prepare_features(test_df, LABEL_COL)

    #Ensure validation and test contain the same feature columns as training
    val_X = val_X.reindex(columns=feature_cols, fill_value=0)
    test_X = test_X.reindex(columns=feature_cols, fill_value=0)

    #Standardize numerical features
    train_scaled, val_scaled, test_scaled, scaler = scale_features(
        train_X,
        val_X,
        test_X)

    #Replace original features with scaled values
    train_scaled_df = train_df.copy()
    val_scaled_df = val_df.copy()
    test_scaled_df = test_df.copy()

    train_scaled_df[feature_cols] = train_scaled
    val_scaled_df[feature_cols] = val_scaled
    test_scaled_df[feature_cols] = test_scaled

    #Convert game-level data into rolling sequences
    X_train, y_train = create_cnn_sequences(
        train_scaled_df,
        feature_cols,
        LABEL_COL,
        WINDOW)

    X_val, y_val = create_cnn_sequences(
        val_scaled_df,
        feature_cols,
        LABEL_COL,
        WINDOW)

    X_test, y_test = create_cnn_sequences(
        test_scaled_df,
        feature_cols,
        LABEL_COL,
        WINDOW)

    return (X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols)

def calculate_class_weights(y_train, max_positive_weight = 3):
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train)

    class_weight_dict = {
        0: float(weights[0]),
        1: min(float(weights[1]), max_positive_weight)}

    return class_weight_dict

#Build GRU
def build_model(input_shape):
    model = Sequential([
        tf.keras.Input(shape=input_shape),

        #First GRU reads the full sequence and returns one hidden-state vector for each game
        GRU(
            units=64,
            return_sequences=True),
        Dropout(0.3),

        #Second GRU summarizes the entire five-game sequence into one output vector
        GRU(
            units=32,
            return_sequences=False),
        Dropout(0.3),

        #Combine the sequence information
        Dense(
            units=32,
            activation="relu"),
        Dropout(0.3),

        #Produce one probability between 0 and 1
        Dense(
            units=1,
            activation="sigmoid")])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.AUC(
                name="pr_auc",
                curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")])

    return model

#Train GRU using early stopping
def train_model(model, X_train, y_train, X_val, y_val, class_weight_dict):
    #Stop training if validation AUC stops improving
    early_stop = EarlyStopping(
        monitor="val_pr_auc",
        patience=5,
        mode="max",
        restore_best_weights=True)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        class_weight=class_weight_dict,
        verbose=1)

    return history

#Evaluate performance on test set
def evaluate_model(model, X_test, y_test, threshold=0.7):
    #Predicted probabilities
    probabilities = model.predict(X_test, verbose=0).ravel()

    #Convert probabilities to binary predictions
    predictions = (probabilities >= threshold).astype(int)

    #Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test,probabilities)
    pr_auc = average_precision_score(y_test, probabilities)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    #Confusion matrix
    cm = confusion_matrix(y_test, predictions)

    tn, fp, fn, tp = cm.ravel()
    print("\nTest Results")
    print("-------------------------")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print(f"PR AUC   : {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nConfusion Matrix")
    print(cm)
    print(f"""
True Negatives : {tn}
False Positives: {fp}
False Negatives: {fn}
True Positives : {tp}
""")

    print("Classification Report")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["No Injury", "Injury"],
            zero_division=0))

    return {
        "accuracy": accuracy,
        "auc": auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm}

def main():
    #Make TensorFlow results more reproducible
    np.random.seed(42)
    tf.random.set_seed(42)

    #Load feature-engineered datasets
    print("Loading data...")
    train_df, val_df, test_df = load_data()

    #Prepare GRU inputs
    print("Preparing GRU datasets...")
    (X_train, y_train, X_val, y_val, X_test, y_test, scaler, feature_cols) = prepare_datasets(
        train_df,
        val_df,
        test_df)

    ''''#Display input dimensions
    print("\nDataset shapes:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)'''

    print("\nTraining labels:")
    print(pd.Series(y_train).value_counts())

    print("\nValidation labels:")
    print(pd.Series(y_val).value_counts())

    print("\nTesting labels:")
    print(pd.Series(y_test).value_counts())

    #Calculate weights after sequence labels
    class_weight_dict = calculate_class_weights(y_train, max_positive_weight = 3)

    print("\nClass weights:")
    print(class_weight_dict)

    #Build GRU
    print("\nBuilding GRU model...")
    model = build_model(input_shape=(WINDOW, X_train.shape[2]))

    #Display network architecture
    model.summary()

    #Train model
    print("\nTraining GRU model...")
    train_model(model, X_train, y_train, X_val, y_val, class_weight_dict)

    #Evaluate model
    print("\nEvaluating GRU model...")
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()