import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from cnn_dataset import prepare_features, scale_features, create_cnn_sequences

#Rolling window size
WINDOW = 5
LABEL_COL = "injury_label"

#Load train, validation, and test features
def load_data():
    train_df = pd.read_csv("../pitcher_game_features_train.csv")
    val_df = pd.read_csv("../pitcher_game_features_val.csv")
    test_df = pd.read_csv("../pitcher_game_features_test.csv")

    #Convert dates to datetime objects
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

    return X_train, y_train, X_val, y_val, X_test, y_test

#Build 1D-CNN
def build_model(input_shape):
    model = Sequential([
        #Input sequence:(window, features)
        tf.keras.Input(shape=input_shape),

        #Learn local patterns across consecutive games
        Conv1D(
            filters=64,
            kernel_size=2,
            activation="relu",
            padding="same"),
        Dropout(0.3),

        #Learn higher-level temporal patterns
        Conv1D(
            filters=128,
            kernel_size=2,
            activation="relu",
            padding="same"),
        Dropout(0.3),

        #Convert feature maps into a single vector
        Flatten(),

        #Fully connected layer
        Dense(64, activation="relu"),
        Dropout(0.3),

        #Binary injury prediction
        Dense(1, activation="sigmoid")])

    #Configure optimizer, loss function, and evaluation metrics
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")])

    return model

#Train CNN using early stopping
def train_model(model, X_train, y_train, X_val, y_val):
    #Stop training if validation AUC stops improving
    early_stop = EarlyStopping(
        monitor="val_auc",
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
        verbose=1)

    return history

#Evaluate performance on test set
def evaluate_model(model, X_test, y_test):
    results = model.evaluate(X_test, y_test, verbose=1)

    print("Test results:")
    for name, value in zip(model.metrics_names, results):
        print(f"{name}: {value:.4f}")

def main():
    #Load feature-engineered datasets
    print("Loading data...")
    train_df, val_df, test_df = load_data()

    #Prepare CNN inputs
    print("Preparing CNN datasets...")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_datasets(
        train_df,
        val_df,
        test_df)

    #Display input dimensions
    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)

    #Display number of positive injury labels
    print("Positive train labels:", y_train.sum())
    print("Positive validation labels:", y_val.sum())
    print("Positive test labels:", y_test.sum())

    #Build CNN
    print("Building model...")
    model = build_model(
        input_shape=(WINDOW, X_train.shape[2]))

    #Display network architecture
    model.summary()

    #Show class distribution
    print("Training labels:")
    print(pd.Series(y_train).value_counts())

    print("\nValidation labels:")
    print(pd.Series(y_val).value_counts())

    print("\nTesting labels:")
    print(pd.Series(y_test).value_counts())

    #Train model
    print("Training model...")
    train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val)

    #Evaluate final performance
    print("Evaluating model...")
    evaluate_model(
        model,
        X_test,
        y_test)

if __name__ == "__main__":
    main()