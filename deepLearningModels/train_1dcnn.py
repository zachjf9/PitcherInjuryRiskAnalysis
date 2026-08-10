import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from cnn_dataset import prepare_features, scale_features, create_sequences
from sklearn.utils.class_weight import compute_class_weight

#Rolling window size
WINDOW = 5
LABEL_COL = "injury_label"

#Load train, validation, and test features
def load_data():
    train_df = pd.read_csv("pitcher_game_features_train.csv")
    val_df = pd.read_csv("pitcher_game_features_val.csv")
    test_df = pd.read_csv("pitcher_game_features_test.csv")

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
    val_X = val_X.reindex(columns = feature_cols, fill_value = 0)
    test_X = test_X.reindex(columns = feature_cols, fill_value = 0)

    #Standardize numerical features
    train_scaled, val_scaled, test_scaled, scaler = scale_features(train_X, val_X, test_X)

    #Replace original features with scaled values
    train_scaled_df = train_df.copy()
    val_scaled_df = val_df.copy()
    test_scaled_df = test_df.copy()

    train_scaled_df[feature_cols] = train_scaled
    val_scaled_df[feature_cols] = val_scaled
    test_scaled_df[feature_cols] = test_scaled

    #Convert game-level data into rolling sequences
    X_train, y_train = create_sequences(train_scaled_df, feature_cols, LABEL_COL, WINDOW)
    X_val, y_val = create_sequences(val_scaled_df, feature_cols, LABEL_COL, WINDOW)
    X_test, y_test = create_sequences(test_scaled_df, feature_cols, LABEL_COL, WINDOW)

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols

#Find best class weight that produces highest validation
def find_best_class_weight(X_train, y_train, X_val, y_val, input_shape, positive_weights = None):
    #Test weights
    if positive_weights is None:
        positive_weights = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    best_weight = None
    best_val_pr_auc = -1.0
    best_model = None
    results = []

    print("\nClass-weight search")
    print("--------------------------------------------")

    for positive_weight in positive_weights:
        #Reset seeds, each experiment begins consistently
        np.random.seed(42)
        tf.random.set_seed(42)

        #Clear the previous TensorFlow model from memory
        tf.keras.backend.clear_session()

        class_weight_dict = {0: 1.0, 1: positive_weight}

        print(f"\nTraining with class weights: " f"{class_weight_dict}")

        #Build a new model for this weight
        model = build_model(
            input_shape = input_shape)

        #Train the model
        history = train_model(model, X_train, y_train, X_val, y_val, class_weight_dict)

        #Best validation PR-AUC reached during training
        val_pr_auc = max(history.history["val_pr_auc"])

        #Best validation ROC-AUC for reference
        val_auc = max(history.history["val_auc"])

        results.append({
            "positive_weight": positive_weight,
            "val_pr_auc": val_pr_auc,
            "val_auc": val_auc})

        print(f"Best validation PR-AUC: {val_pr_auc:.4f}")
        print(f"Best validation ROC-AUC: {val_auc:.4f}")

        #Save the strongest model and weight
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_weight = positive_weight
            best_model = model

    print("\nClass-weight results")
    print("--------------------------------------------")

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index = False))

    print("\nSelected class weight")
    print("--------------------------------------------")
    print(f"Class 0 weight: 1.0")
    print(f"Class 1 weight: {best_weight}")
    print(f"Validation PR-AUC: {best_val_pr_auc:.4f}")

    return best_model, best_weight, results_df

#Build 1D-CNN
def build_model(input_shape):
    model = Sequential([
        #Input sequence:(window, features)
        tf.keras.Input(shape = input_shape),

        #Learn local patterns across consecutive games
        Conv1D(filters = 32, kernel_size = 2, activation = "relu", padding = "same"),
        Dropout(0.3),

        #Learn higher-level temporal patterns
        Conv1D(filters = 64, kernel_size = 2, activation = "relu", padding = "same"),
        Dropout(0.3),

        #Convert feature maps into a single vector
        GlobalAveragePooling1D(),

        #Fully connected layer
        Dense(32, activation = "relu"),
        Dropout(0.3),

        #Binary injury prediction
        Dense(1, activation = "sigmoid")])

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)

    #Configure optimizer, loss function, and evaluation metrics
    model.compile(
        optimizer = optimizer,
        loss = "binary_crossentropy",
        metrics = [
            "accuracy",
            tf.keras.metrics.AUC(name = "auc"),
            tf.keras.metrics.AUC(name = "pr_auc", curve = "PR"),
            tf.keras.metrics.Precision(name = "precision"),
            tf.keras.metrics.Recall(name = "recall")])

    return model

#Train CNN using early stopping
def train_model(model, X_train, y_train, X_val, y_val, class_weight_dict):
    #Stop training if validation AUC stops improving
    early_stop = EarlyStopping(
        monitor = "val_pr_auc",
        patience = 5,
        mode = "max",
        restore_best_weights = True)

    history = model.fit(
        X_train,
        y_train,
        validation_data = (X_val, y_val),
        epochs = 50,
        batch_size = 32,
        callbacks = [early_stop],
        class_weight = class_weight_dict,
        verbose = 1)

    return history

#Find best threshold
def find_best_threshold(model, X_val, y_val):
    #Generate validation probabilities
    probabilities = model.predict(X_val, verbose = 0).ravel()

    best_threshold = 0.5
    best_f1 = float("-inf")
    best_precision = 0.0
    best_recall = 0.0

    print("\nValidation threshold results")
    print("-" * 45)

    #Test thresholds from 0.05 to 0.95
    for threshold in np.arange(0.05, 1.00, 0.05):
        predictions = (probabilities >= threshold).astype(int)

        precision = precision_score(y_val, predictions, zero_division = 0)
        recall = recall_score(y_val, predictions, zero_division = 0)
        current_f1 = f1_score(y_val, predictions, zero_division = 0)

        print(
            f"Threshold: {threshold:.2f} | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {current_f1:.4f}")

        # ave the threshold with the best validation F1
        if current_f1 > best_f1:
            best_threshold = threshold
            best_f1 = current_f1
            best_precision = precision
            best_recall = recall

    print("\nSelected validation threshold")
    print("-" * 45)
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall: {best_recall:.4f}")
    print(f"F1 Score: {best_f1:.4f}")

    return best_threshold

#Evaluate performance on test set
def evaluate_model(model, X_test, y_test, threshold = 0.5):
    #Predicted probabilities
    probabilities = model.predict(X_test, verbose = 0).ravel()

    #Convert probabilities to binary predictions
    predictions = (probabilities >= threshold).astype(int)

    #Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)
    precision = precision_score(y_test, predictions, zero_division = 0)
    recall = recall_score(y_test, predictions, zero_division = 0)
    f1 = f1_score(y_test, predictions, zero_division = 0)

    #Confusion matrix
    cm = confusion_matrix(y_test, predictions, labels = [0, 1])

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
            target_names = ["No Injury", "Injury"],
            zero_division = 0))

    return {
        "accuracy": accuracy,
        "auc": auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm}

def permutation_feature_importance(
        model,
        X,
        y,
        feature_cols,
        n_repeats = 5,
        top_n = 15):

    #Get baseline model predictions
    baseline_probabilities = model.predict(X, verbose = 0).ravel()

    #Calculate baseline ROC-AUC
    baseline_auc = roc_auc_score(y, baseline_probabilities)

    print(f"Baseline ROC-AUC: {baseline_auc:.4f}")

    results = []

    rng = np.random.default_rng(42)

    #Test each feature individually
    for feature_index, feature_name in enumerate(feature_cols):

        auc_drops = []

        for _ in range(n_repeats):
            #Copy the original sequence data
            X_permuted = X.copy()

            #Shuffle this feature across samples
            permutation = rng.permutation(len(X_permuted))

            X_permuted[:, :, feature_index] = (X_permuted[permutation, :, feature_index])

            #Predict using shuffled feature
            permuted_probabilities = model.predict(X_permuted, verbose = 0).ravel()

            #Calculate ROC-AUC after shuffling
            permuted_auc = roc_auc_score(y, permuted_probabilities)

            #Importance = loss in ROC-AUC
            auc_drop = baseline_auc - permuted_auc

            auc_drops.append(auc_drop)

        #Average importance across repeated shuffles
        results.append({"Feature": feature_name, "Importance": np.mean(auc_drops)})

    importance_df = pd.DataFrame(results)

    #Sort from most important to least important
    importance_df = (importance_df.sort_values("Importance", ascending = False).reset_index(drop = True))

    #Keep top N
    top_features = importance_df.head(top_n)

    #Create horizontal bar graph
    plt.figure(figsize = (10, 7))

    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
    plt.barh(top_features["Feature"][::-1], top_features["Importance"][::-1], color = colors)

    plt.xlabel("Decrease in ROC-AUC")
    plt.ylabel("Feature")
    plt.title("Top Feature Importances (1D-CNN)")

    plt.tight_layout()
    plt.show()

    return importance_df

def main():
    #Load feature-engineered datasets
    print("Loading data...")
    train_df, val_df, test_df = load_data()

    #Prepare CNN inputs
    #print("Preparing CNN datasets...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_datasets(train_df, val_df, test_df)

    #Display input dimensions
    '''print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("X_test shape:", X_test.shape)'''

    #Show class distribution
    '''print("Training labels:")
    print(pd.Series(y_train).value_counts())

    print("\nValidation labels:")
    print(pd.Series(y_val).value_counts())

    print("\nTesting labels:")
    print(pd.Series(y_test).value_counts())'''

    #Calculate weights after sequence labels
    model, best_positive_weight, weight_results = find_best_class_weight(
        X_train,
        y_train,
        X_val,
        y_val,
        input_shape = (WINDOW, X_train.shape[2]))

    class_weight_dict = {0: 1.0, 1: best_positive_weight}

    print("\nSelected class weights:")
    print(class_weight_dict)

    #Display network architecture
    #model.summary()

    #Threshold using validation data
    best_threshold = find_best_threshold(model, X_val, y_val)

    #Evaluate final performance
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test, threshold = best_threshold)

    importance_df = permutation_feature_importance(model, X_test, y_test, feature_cols, n_repeats = 3, top_n = 15)

if __name__ == "__main__":
    main()