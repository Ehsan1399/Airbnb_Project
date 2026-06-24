
#Deep Neural Network model for Airbnb occupancy rate prediction.



import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")

RANDOM_STATE = 123

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs") / "dnn"
AIRBNB_FILE = DATA_DIR / "airbnb.df.csv"
AMENITIES_FILE = DATA_DIR / "amenities.df.csv"

FEATURES = [
    "city",
    "log_price",
    "host_response_time",
    "host_response_rate",
    "host_acceptance_rate",
    "host_is_superhost",
    "host_listings_count",
    "host_has_profile_pic",
    "host_identity_verified",
    "room_type",
    "accommodates",
    "minimum_nights",
    "review_scores_value",
    "instant_bookable",
    "amenities_count",
    "essentials",
    "kitchen",
    "microwave",
    "patio",
    "refrigerator",
    "video_games",
    "wifi",
    "cleaning",
    "fan",
    "fenced",
    "furniture",
    "glasses",
    "laundromat",
    "blinds",
    "tables",
    "wine",
    "grill",
    "firepit",
    "utensils",
    "fireplace",
    "pool",
    "chairs",
    "loungers",
    "staff",
]


def load_data() -> pd.DataFrame:
    """Load and merge Airbnb listing and amenities datasets."""
    airbnb_df = pd.read_csv(AIRBNB_FILE, encoding="latin1")
    amenities_df = pd.read_csv(AMENITIES_FILE, encoding="latin1")
    return pd.concat([airbnb_df, amenities_df], axis=1)


def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Clean data and prepare predictors and target variable."""
    data = data.copy()

    data = data[(data["price"] >= 50) & (data["price"] <= 750)]
    data["log_price"] = np.log(data["price"])
    data["occupancy_rate"] = ((30 - data["availability_30"]) / 30 * 100).round(2)

    model_data = data[FEATURES + ["occupancy_rate"]].copy()

    X = model_data[FEATURES].copy()
    y = model_data["occupancy_rate"]

    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    cat_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y


def build_dnn(
    input_dim: int,
    hidden_layers: tuple[int, ...],
    dropout_rate: float,
    learning_rate: float,
) -> Sequential:
    """Build and compile a DNN regression model."""
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation="relu", input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    for units in hidden_layers[1:]:
        model.add(Dense(units, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation="linear"))

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )

    return model


def tune_dnn(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
) -> tuple[Sequential, dict, pd.DataFrame]:
    """Tune DNN hyperparameters using a validation set."""
    param_grid = {
        "hidden_layers": [(64, 32), (128, 64, 32)],
        "dropout_rate": [0.2, 0.3],
        "learning_rate": [0.001, 0.0005],
        "batch_size": [32, 64],
    }

    best_score = -np.inf
    best_params = {}
    best_model = None
    results = []

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    for hidden_layers in param_grid["hidden_layers"]:
        for dropout_rate in param_grid["dropout_rate"]:
            for learning_rate in param_grid["learning_rate"]:
                for batch_size in param_grid["batch_size"]:
                    model = build_dnn(
                        input_dim=X_train.shape[1],
                        hidden_layers=hidden_layers,
                        dropout_rate=dropout_rate,
                        learning_rate=learning_rate,
                    )

                    history = model.fit(
                        X_train,
                        y_train,
                        validation_data=(X_val, y_val),
                        epochs=100,
                        batch_size=batch_size,
                        callbacks=[early_stop],
                        verbose=0,
                    )

                    y_val_pred = model.predict(X_val, verbose=0).flatten()
                    val_mse = mean_squared_error(y_val, y_val_pred)
                    val_rmse = np.sqrt(val_mse)
                    val_mae = mean_absolute_error(y_val, y_val_pred)
                    val_r2 = r2_score(y_val, y_val_pred)

                    params = {
                        "hidden_layers": hidden_layers,
                        "dropout_rate": dropout_rate,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "epochs_used": len(history.history["loss"]),
                    }

                    results.append(
                        {
                            **params,
                            "validation_MSE": val_mse,
                            "validation_RMSE": val_rmse,
                            "validation_MAE": val_mae,
                            "validation_R2": val_r2,
                        }
                    )

                    if val_r2 > best_score:
                        best_score = val_r2
                        best_params = params
                        best_model = model

    results_df = pd.DataFrame(results).sort_values("validation_R2", ascending=False)

    if best_model is None:
        raise RuntimeError("DNN tuning failed. No model was selected.")

    return best_model, best_params, results_df


def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: pd.Series) -> dict:
    """Evaluate the final model on the test set."""
    y_pred = model.predict(X_test, verbose=0).flatten()

    mse = mean_squared_error(y_test, y_pred)
    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }


def main() -> None:
    """Run the DNN analysis pipeline."""
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    X, y = preprocess_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_scaled,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    best_model, best_params, tuning_results = tune_dnn(
        X_train_sub,
        y_train_sub,
        X_val,
        y_val,
    )

    metrics = evaluate_model(best_model, X_test_scaled, y_test)

    tuning_results.to_excel(
        OUTPUT_DIR / "dnn_hyperparameter_tuning_results.xlsx",
        index=False,
    )

    metrics_df = pd.DataFrame(
        {
            "Model": ["DNN Tuned Best Model"],
            "Hidden Layers": [best_params["hidden_layers"]],
            "Dropout Rate": [best_params["dropout_rate"]],
            "Learning Rate": [best_params["learning_rate"]],
            "Batch Size": [best_params["batch_size"]],
            "Epochs Used": [best_params["epochs_used"]],
            **{key: [value] for key, value in metrics.items()},
        }
    )

    metrics_df.to_excel(OUTPUT_DIR / "dnn_best_model_metrics.xlsx", index=False)

    print("Final DNN hyperparameters selected:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    print("\nPerformance of best DNN model on test data:")
    print(f"MAE = {metrics['MAE']:.2f}")
    print(f"MSE = {metrics['MSE']:.2f}")
    print(f"RMSE = {metrics['RMSE']:.2f}")
    print(f"R² = {metrics['R2']:.4f} ({metrics['R2'] * 100:.2f}%)")

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
