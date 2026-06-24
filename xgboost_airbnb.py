#XGBoost grid search for Airbnb occupancy prediction.

Expected project structure:


from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/xgboost")
AIRBNB_FILE = DATA_DIR / "airbnb.df.csv"
AMENITIES_FILE = DATA_DIR / "amenities.df.csv"

RANDOM_STATE = 123
TEST_SIZE = 0.30
CV_FOLDS = 5

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

PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}


def load_data() -> pd.DataFrame:
    airbnb_df = pd.read_csv(AIRBNB_FILE, encoding="latin1")
    amenities_df = pd.read_csv(AMENITIES_FILE, encoding="latin1")
    return pd.concat([airbnb_df, amenities_df], axis=1)


def preprocess_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = data.copy()
    data = data[(data["price"] >= 50) & (data["price"] <= 750)]
    data["log_price"] = np.log(data["price"])
    data["occupancy_rate"] = ((30 - data["availability_30"]) / 30 * 100).round(2)

    model_data = data[FEATURES + ["occupancy_rate"]].copy()
    x = model_data[FEATURES].copy()
    y = model_data["occupancy_rate"]

    bool_cols = x.select_dtypes(include=["bool"]).columns
    x[bool_cols] = x[bool_cols].astype(int)

    cat_cols = x.select_dtypes(include=["object"]).columns
    x = pd.get_dummies(x, columns=cat_cols, drop_first=True)

    return x, y


def run_grid_search(x_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        cv=CV_FOLDS,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid_search.fit(x_train, y_train)
    return grid_search


def evaluate_model(model: XGBRegressor, x_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)

    metrics = pd.DataFrame(
        {
            "Model": ["XGBoost Grid Search Best Model"],
            "MSE": [mse],
            "RMSE": [np.sqrt(mse)],
            "MAE": [mean_absolute_error(y_test, y_pred)],
            "R2": [r2_score(y_test, y_pred)],
        }
    )
    return metrics, y_pred


def save_grid_search_results(grid_search: GridSearchCV) -> pd.DataFrame:
    results = pd.DataFrame(grid_search.cv_results_)
    summary = results[
        ["params", "mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    summary.to_excel(OUTPUT_DIR / "xgboost_grid_search_results_summary.xlsx", index=False)
    return summary


def plot_residual_distribution(y_test: pd.Series, y_pred: np.ndarray) -> None:
    residuals = y_test - y_pred
    mean_residual = residuals.mean()
    std_residual = residuals.std()

    plt.figure(figsize=(10, 6), dpi=300)
    plt.hist(residuals, bins=30, edgecolor="black", alpha=0.8)
    plt.axvline(x=0, linestyle="--", linewidth=1, label="Zero error")
    plt.axvline(x=mean_residual, linestyle="-", linewidth=1.5, label=f"Mean: {mean_residual:.2f}")
    plt.title("Residual Distribution: Best XGBoost Model", fontsize=14, fontweight="bold")
    plt.xlabel("Residuals", fontsize=12, fontweight="bold")
    plt.ylabel("Frequency", fontsize=12, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residual_distribution_best_xgboost.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Mean residual: {mean_residual:.4f}")
    print(f"Standard deviation of residuals: {std_residual:.4f}")


def save_feature_importance(model: XGBRegressor, feature_names: pd.Index) -> pd.DataFrame:
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    feature_importance["feature"] = feature_importance["feature"].replace({"log_price": "price"})
    feature_importance = feature_importance.sort_values("importance", ascending=True)
    feature_importance.to_excel(OUTPUT_DIR / "xgboost_all_feature_importance.xlsx", index=False)
    return feature_importance


def plot_feature_importance(feature_importance: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    top_features = feature_importance.tail(top_n)

    plt.figure(figsize=(6, 4), dpi=300)
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Feature Importance: XGBoost", fontsize=12, fontweight="bold")
    plt.xlabel("Importance", fontsize=10, fontweight="bold")
    plt.ylabel("Feature", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance_xgboost_top_20.png", dpi=300, bbox_inches="tight")
    plt.close()

    top_features = top_features.sort_values("importance", ascending=False)
    top_features.to_excel(OUTPUT_DIR / "xgboost_top_20_feature_importance.xlsx", index=False)
    return top_features


def print_results(grid_search: GridSearchCV, metrics: pd.DataFrame, top_features: pd.DataFrame) -> None:
    print("Best XGBoost hyperparameters:")
    for key, value in grid_search.best_params_.items():
        print(f"{key}: {value}")

    print("\nBest cross-validation R²:")
    print(f"{grid_search.best_score_:.4f}")

    print("\nPerformance on test data:")
    print(metrics.to_string(index=False))

    print("\nTop 20 important features:")
    print(top_features.to_string(index=False))

    row = metrics.iloc[0]
    print("\nSuggested manuscript sentence:")
    print(
        "Grid Search with five-fold cross-validation selected the following XGBoost "
        f"hyperparameters: n_estimators = {grid_search.best_params_['n_estimators']}, "
        f"max_depth = {grid_search.best_params_['max_depth']}, "
        f"learning_rate = {grid_search.best_params_['learning_rate']}, "
        f"subsample = {grid_search.best_params_['subsample']}, and "
        f"colsample_bytree = {grid_search.best_params_['colsample_bytree']}. "
        f"The optimized XGBoost model achieved MAE = {row['MAE']:.2f}, "
        f"MSE = {row['MSE']:.2f}, RMSE = {row['RMSE']:.2f}, "
        f"and R² = {row['R2'] * 100:.2f}% on the test dataset."
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    x, y = preprocess_data(data)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Data shape: {data.shape}")
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")

    grid_search = run_grid_search(x_train, y_train)
    best_model = grid_search.best_estimator_

    grid_summary = save_grid_search_results(grid_search)
    metrics, y_pred = evaluate_model(best_model, x_test, y_test)
    metrics.to_excel(OUTPUT_DIR / "xgboost_best_model_metrics.xlsx", index=False)

    plot_residual_distribution(y_test, y_pred)
    feature_importance = save_feature_importance(best_model, x.columns)
    top_features = plot_feature_importance(feature_importance)

    print("\nTop 10 grid search results:")
    print(grid_summary.head(10).to_string(index=False))
    print_results(grid_search, metrics, top_features)
    print(f"\nOutputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
