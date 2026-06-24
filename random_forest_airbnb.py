"""Random Forest model with grid-search hyperparameter tuning for Airbnb occupancy analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
AIRBNB_FILE = DATA_DIR / "airbnb.df.csv"
AMENITIES_FILE = DATA_DIR / "amenities.df.csv"
RANDOM_STATE = 42

FEATURES = [
    "city", "log_price", "host_response_time", "host_response_rate",
    "host_acceptance_rate", "host_is_superhost", "host_listings_count",
    "host_has_profile_pic", "host_identity_verified", "room_type",
    "accommodates", "minimum_nights", "review_scores_value",
    "instant_bookable", "amenities_count", "essentials", "kitchen",
    "microwave", "patio", "refrigerator", "video_games", "wifi",
    "cleaning", "fan", "fenced", "furniture", "glasses", "laundromat",
    "blinds", "tables", "wine", "grill", "firepit", "utensils",
    "fireplace", "pool", "chairs", "loungers", "staff"
]

PARAM_GRID = {
    "n_estimators": [100, 300, 500],
    "max_depth": [10, 20, 30],
    "min_samples_split": [5, 10, 15],
    "min_samples_leaf": [3, 4, 5],
}


def load_data(airbnb_file: Path = AIRBNB_FILE, amenities_file: Path = AMENITIES_FILE) -> pd.DataFrame:
    airbnb_df = pd.read_csv(airbnb_file, encoding="latin1")
    amenities_df = pd.read_csv(amenities_file, encoding="latin1")
    return pd.concat([airbnb_df, amenities_df], axis=1)


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    data = data.copy()
    data = data[(data["price"] >= 50) & (data["price"] <= 750)]
    data["log_price"] = np.log(data["price"])
    data["occupancy_rate"] = ((30 - data["availability_30"]) / 30 * 100).round(2)

    missing_columns = [col for col in FEATURES + ["occupancy_rate"] if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    model_data = data[FEATURES + ["occupancy_rate"]].copy()
    X = model_data[FEATURES].copy()
    y = model_data["occupancy_rate"]

    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    cat_cols = X.select_dtypes(include=["object"]).columns
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    return X, y


def run_grid_search(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def evaluate_model(model: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    metrics = pd.DataFrame({
        "Model": ["Random Forest Grid Search Best Model"],
        "MSE": [mse],
        "RMSE": [np.sqrt(mse)],
        "MAE": [mean_absolute_error(y_test, y_pred)],
        "R2": [r2_score(y_test, y_pred)],
    })
    return metrics, y_pred


def save_grid_search_results(grid_search: GridSearchCV, output_dir: Path) -> None:
    results = pd.DataFrame(grid_search.cv_results_)
    summary = results[[
        "params", "mean_test_score", "std_test_score",
        "mean_train_score", "rank_test_score"
    ]].sort_values("rank_test_score")
    summary.to_excel(output_dir / "grid_search_results_summary.xlsx", index=False)


def save_feature_importance(model: RandomForestRegressor, feature_names: pd.Index, output_dir: Path) -> pd.DataFrame:
    feature_importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    })
    feature_importance["feature"] = feature_importance["feature"].replace({"log_price": "price"})
    feature_importance = feature_importance.sort_values("importance", ascending=True)
    feature_importance.to_excel(output_dir / "all_feature_importance.xlsx", index=False)
    feature_importance.tail(20).sort_values("importance", ascending=False).to_excel(
        output_dir / "top_20_feature_importance.xlsx", index=False
    )
    return feature_importance


def plot_residual_distribution(y_test: pd.Series, y_pred: np.ndarray, output_dir: Path) -> None:
    residuals = y_test - y_pred
    mean_residual = residuals.mean()

    plt.figure(figsize=(10, 6), dpi=300)
    plt.hist(residuals, bins=30, edgecolor="black", alpha=0.8)
    plt.axvline(x=0, color="black", linestyle="--", linewidth=1, label="Zero error")
    plt.axvline(x=mean_residual, linestyle="-", linewidth=1.5, label=f"Mean: {mean_residual:.2f}")
    plt.title("Residual Distribution: Best Random Forest Model", fontsize=14, fontweight="bold")
    plt.xlabel("Residuals", fontsize=12, fontweight="bold")
    plt.ylabel("Frequency", fontsize=12, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "residual_distribution_best_random_forest.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_feature_importance(feature_importance: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 6), dpi=300)
    plt.barh(feature_importance["feature"], feature_importance["importance"])
    plt.title("Feature Importance: Best Random Forest Model", fontsize=14, fontweight="bold")
    plt.xlabel("Importance", fontsize=12, fontweight="bold")
    plt.ylabel("Feature", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_best_random_forest_all_features.png", dpi=300, bbox_inches="tight")
    plt.close()

    top_20 = feature_importance.tail(20)
    plt.figure(figsize=(6, 4), dpi=300)
    plt.barh(top_20["feature"], top_20["importance"])
    plt.title("Feature Importance: Random Forest", fontsize=12, fontweight="bold")
    plt.xlabel("Importance", fontsize=10, fontweight="bold")
    plt.ylabel("Feature", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_random_forest_top_20.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_data()
    X, y = prepare_data(data)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=RANDOM_STATE
    )

    grid_search = run_grid_search(X_train, y_train)
    best_model = grid_search.best_estimator_

    save_grid_search_results(grid_search, OUTPUT_DIR)
    metrics, y_pred = evaluate_model(best_model, X_test, y_test)
    metrics.to_excel(OUTPUT_DIR / "best_model_metrics.xlsx", index=False)

    feature_importance = save_feature_importance(best_model, X.columns, OUTPUT_DIR)
    plot_residual_distribution(y_test, y_pred, OUTPUT_DIR)
    plot_feature_importance(feature_importance, OUTPUT_DIR)

    print("Best hyperparameters:")
    print(grid_search.best_params_)
    print("\nBest cross-validation R2:")
    print(round(grid_search.best_score_, 4))
    print("\nTest-set performance:")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
