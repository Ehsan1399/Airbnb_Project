#Data cleaning and preprocessing for the Airbnb occupancy analysis.



from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

CITY_FILES = {
    "Austin": "austin.csv",
    "New Orleans": "new_orleans.csv",
    "New York": "new_york.csv",
    "San Francisco": "san_francisco.csv",
    "Denver": "denver.csv",
}

COLUMNS_TO_DROP = [
    "name",
    "description",
    "neighborhood_overview",
    "host_about",
    "neighbourhood_cleansed",
    "neighbourhood",
    "neighbourhood_group_cleansed",
    "listing_url",
    "picture_url",
    "host_url",
    "host_thumbnail_url",
    "host_picture_url",
    "scrape_id",
    "last_scraped",
    "source",
    "calendar_last_scraped",
    "license",
    "latitude",
    "longitude",
    "calendar_updated",
    "bathrooms",
    "minimum_minimum_nights",
    "maximum_minimum_nights",
    "minimum_maximum_nights",
    "maximum_maximum_nights",
    "minimum_nights_avg_ntm",
    "maximum_nights_avg_ntm",
    "availability_60",
    "availability_90",
    "availability_365",
    "calculated_host_listings_count",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    "property_type",
    "first_review",
    "last_review",
]

BOOLEAN_COLUMNS = [
    "host_identity_verified",
    "host_is_superhost",
    "host_has_profile_pic",
    "has_availability",
    "instant_bookable",
]

FINAL_MODEL_COLUMNS = [
    "city",
    "price",
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
    "occupancy_rate",
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

AMENITY_KEYWORDS = {
    "essentials": ["essential"],
    "kitchen": ["kitchen"],
    "microwave": ["microwave"],
    "patio": ["patio", "balcony"],
    "refrigerator": ["refrigerator", "fridge"],
    "video_games": ["video game", "game console"],
    "wifi": ["wifi", "wi-fi", "internet"],
    "cleaning": ["cleaning"],
    "fan": ["fan"],
    "fenced": ["fenced", "fully fenced"],
    "furniture": ["furniture", "outdoor furniture"],
    "glasses": ["wine glasses", "glasses"],
    "laundromat": ["laundromat"],
    "blinds": ["blinds", "room-darkening"],
    "tables": ["table"],
    "wine": ["wine"],
    "grill": ["grill", "bbq"],
    "firepit": ["fire pit", "firepit"],
    "utensils": ["utensil", "silverware", "dishes"],
    "fireplace": ["fireplace"],
    "pool": ["pool"],
    "chairs": ["chair"],
    "loungers": ["lounger", "sun lounger"],
    "staff": ["staff"],
}


def load_city_data(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    frames = []
    for city, filename in CITY_FILES.items():
        file_path = raw_data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing required input file: {file_path}")
        frame = pd.read_csv(file_path, low_memory=False)
        frame["city"] = city
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def drop_unneeded_columns(data: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [col for col in COLUMNS_TO_DROP if col in data.columns]
    return data.drop(columns=columns_to_drop)


def standardize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "bathrooms_text": "bathrooms",
        "host_location": "city",
        "host_neighbourhood": "neighbourhood",
    }
    return data.rename(columns={k: v for k, v in rename_map.items() if k in data.columns})


def clean_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    if "price" in data.columns:
        data["price"] = (
            data["price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        data["price"] = pd.to_numeric(data["price"], errors="coerce")

    for column in ["host_acceptance_rate", "host_response_rate"]:
        if column in data.columns:
            data[column] = (
                data[column]
                .astype(str)
                .str.replace("%", "", regex=False)
                .replace({"nan": np.nan, "None": np.nan})
            )
            data[column] = pd.to_numeric(data[column], errors="coerce") / 100

    if "bathrooms" in data.columns:
        data["bathrooms"] = (
            data["bathrooms"]
            .astype(str)
            .str.extract(r"(\d+(?:\.\d+)?)")[0]
        )
        data["bathrooms"] = pd.to_numeric(data["bathrooms"], errors="coerce")

    return data


def convert_boolean_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    true_values = {"t", "true", "1", "yes", "y"}
    for column in BOOLEAN_COLUMNS:
        if column in data.columns:
            data[column] = (
                data[column]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(true_values)
                .astype(int)
            )
    return data


def parse_amenities(value: object) -> list[str]:
    if pd.isna(value):
        return []

    text = str(value)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip().lower() for item in parsed]
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"^[\[]|[\]$]", "", text)
    items = [item.strip().strip('"').strip("'").lower() for item in cleaned.split(",")]
    return [item for item in items if item]


def add_amenity_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    amenities_list = data.get("amenities", pd.Series([""] * len(data))).apply(parse_amenities)

    data["amenities_count"] = amenities_list.apply(len)
    joined_amenities = amenities_list.apply(lambda items: " | ".join(items))

    for feature, keywords in AMENITY_KEYWORDS.items():
        pattern = "|".join(re.escape(keyword.lower()) for keyword in keywords)
        data[feature] = joined_amenities.str.contains(pattern, regex=True, na=False).astype(int)

    return data


def filter_short_term_listings(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["minimum_nights"] = pd.to_numeric(data["minimum_nights"], errors="coerce")
    return data[data["minimum_nights"] <= 10]


def remove_price_outliers(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data = data[(data["price"] >= 50) & (data["price"] <= 750)]

    mean_price = data["price"].mean()
    std_price = data["price"].std()
    upper_limit = mean_price + 3 * std_price
    lower_limit = mean_price - 3 * std_price

    return data[(data["price"] >= lower_limit) & (data["price"] <= upper_limit)]


def engineer_target_variables(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["availability_30"] = pd.to_numeric(data["availability_30"], errors="coerce")
    data["occupancy_rate"] = ((30 - data["availability_30"]) / 30 * 100).round(2)
    data["log_price"] = np.log(data["price"])
    return data


def select_final_columns(data: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in FINAL_MODEL_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required final columns: {missing_columns}")
    return data[FINAL_MODEL_COLUMNS].copy()


def clean_airbnb_data(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    data = load_city_data(raw_data_dir)
    data = drop_unneeded_columns(data)
    data = standardize_column_names(data)
    data = clean_numeric_columns(data)
    data = convert_boolean_columns(data)
    data = filter_short_term_listings(data)
    data = add_amenity_features(data)
    data = remove_price_outliers(data)
    data = engineer_target_variables(data)
    data = data.dropna()
    return select_final_columns(data)


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    final_data = clean_airbnb_data()
    output_file = PROCESSED_DATA_DIR / "airbnb_cleaned.csv"
    final_data.to_csv(output_file, index=False)

    print(f"Final dataset shape: {final_data.shape}")
    print(f"Saved cleaned dataset to: {output_file}")


if __name__ == "__main__":
    main()
