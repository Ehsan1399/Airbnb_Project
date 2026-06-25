#Create amenity indicator variables from Inside Airbnb listing data.

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


DATA_DIR = Path("data") / "raw"
OUTPUT_DIR = Path("outputs")

CITY_FILES = {
    "Austin": DATA_DIR / "Austin.csv",
    "New Orleans": DATA_DIR / "New_Orleans.csv",
    "New York": DATA_DIR / "New_York.csv",
    "San Francisco": DATA_DIR / "San_Francisco.csv",
    "Denver": DATA_DIR / "Denver.csv",
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


def load_city_data(city_files: dict[str, Path]) -> pd.DataFrame:
    """Load raw listing files and add a city label to each file."""
    dataframes = []

    for city, file_path in city_files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"Missing input file: {file_path}")

        df = pd.read_csv(file_path, low_memory=False)
        df["city"] = city
        dataframes.append(df)

    return pd.concat(dataframes, ignore_index=True)


def clean_numeric_column(series: pd.Series) -> pd.Series:
    """Remove non-numeric characters and convert a column to numeric values."""
    cleaned = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def clean_listing_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning steps required before amenity text mining."""
    data = data.copy()

    data = data.drop(columns=[col for col in COLUMNS_TO_DROP if col in data.columns])

    if "minimum_nights" in data.columns:
        data["minimum_nights"] = pd.to_numeric(data["minimum_nights"], errors="coerce")
        data = data[data["minimum_nights"] < 10]

    rename_map = {
        "bathrooms_text": "bathrooms",
        "host_location": "city",
        "host_neighbourhood": "neighbourhood",
    }
    data = data.rename(columns={old: new for old, new in rename_map.items() if old in data.columns})

    for col in ["host_acceptance_rate", "host_response_rate"]:
        if col in data.columns:
            data[col] = clean_numeric_column(data[col]) / 100

    if "bathrooms" in data.columns:
        data["bathrooms"] = clean_numeric_column(data["bathrooms"])

    if "price" in data.columns:
        data["price"] = clean_numeric_column(data["price"])

    for col in BOOLEAN_COLUMNS:
        if col in data.columns:
            data[col] = data[col].map({"t": 1, "f": 0, True: 1, False: 0})

    if "amenities" not in data.columns:
        raise ValueError("The input data must contain an 'amenities' column.")

    required_columns = ["id", "amenities"]
    data = data.dropna(subset=[col for col in required_columns if col in data.columns])

    return data.reset_index(drop=True)


def normalize_amenity_text(value: object) -> str:
    """Convert the raw amenities JSON-like text into clean text."""
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = re.sub(r"[\[\]\{\}\"']", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def create_amenity_indicators(
    data: pd.DataFrame,
    min_document_frequency: float = 0.03,
) -> pd.DataFrame:
    """Create binary amenity variables from the amenities text field.

    The default min_document_frequency approximates the R removeSparseTerms
    threshold of 0.97 by retaining terms present in at least 3% of listings.
    """
    amenity_text = data["amenities"].apply(normalize_amenity_text)

    vectorizer = CountVectorizer(
        binary=True,
        stop_words="english",
        min_df=min_document_frequency,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",
    )

    amenity_matrix = vectorizer.fit_transform(amenity_text)
    feature_names = vectorizer.get_feature_names_out()

    amenities_df = pd.DataFrame(
        amenity_matrix.toarray(),
        columns=feature_names,
        index=data.index,
    )

    amenities_df.insert(0, "id", data["id"].values)
    amenities_df["amenities_count"] = amenities_df.drop(columns=["id"]).sum(axis=1)

    return amenities_df


def main() -> None:
    """Run the amenity text-mining workflow."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_data = load_city_data(CITY_FILES)
    cleaned_data = clean_listing_data(raw_data)
    amenities_df = create_amenity_indicators(cleaned_data)

    cleaned_data.to_csv(OUTPUT_DIR / "cleaned_listing_data_for_amenities.csv", index=False)
    amenities_df.to_csv(OUTPUT_DIR / "amenities_df.csv", index=False)

    print(f"Cleaned listing data shape: {cleaned_data.shape}")
    print(f"Amenities data shape: {amenities_df.shape}")
    print(f"Outputs saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
