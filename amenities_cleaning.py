#Create amenity indicator variables from raw Inside Airbnb listing data.


from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

CITY_FILES = [
    "Austin.csv",
    "New Orleans.csv",
    "New York.csv",
    "San Francisco.csv",
    "Denver.csv",
]

OUTPUT_EXCEL = PROCESSED_DATA_DIR / "Amenties_df.xlsx"
OUTPUT_CSV = PROCESSED_DATA_DIR / "Amenties_df.csv"
SPARSITY_THRESHOLD = 0.97

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


def load_city_data(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and combine raw listing datasets for all selected cities."""
    dataframes = []

    for file_name in CITY_FILES:
        file_path = raw_data_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing input file: {file_path}. Place all raw city CSV files in {raw_data_dir}."
            )
        dataframes.append(pd.read_csv(file_path, encoding="latin1", low_memory=False))

    return pd.concat(dataframes, ignore_index=True)


def clean_listing_data(data: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning steps used before amenities text mining."""
    data = data.copy()

    existing_drop_columns = [col for col in COLUMNS_TO_DROP if col in data.columns]
    data = data.drop(columns=existing_drop_columns)

    data = data.dropna()

    if "minimum_nights" not in data.columns:
        raise KeyError("The required column 'minimum_nights' was not found.")
    data = data[data["minimum_nights"] < 10].copy()

    data = data.rename(columns={"bathrooms_text": "bathrooms", "host_location": "city"})

    for column in ["host_acceptance_rate", "host_response_rate"]:
        if column in data.columns:
            data[column] = (
                data[column]
                .astype(str)
                .str.replace("%", "", regex=False)
                .replace({"nan": pd.NA, "": pd.NA})
            )
            data[column] = pd.to_numeric(data[column], errors="coerce") / 100

    if "bathrooms" in data.columns:
        data["bathrooms"] = (
            data["bathrooms"]
            .astype(str)
            .str.replace(r" .*", "", regex=True)
            .replace({"nan": pd.NA, "": pd.NA})
        )
        data["bathrooms"] = pd.to_numeric(data["bathrooms"], errors="coerce")

    for column in BOOLEAN_COLUMNS:
        if column in data.columns:
            data[column] = data[column].astype(str).str.lower().eq("t").astype(int)

    if "price" in data.columns:
        data["price"] = (
            data["price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        data["price"] = pd.to_numeric(data["price"], errors="coerce")

    data = data.dropna()

    if "amenities" not in data.columns:
        raise KeyError("The required column 'amenities' was not found.")

    return data


def preprocess_amenities_text(value: object) -> str:
    """Make lowercase, remove punctuation, remove stop words."""
    text = "" if pd.isna(value) else str(value).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = [word for word in text.split() if word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def create_amenities_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Create the filtered amenity document-term matrix and amenities_count."""
    amenities_text = data["amenities"].map(preprocess_amenities_text)

    min_document_frequency = max(
        1,
        math.ceil((1 - SPARSITY_THRESHOLD) * len(amenities_text)),
    )

    vectorizer = CountVectorizer(
        binary=True,
        lowercase=False,
        token_pattern=r"(?u)\b\w\w+\b",
        min_df=min_document_frequency,
    )

    amenity_matrix = vectorizer.fit_transform(amenities_text)
    amenities_df = pd.DataFrame.sparse.from_spmatrix(
        amenity_matrix,
        columns=vectorizer.get_feature_names_out(),
    )
    amenities_df = amenities_df.sparse.to_dense().astype(int)
    amenities_df["amenities_count"] = amenities_df.sum(axis=1)

    return amenities_df


def save_outputs(amenities_df: pd.DataFrame) -> None:
    """Save the amenities dataset in Excel and CSV formats."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    amenities_df.to_excel(OUTPUT_EXCEL, index=False)
    amenities_df.to_csv(OUTPUT_CSV, index=False)


def main() -> None:
    combined_data = load_city_data()
    cleaned_data = clean_listing_data(combined_data)
    amenities_df = create_amenities_dataframe(cleaned_data)
    save_outputs(amenities_df)

    print(f"Combined raw data shape: {combined_data.shape}")
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Amenities data shape: {amenities_df.shape}")
    print(f"Saved Excel file to: {OUTPUT_EXCEL}")
    print(f"Saved CSV file to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
