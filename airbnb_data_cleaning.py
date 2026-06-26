
#Airbnb listing data cleaning.


from pathlib import Path

import numpy as np
import pandas as pd


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


CITY_FILES = {
    "Austin": "Austin.csv",
    "New Orleans": "New_Orleans.csv",
    "New York": "New_York.csv",
    "San Francisco": "San_Francisco.csv",
    "Denver": "Denver_CO.csv",
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


def load_city_data(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Load and combine Airbnb listing data from the five city CSV files."""
    frames = []

    for city, filename in CITY_FILES.items():
        file_path = raw_data_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing input file: {file_path}. "
                "Place the raw Inside Airbnb city CSV files in data/raw/."
            )

        df = pd.read_csv(file_path, low_memory=False)
        df["source_city"] = city
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def clean_percentage(series: pd.Series) -> pd.Series:
    """Convert percentage strings such as '95%' to decimal values such as 0.95."""
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        .astype(float)
        / 100
    )


def clean_price(series: pd.Series) -> pd.Series:
    """Convert price strings such as '$150.00' to numeric values."""
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        .astype(float)
    )


def clean_bathrooms(series: pd.Series) -> pd.Series:
    """Extract the numeric bathroom count from strings such as '1 bath'."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"(\d+\.?\d*)", expand=False),
        errors="coerce",
    )


def convert_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Inside Airbnb t/f Boolean columns to 1/0 indicators."""
    for column in BOOLEAN_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map({"t": 1, "f": 0, True: 1, False: 0}).fillna(0).astype(int)

    return df


def clean_airbnb_data(raw_data_dir: Path = RAW_DATA_DIR) -> pd.DataFrame:
    """Run the Airbnb listing data-cleaning workflow."""
    data = load_city_data(raw_data_dir)

    data = data.drop(columns=[col for col in COLUMNS_TO_DROP if col in data.columns])

    data = data.dropna()

    if "minimum_nights" in data.columns:
        data = data[data["minimum_nights"] < 10]

    data = data.rename(
        columns={
            "bathrooms_text": "bathrooms",
            "host_location": "city",
        }
    )

    if "host_acceptance_rate" in data.columns:
        data["host_acceptance_rate"] = clean_percentage(data["host_acceptance_rate"])

    if "host_response_rate" in data.columns:
        data["host_response_rate"] = clean_percentage(data["host_response_rate"])

    if "bathrooms" in data.columns:
        data["bathrooms"] = clean_bathrooms(data["bathrooms"])

    data = convert_boolean_columns(data)

    if "price" in data.columns:
        data["price"] = clean_price(data["price"])

    if "amenities" in data.columns:
        data = data.drop(columns=["amenities"])

    data = data.dropna()

    if "price" in data.columns:
        data = data[(data["price"] <= 750) & (data["price"] >= 50)]
        price_mean = data["price"].mean()
        price_std = data["price"].std()
        data = data[data["price"] <= price_mean + 3 * price_std]
        data["log_price"] = np.log(data["price"])

    if "availability_30" in data.columns:
        data["occupancy_rate"] = ((30 - data["availability_30"]) / 30 * 100).round(2)

    return data


def main() -> None:
    airbnb_df = clean_airbnb_data()

    excel_path = PROCESSED_DATA_DIR / "Airbnb_df.xlsx"
    csv_path = PROCESSED_DATA_DIR / "Airbnb_df.csv"

    airbnb_df.to_excel(excel_path, index=False)
    airbnb_df.to_csv(csv_path, index=False)

    print(f"Saved cleaned Excel file to: {excel_path}")
    print(f"Saved cleaned CSV file to: {csv_path}")
    print(f"Final dataset shape: {airbnb_df.shape[0]} rows x {airbnb_df.shape[1]} columns")


if __name__ == "__main__":
    main()
