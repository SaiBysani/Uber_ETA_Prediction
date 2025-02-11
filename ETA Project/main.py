# main.py
import sys
import os
import pickle
from pathlib import Path

# Add the src directory to the Python path so we can import our modules
current_dir = Path(__file__).parent
src_dir = os.path.join(current_dir, "src")
sys.path.append(str(src_dir))

# Import our project modules
from config import TRAIN_DATA_PATH, COLUMN_MAPPING
from data.data_loader import load_data
from data.data_cleaner import (
    rename_columns,
    update_datatype,
    trimmed_columns,
    convert_nan,
    null_values_handling,
    extract_label_value,
    convert_datatypes,
)
from features.feature_engineering import (
    is_weekend,
    is_peak_hour,
    bin_ratings,
    vectorized_distance_calculation,
    encode_temporal_features,
    drop_temporal_columns,
    label_encode_ordinal_features,
    one_hot_encode_nominal_features,
)
from models.model import train_model


def save_preprocessing_objects(label_encoders, column_mapping):
    """
    Save preprocessing objects that will be needed for making predictions.
    These objects ensure new data is processed the same way as training data.
    """
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save label encoders
    with open("models/label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

    # Save column mapping
    with open("models/column_mapping.pkl", "wb") as f:
        pickle.dump(column_mapping, f)

    print("Saved preprocessing objects successfully!")


def main():
    # Step 1: Load the training data
    print("Loading data...")
    df = load_data(TRAIN_DATA_PATH)

    # Step 2: Clean and preprocess the data
    print("Cleaning data...")
    df = rename_columns(df, COLUMN_MAPPING)
    update_datatype(df)
    df = trimmed_columns(df)
    convert_nan(df)

    # Step 3: Handle missing values
    print("Handling missing values...")
    df = null_values_handling(df)
    df = extract_label_value(df)
    # Step 4: Convert datatypes
    print("Converting datatypes...")
    df = convert_datatypes(df)

    # Step 5: Feature engineering
    print("Engineering features...")
    # Add basic features
    df["order_date_is_weekend"] = df["Order_Date"].apply(is_weekend)
    df["Peak_Hour_Category"] = df["Time_Ordered"].apply(is_peak_hour)
    df["Ratings_Category"] = df["Delivery_person_Ratings"].apply(bin_ratings)
    df["distance"] = vectorized_distance_calculation(df)

    # Add advanced temporal features
    df = encode_temporal_features(df)
    df = drop_temporal_columns(df)

    # # Encode categorical features
    # df, label_encoders = encode_categorical_features(df)
    # Process temporal features

    # Encode categorical features - keeping the encoders for Streamlit
    df, label_encoders = label_encode_ordinal_features(df)
    df = one_hot_encode_nominal_features(df)

    # Save preprocessing objects
    save_preprocessing_objects(label_encoders, COLUMN_MAPPING)

    # Step 6: Remove unnecessary columns
    print("Preparing final dataset...")
    if "ID" in df.columns:
        df = df.drop("ID", axis=1)
    if "Delivery_person_ID" in df.columns:
        df = df.drop("Delivery_person_ID", axis=1)

    # Step 7: Separate features and target
    X = df.drop("Time_taken(min)", axis=1)
    y = df["Time_taken(min)"]

    # Step 8: Train the model
    print("Training model...")
    train_model(X, y)

    print("Pipeline completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
