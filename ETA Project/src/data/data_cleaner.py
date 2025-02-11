# src/data/data_cleaner.py

import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


def rename_columns(df, column_mapping):
    """
    Rename columns in a pandas DataFrame using a dictionary mapping.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_mapping (dict): Dictionary with old column names as keys and new names as values.

    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    return df.rename(columns=column_mapping)


def update_datatype(df):
    """
    Update data types of columns in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    df["Delivery_person_Age"] = df["Delivery_person_Age"].astype("float64")
    df["Delivery_person_Ratings"] = df["Delivery_person_Ratings"].astype("float64")
    df["Multiple_Deliveries"] = df["Multiple_Deliveries"].astype("float64")
    df["Order_Date"] = pd.to_datetime(df["Order_Date"], format="%d-%m-%Y")

    return df


def trimmed_columns(df):
    """
    Trim whitespace from specified categorical columns in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with trimmed columns.
    """
    categorical_columns = [
        "Weather_Conditions",
        "Type_of_order",
        "Type_of_vehicle",
        "Festival",
        "City_Type",
    ]
    for column in categorical_columns:
        df[column] = df[column].str.strip()
    return df


def convert_nan(df):
    """
    Convert string 'NaN' to np.nan.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """
    df.replace("NaN", float(np.nan), regex=True, inplace=True)

    return df


def null_values_handling(df):
    """
    Perform KNN imputation on numeric columns

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with null values

    Returns:
    --------
    pandas.DataFrame
        DataFrame with null values imputed using KNN
    """
    # Create a copy of the DataFrame
    df = df.copy()

    # Select numeric columns for imputation
    numeric_cols = [
        "Delivery_person_Age",
        "Delivery_person_Ratings",
        "Multiple_Deliveries",
    ]

    # Separate numeric columns
    numeric_df = df[numeric_cols].copy()

    # Standardize the numeric columns before imputation
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_df)

    # Initialize KNN Imputer
    imputer = KNNImputer(n_neighbors=5)

    # Perform imputation on scaled data
    imputed_scaled = imputer.fit_transform(numeric_scaled)

    # Inverse transform to get back to original scale
    imputed_data = scaler.inverse_transform(imputed_scaled)

    # Replace original numeric columns with imputed values
    df[numeric_cols] = imputed_data

    # Handle categorical columns separately
    categorical_cols = [
        "Weather_Conditions",
        "Road_traffic_density",
        "Type_of_vehicle",
        "Festival",
        "City_Type",
    ]

    # Fill categorical columns with mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Handle time-related columns
    time_columns = ["Time_Ordered", "Time_Order_picked"]
    for col in time_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df


def extract_label_value(df):
    """
    Extracts the label value (Time_taken(min)) from the 'Time_taken(min)' column in the given DataFrame df.
    """

    df["Time_taken(min)"] = df["Time_taken(min)"].apply(
        lambda x: int(x.replace("(min)", "").strip())
    )

    return df


def convert_datatypes(df):
    """
    Convert datatypes for columns based on their names and content

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame to convert datatypes

    Returns:
    --------
    pandas.DataFrame
        DataFrame with converted datatypes
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Conversion for specific columns
    # Categorical columns (convert to category for memory efficiency)
    categorical_columns = [
        "Weather_Conditions",
        "Road_traffic_density",
        "Type_of_order",
        "Type_of_vehicle",
        "Festival",
        "City_Type",
    ]

    # Datetime-related columns
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])

    # Numeric columns with potential issues
    numeric_columns = {
        "Delivery_person_Age": "int64",
        "Delivery_person_Ratings": "float64",
        "Vehicle_condition": "int64",
        "Multiple_Deliveries": "int64",  # Assuming you want to convert float to int
    }

    # Convert categorical columns
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    # Convert time columns to datetime.time
    time_columns = ["Time_Ordered", "Time_Order_picked"]
    for col in time_columns:
        df[col] = pd.to_datetime(
            df[col], format="%H:%M:%S", errors="coerce"
        ).dt.strftime("%H:%M:%S")

    # Convert numeric columns
    for col, dtype in numeric_columns.items():
        # # Handle potential non-numeric values
        # if col == "Time_taken(min)":
        #     # Remove any non-numeric characters and convert to int
        #     df[col] = df[col].apply(lambda x: int(x.split(" ")[1].strip()))

        # Convert to specified dtype
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].astype(dtype)

    # Latitude and Longitude columns (ensure float)
    lat_long_columns = [
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    for col in lat_long_columns:
        df[col] = df[col].astype("float64")

    return df
