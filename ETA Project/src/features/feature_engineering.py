# src/features/feature_engineering.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def is_weekend(date):
    """
    Determine if a given date is a weekend (Saturday or Sunday).

    Args:
        date: Can be a string, datetime.date, or datetime.datetime object

    Returns:
        int: 1 if weekend, 0 if weekday
    """
    if isinstance(date, str):
        # Convert string to datetime
        date = pd.to_datetime(date)

    # Use weekday() method which returns 0-6 (Monday is 0 and Sunday is 6)
    # Weekend days (Saturday and Sunday) are 5 and 6
    return 1 if date.weekday() >= 5 else 0


def is_peak_hour(time):
    """
    Categorize the time into peak hours based on the hour of the day.

    Args:
        time (str or datetime): The time to be categorized.

    Returns:
        str: Peak hour category.
    """
    if pd.isna(time):
        return "Unknown"
    hour = pd.to_datetime(time).hour
    if 7 <= hour < 10:
        return "Morning Peak"
    elif 12 <= hour < 14:
        return "Lunch Peak"
    elif 17 <= hour < 20:
        return "Evening Peak"
    else:
        return "Off-Peak"


def bin_ratings(rating):
    """
    Categorize delivery person ratings into descriptive bins.

    Args:
        rating (float): The rating to be categorized.

    Returns:
        str: Rating category.
    """
    if pd.isna(rating):
        return "Not Rated"
    elif rating < 3:
        return "Low"
    elif 3 <= rating < 4:
        return "Average"
    elif 4 <= rating < 4.5:
        return "Good"
    else:
        return "Excellent"


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points using Haversine formula.

    Args:
        lat1, lon1, lat2, lon2 (float): Latitude and longitude of two points.

    Returns:
        float: Distance in kilometers.
    """
    R = 6371  # Radius of the Earth in kilometers
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def vectorized_distance_calculation(df):
    """
    Calculate distances for all rows in a DataFrame using vectorized approach.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.Series: Distance in kilometers.
    """
    return df.apply(
        lambda row: calculate_distance(
            row["Restaurant_latitude"],
            row["Restaurant_longitude"],
            row["Delivery_location_latitude"],
            row["Delivery_location_longitude"],
        ),
        axis=1,
    )


def encode_temporal_features(df):
    """
    Perform comprehensive temporal feature engineering with cyclical encoding
    on datetime columns in the DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing 'Time_Ordered' and 'Time_Order_picked' columns

    Returns:
    --------
    pandas.DataFrame
        DataFrame with engineered temporal features using cyclical encoding
    """

    def cyclical_encoding(df, col, max_val):
        """
        Convert a cyclic feature into its sine and cosine components.
        This helps capture the cyclic nature of time-based features.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame
        col : str
            Column name to encode
        max_val : int
            Maximum value of the cycle (e.g., 24 for hours, 7 for days)
        """
        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)
        return df.drop(col, axis=1)

    # Extract hour components from datetime columns
    df["hour_ordered"] = pd.to_datetime(df["Time_Ordered"]).dt.hour
    df["hour_picked"] = pd.to_datetime(df["Time_Order_picked"]).dt.hour

    # Apply cyclical encoding to hour features
    df = cyclical_encoding(df, "hour_ordered", 24)
    df = cyclical_encoding(df, "hour_picked", 24)

    return df


def drop_temporal_columns(df):
    """
    Drop the original temporal columns after feature engineering.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing temporal columns to be dropped

    Returns:
    --------
    pandas.DataFrame
        DataFrame with specified temporal columns removed
    """
    columns_to_drop = [
        "Order_Date",  # Already extracted day of week
        "Time_Ordered",  # Already extracted hour
        "Time_Order_picked",  # Already extracted hour
    ]

    # Drop only the columns that exist in the DataFrame
    existing_columns = [col for col in columns_to_drop if col in df.columns]
    if existing_columns:
        df = df.drop(columns=existing_columns)

    return df


# def encode_categorical_features(df):
#     """
#     Encode categorical features using Label Encoding for ordinal features
#     and One-Hot Encoding for nominal features.

#     Parameters:
#     -----------
#     df : pandas.DataFrame
#         Input DataFrame containing categorical columns to be encoded

#     Returns:
#     --------
#     tuple
#         - pandas.DataFrame: DataFrame with encoded categorical features
#         - dict: Dictionary of fitted LabelEncoders for ordinal columns
#     """

#     # Define ordinal columns (ordered categories)
#     ordinal_columns = [
#         "Road_traffic_density",
#         "Vehicle_condition",
#         "Day_of_Week",  # 0=Monday to 6=Sunday
#         "Ratings_Category",  # Already binned from low to excellent
#         "Delivery_person_Ratings",
#     ]

#     # Define nominal columns (unordered categories)
#     nominal_columns = [
#         "Weather_Conditions",
#         "Type_of_vehicle",
#         "Festival",
#         "City_Type",
#         "Type_of_order",
#         "Peak_Hour_Category",
#     ]

#     # Initialize dictionary to store label encoders
#     label_encoders = {}

#     # Apply Label Encoding to ordinal columns
#     for col in ordinal_columns:
#         if col in df.columns:  # Only process columns that exist in the DataFrame
#             label_encoders[col] = LabelEncoder()
#             df[col] = label_encoders[col].fit_transform(df[col])

#     # Apply One-Hot Encoding to nominal columns
#     # Only process columns that exist in the DataFrame
#     nominal_columns_present = [col for col in nominal_columns if col in df.columns]
#     if nominal_columns_present:
#         df = pd.get_dummies(df, columns=nominal_columns_present, drop_first=True)

#     return df, label_encoders


def label_encode_ordinal_features(df, ordinal_columns=None):
    """
    Encode ordinal categorical features using Label Encoding.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing ordinal categorical columns
    ordinal_columns : list, optional
        List of ordinal columns to encode. If None, uses default ordinal columns

    Returns:
    --------
    tuple
        - pandas.DataFrame: DataFrame with encoded ordinal features
        - dict: Dictionary of fitted LabelEncoders
    """
    if ordinal_columns is None:
        ordinal_columns = [
            "Road_traffic_density",
            "Vehicle_condition",
            "Day_of_Week",  # 0=Monday to 6=Sunday
            "Ratings_Category",  # Already binned from low to excellent
            "Delivery_person_Ratings",
        ]

    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Initialize dictionary to store label encoders
    label_encoders = {}

    # Apply Label Encoding to ordinal columns
    for col in ordinal_columns:
        if col in df.columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])

    return df, label_encoders


def one_hot_encode_nominal_features(df, nominal_columns=None):
    """
    Encode nominal categorical features using One-Hot Encoding.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing nominal categorical columns
    nominal_columns : list, optional
        List of nominal columns to encode. If None, uses default nominal columns

    Returns:
    --------
    pandas.DataFrame
        DataFrame with one-hot encoded nominal features
    """
    if nominal_columns is None:
        nominal_columns = [
            "Weather_Conditions",
            "Type_of_vehicle",
            "Festival",
            "City_Type",
            "Type_of_order",
            "Peak_Hour_Category",
        ]

    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()

    # Filter for columns that exist in the DataFrame
    nominal_columns_present = [col for col in nominal_columns if col in df.columns]

    # Apply One-Hot Encoding if there are nominal columns present
    if nominal_columns_present:
        df = pd.get_dummies(df, columns=nominal_columns_present, drop_first=True)

    return df
