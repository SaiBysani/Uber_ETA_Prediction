# streamlit_app.py
import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add the src directory to the Python path
current_dir = Path(__file__).parent
src_dir = os.path.join(current_dir, "src")
sys.path.append(str(src_dir))

# Import project modules
from config import COLUMN_MAPPING
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


def load_preprocessing_objects():
    """Load all necessary preprocessing objects"""
    try:
        with open("models/eta_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, label_encoders, scaler
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None


def prepare_features(input_df, label_encoders, scaler):
    """Prepare features using the same pipeline as in main.py"""
    try:
        # Verify input DataFrame
        if input_df is None or input_df.empty:
            st.error("Input DataFrame is empty or None")
            return None

        # Make a copy of the input DataFrame
        df = input_df.copy()

        # Debug checkpoint
        st.write("Initial DataFrame columns:", df.columns.tolist())

        # First apply column mapping
        if COLUMN_MAPPING:
            df = rename_columns(df, COLUMN_MAPPING)
            st.write("Columns after mapping:", df.columns.tolist())

        # Verify required columns are present
        required_columns = [
            "Order_Date",
            "Time_Ordered",
            "Delivery_person_Ratings",
            "Restaurant_latitude",
            "Restaurant_longitude",
            "Delivery_location_latitude",
            "Delivery_location_longitude",
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None

        # Basic feature engineering first
        try:
            df["order_date_is_weekend"] = df["Order_Date"].apply(is_weekend)
            df["Peak_Hour_Category"] = df["Time_Ordered"].apply(is_peak_hour)
            df["Ratings_Category"] = df["Delivery_person_Ratings"].apply(bin_ratings)
            df["distance"] = vectorized_distance_calculation(df)
            st.write("Basic feature engineering completed successfully")
        except Exception as e:
            st.error(f"Error in basic feature engineering: {str(e)}")
            return None

        # Apply each transformation with error checking
        transformations = [
            (update_datatype, "update_datatype"),
            (trimmed_columns, "trimmed_columns"),
            (convert_nan, "convert_nan"),
            (null_values_handling, "null_values_handling"),
            (extract_label_value, "extract_label_value"),
            (convert_datatypes, "convert_datatypes"),
            (encode_temporal_features, "encode_temporal_features"),
            (drop_temporal_columns, "drop_temporal_columns"),
        ]

        for transform_func, name in transformations:
            try:
                df = transform_func(df)
                if df is None:
                    st.error(f"Error in {name}: returned None")
                    return None
                st.write(f"Completed {name} successfully")
            except Exception as e:
                st.error(f"Error in {name}: {str(e)}")
                return None

        # Encode categorical features
        try:
            df, _ = label_encode_ordinal_features(df)
            df = one_hot_encode_nominal_features(df)
            st.write("Categorical encoding completed successfully")
        except Exception as e:
            st.error(f"Error in categorical encoding: {str(e)}")
            return None

        # Drop unnecessary columns
        try:
            columns_to_drop = ["ID", "Delivery_person_ID"]
            existing_columns = [col for col in columns_to_drop if col in df.columns]
            if existing_columns:
                df = df.drop(existing_columns, axis=1)
            st.write("Column dropping completed successfully")
        except Exception as e:
            st.error(f"Error dropping columns: {str(e)}")
            return None

        return df

    except Exception as e:
        st.error(f"Error in feature preparation step: {str(e)}")
        if "df" in locals():
            st.write("Current DataFrame columns:", df.columns.tolist())
        return None


def main():
    st.set_page_config(
        page_title="Food Delivery Time Prediction",
        page_icon="🚚",
        layout="wide",
    )

    st.title("🚚 Food Delivery Time Prediction")
    st.write("""
    This app predicts food delivery times using various factors like order details,
    location, delivery person information, and weather conditions.
    """)

    # Load model and preprocessing objects
    model, label_encoders, scaler = load_preprocessing_objects()
    if not all([model, label_encoders, scaler]):
        st.error(
            "Failed to load necessary model files. Please ensure all required files are present."
        )
        return

    # Create three columns for input
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📦 Order Details")
        order_date = st.date_input("Order Date", datetime.now())
        order_time = st.time_input("Order Time", datetime.now().time())
        order_type = st.selectbox(
            "Type of Order", ["Snack", "Meal", "Drinks", "Buffet"]
        )
        multiple_deliveries = st.selectbox(
            "Multiple Deliveries",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    with col2:
        st.subheader("👤 Delivery Person Details")
        delivery_person_id = st.text_input("Delivery Person ID", "DEL1234")
        delivery_person_age = st.number_input("Age", 18, 60, 25)
        delivery_person_ratings = st.slider("Ratings", 1.0, 5.0, 4.0, 0.1)
        vehicle_condition = st.selectbox(
            "Vehicle Condition",
            [0, 1, 2, 3],
            format_func=lambda x: ["Poor", "Fair", "Good", "Excellent"][x],
        )
        vehicle_type = st.selectbox(
            "Vehicle Type", ["motorcycle", "scooter", "bicycle"]
        )

    with col3:
        st.subheader("🌍 Environmental Conditions")
        weather_conditions = st.selectbox(
            "Weather", ["Fog", "Stormy", "Sandstorms", "Windy", "Cloudy", "Sunny"]
        )
        road_traffic = st.selectbox("Traffic Density", ["Low", "Medium", "High", "Jam"])
        city = st.selectbox("City Type", ["Urban", "Metropolitan", "Semi-Urban"])
        festival = st.selectbox("Festival", ["No", "Yes"])

    # Location details in two columns
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("🏪 Restaurant Location")
        restaurant_latitude = st.number_input("Restaurant Latitude", -90.0, 90.0, 20.0)
        restaurant_longitude = st.number_input(
            "Restaurant Longitude", -180.0, 180.0, 80.0
        )

    with col5:
        st.subheader("📍 Delivery Location")
        delivery_latitude = st.number_input("Delivery Latitude", -90.0, 90.0, 20.1)
        delivery_longitude = st.number_input("Delivery Longitude", -180.0, 180.0, 80.1)

    if st.button("Predict Delivery Time", type="primary"):
        try:
            with st.spinner("Calculating delivery time..."):
                # Create input DataFrame
                input_data = {
                    "ID": "DELIVERY_001",
                    "Delivery_person_ID": delivery_person_id,
                    "Delivery_person_Age": delivery_person_age,
                    "Delivery_person_Ratings": delivery_person_ratings,
                    "Restaurant_latitude": restaurant_latitude,
                    "Restaurant_longitude": restaurant_longitude,
                    "Delivery_location_latitude": delivery_latitude,
                    "Delivery_location_longitude": delivery_longitude,
                    "Order_Date": order_date.strftime("%d-%m-%Y"),
                    "Time_Orderd": order_time.strftime(
                        "%H:%M:%S"
                    ),  # Note the spelling to match mapping
                    "Time_Order_picked": (
                        datetime.combine(order_date, order_time) + timedelta(minutes=15)
                    ).strftime("%H:%M:%S"),
                    "Weatherconditions": weather_conditions,  # Note the spelling to match mapping
                    "Road_traffic_density": road_traffic,
                    "Vehicle_condition": vehicle_condition,
                    "Type_of_order": order_type,
                    "Type_of_vehicle": vehicle_type,
                    "multiple_deliveries": multiple_deliveries,  # Note the case to match mapping
                    "Festival": festival,
                    "City": city,  # Note the case to match mapping
                }

                # Create DataFrame and show debug info
                input_df = pd.DataFrame([input_data])
                st.write("Input DataFrame shape:", input_df.shape)
                st.write("Input columns:", input_df.columns.tolist())

                # Prepare features
                features_df = prepare_features(input_df, label_encoders, scaler)

                if features_df is None:
                    st.error(
                        "Feature preparation failed. Please check the errors above."
                    )
                    return

                # Show processed data debug info
                st.write("Processed DataFrame shape:", features_df.shape)
                st.write("Processed columns:", features_df.columns.tolist())

                # Make prediction
                prediction = model.predict(features_df)[0]

                # Calculate delivery timeline
                order_datetime = datetime.combine(order_date, order_time)
                pickup_time = order_datetime + timedelta(minutes=15)
                delivery_time = pickup_time + timedelta(minutes=int(prediction))

                # Display results
                st.success(f"⏱️ Estimated Delivery Time: {prediction:.0f} minutes")

                # Show timeline
                st.subheader("📊 Delivery Timeline")
                col6, col7, col8 = st.columns(3)

                with col6:
                    st.metric("Order Time", order_datetime.strftime("%H:%M"))
                with col7:
                    st.metric("Pickup Time", pickup_time.strftime("%H:%M"))
                with col8:
                    st.metric("Delivery Time", delivery_time.strftime("%H:%M"))

                # Show additional insights
                st.subheader("📈 Delivery Insights")
                distance = vectorized_distance_calculation(
                    pd.DataFrame(
                        [
                            {
                                "Restaurant_latitude": restaurant_latitude,
                                "Restaurant_longitude": restaurant_longitude,
                                "Delivery_location_latitude": delivery_latitude,
                                "Delivery_location_longitude": delivery_longitude,
                            }
                        ]
                    )
                )[0]

                col9, col10, col11 = st.columns(3)
                with col9:
                    st.metric("Distance", f"{distance:.2f} km")
                with col10:
                    st.metric("Traffic Condition", road_traffic)
                with col11:
                    st.metric("Weather", weather_conditions)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Please check your inputs and try again.")


if __name__ == "__main__":
    main()
