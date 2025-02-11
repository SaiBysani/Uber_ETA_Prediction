# src/models/model.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
import os


def train_model(X, y):
    """
    Train a Random Forest Regressor model.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target variable.

    Returns:
        RandomForestRegressor: Trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = [
        "Delivery_person_Age",
        "distance",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    ]
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Model Performance Metrics:")
    print(f"Mean Absolute Error: {mae:.2f} minutes")
    print(f"Root Mean Squared Error: {rmse:.2f} minutes")
    print(f"R² Score: {r2:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame(
        {"feature": X_train.columns, "importance": rf_model.feature_importances_}
    )
    feature_importance = feature_importance.sort_values("importance", ascending=False)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))

    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="r2")
    print(f"\nCross-validation R² scores: {cv_scores}")
    print(f"Average R² from cross-validation: {cv_scores.mean():.4f}")

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save the trained model
    model_path = "models/eta_model.pkl"
    with open(model_path, "wb") as file:
        pickle.dump(rf_model, file)
    print(f"\nModel saved to {model_path}")

    # Save the scaler for preprocessing new data
    scaler_path = "models/scaler.pkl"
    with open(scaler_path, "wb") as file:
        pickle.dump(scaler, file)
    print(f"Scaler saved to {scaler_path}")

    return rf_model
