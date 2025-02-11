# src/config.py
import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Define paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(DATA_DIR, "test.csv")

# Column mapping for standardization
COLUMN_MAPPING = {
    "Time_Orderd": "Time_Ordered",
    "Weatherconditions": "Weather_Conditions",
    "multiple_deliveries": "Multiple_Deliveries",
    "City": "City_Type",
}
