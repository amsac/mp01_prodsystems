# load the pipeline, get a dataframe, validate, run the prediction.
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
import pandas as pd
from bikerental_model.config.core import config
from bikerental_model.processing.data_manager import load_pipeline

# Load the trained pipeline
bikerental_pipe = load_pipeline(file_name="bike_rental_model_output_v0.0.1.pkl")

# Create a sample row with realistic values
sample_data = pd.DataFrame({
    "season": ["winter"],
    "dteday":["2012-11-05"],
    "hr": ["6am"],
    "holiday": ["No"],
    "weekday": ["Sun"],
    "workingday": ["Yes"],
    "weathersit": ["Mist"],
    "temp": [6.10],
    "atemp": [3.0014],
    "hum": [49.0],
    "windspeed": [19.0012],
    "casual": [4],
    "registered": [135],
    "year": [2012],
    "month": [11]
})

# Ensure columns match training data
features = config.model_config_.features
print("Expected Features:", features)
print("Sample Data Columns:", sample_data.columns)
sample_data = sample_data[features]  # Keep only relevant columns
print(sample_data)
# 
# Predict with the pipeline
prediction = bikerental_pipe.predict(sample_data)

print(f"Predicted bike rentals: {prediction[0]}")
