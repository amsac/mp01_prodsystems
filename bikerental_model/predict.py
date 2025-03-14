# load the pipeline, get a dataframe, validate, run the prediction.
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))
import pandas as pd
from bikerental_model.config.core import config
from bikerental_model.processing.data_manager import load_pipeline
from typing import Union
from bikerental_model.processing.validation import validate_inputs
from bikerental_model import __version__ as _version


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
# features = config.model_config_.features
# print("Expected Features:", features)
# print("Sample Data Columns:", sample_data.columns)
# sample_data = sample_data[features]  # Keep only relevant columns
# print(sample_data)
# # 
# # Predict with the pipeline
# prediction = bikerental_pipe.predict(sample_data)

# print(f"Predicted bike rentals: {prediction[0]}")


def make_prediction(*,input_data:Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_inputs(input_df=pd.DataFrame(input_data))
    
    #validated_data=validated_data.reindex(columns=['Pclass','Sex','Age','Fare', 'Embarked','FamilySize','Has_cabin','Title'])
    validated_data=validated_data.reindex(columns=config.model_config_.features)
    #print(validated_data)

    # ------ URGENT ACTION NEEDED--------
    # errors:'' is causing issue at fastapi project. need to fix it.
    results = {"predictions": None, "version": _version, "errors": ''}
    
    predictions = bikerental_pipe.predict(validated_data)

    results = {"predictions": predictions,"version": _version, "errors": ''}
    print(results)
    
    # if not errors:

    #     predictions = bikerental_pipe.predict(validated_data)
    #     results = {"predictions": predictions,"version": _version, "errors": errors}
    #     #print(results)

    return results

if __name__ == "__main__":
    make_prediction(input_data=sample_data)
