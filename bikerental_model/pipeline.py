import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load Data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['year'] = df['dteday'].dt.year
    df['month'] = df['dteday'].dt.month
    return df

# Preprocessing Pipeline
def get_pipeline():
    numeric_features = ['temp', 'atemp', 'hum', 'windspeed']
    categorical_features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    return pipeline

# Train Model
def train_model(df):
    X = df[['temp', 'atemp', 'hum', 'windspeed', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']]
    y = df['cnt']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline = get_pipeline()
    pipeline.fit(X_train, y_train)
    
    joblib.dump(pipeline, 'model.pkl')
    print("Model trained and saved.")
    
    y_pred = pipeline.predict(X_test)
    print(f'MSE: {mean_squared_error(y_test, y_pred)}, R2 Score: {r2_score(y_test, y_pred)}')

if __name__ == "__main__":
    df = load_data('bike-sharing-dataset.csv')
    train_model(df)
