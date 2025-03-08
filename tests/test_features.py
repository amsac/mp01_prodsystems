import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
import numpy as np

from bikerental_model.processing.data_manager import load_raw_dataset
from bikerental_model.config.core import config
from bikerental_model.processing.features import WeekdayImputer, WeathersitImputer, WeekdayOneHotEncoder,Mapper, OutlierHandler

# testing load_raw_dataset function

def test_read_input():
    df=load_raw_dataset(file_name=config.app_config_.training_data_file)
    expected_columns=['dteday', 'season', 'hr', 'holiday', 'weekday', 'workingday',
       'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual',
       'registered', 'cnt']
    assert list(df.columns) == expected_columns , " Columns dont match"
    assert len(df) == 17379


def test_weekday_imputer(sample_input_data):
    # Given
    imputer = WeekdayImputer()
    assert np.isnan(sample_input_data[0].loc[5,'weekday'])

    # When
    subject = imputer.fit(sample_input_data[0]).transform(sample_input_data[0])

    # Then
    assert subject.loc[5,'weekday'] == 'Sun'

def test_weathersit_imputer():
    df=load_raw_dataset(file_name=config.app_config_.training_data_file)
    first_null_index = df['weathersit'].isna().idxmax()
    
    # Given
    imputer = WeathersitImputer()
    assert np.isnan(df.loc[first_null_index,'weathersit'])

    # When
    subject = imputer.fit(df).transform(df)

    # Then
    assert subject.loc[first_null_index,'weathersit' ] == imputer.most_frequent_category

def test_weekday_onehot_encoder():
    df=load_raw_dataset(file_name=config.app_config_.training_data_file)
    raw_col_length = len(df.columns)

    # Given
    encoder = WeekdayOneHotEncoder()
    assert len(df) == 17379

    # When
    subject = encoder.fit(df).transform(df)
    print(subject.columns)

    # Then
    assert len(subject.columns) == raw_col_length + 7 - 1
    assert 'Weekday_Mon' in subject.columns
    assert 'Weekday_Tue' in subject.columns
    assert 'Weekday_Wed' in subject.columns
    assert 'Weekday_Thu' in subject.columns
    assert 'Weekday_Fri' in subject.columns
    assert 'Weekday_Sat' in subject.columns
    assert 'Weekday_Sun' in subject.columns         

def test_holiday_mapper():

    df=load_raw_dataset(file_name=config.app_config_.training_data_file)
    
    # Given
    mapper = Mapper(config.model_config_.holiday_var, config.model_config_.holiday_mappings)

    # When
    subject = mapper.fit(df).transform(df)

    # Then
    assert df[config.model_config_.holiday_var].isin(config.model_config_.holiday_mappings.keys()).all()

def test_season_mapper():

    df=load_raw_dataset(file_name=config.app_config_.training_data_file)
    
    # Given
    mapper = Mapper(config.model_config_.season_var, config.model_config_.season_mapping)

    # When
    subject = mapper.fit(df).transform(df)

    # Then
    assert df[config.model_config_.season_var].isin(config.model_config_.season_mapping.keys()).all()

def test_outlier_handler_with_bike_data():
    """Test OutlierHandler with actual bike sharing dataset"""
    # Given
    df = load_raw_dataset(file_name=config.app_config_.training_data_file)
    handler = OutlierHandler(factor=1.5)
    
    # When
    handler.fit(df)
    result = handler.transform(df)
    
    # Then
    # Check if numeric columns were correctly identified
    expected_numeric_cols = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
    assert all(col in handler.columns for col in expected_numeric_cols)
    
    # Test specific numeric columns
    for col in ['temp', 'atemp', 'hum', 'windspeed']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Check if outliers were properly handled
        assert result[col].max() <= upper_bound
        assert result[col].min() >= lower_bound

# def test_outlier_handler_specific_columns():
#     """Test OutlierHandler on specific columns known to have outliers"""
#     # Given
#     df = load_raw_dataset(file_name=config.app_config_.training_data_file)
#     handler = OutlierHandler(factor=1.5)
    
#     # When
#     original_stats = {
#         'windspeed': {
#             'max': df['windspeed'].max(),
#             'min': df['windspeed'].min()
#         },
#         'hum': {
#             'max': df['hum'].max(),
#             'min': df['hum'].min()
#         }
#     }
    
#     result = handler.fit(df).transform(df)
    
#     # Then
#     # Test windspeed outliers were handled
#     assert result['windspeed'].max() < original_stats['windspeed']['max']
#     assert result['windspeed'].min() > original_stats['windspeed']['min']
    
#     # Test humidity outliers were handled
#     assert result['hum'].max() < original_stats['hum']['max']
#     assert result['hum'].min() > original_stats['hum']['min']
    
#     # Check data integrity
#     assert len(result) == len(df)
#     assert all(result['windspeed'].between(0, 1))  # windspeed should be normalized between 0 and 1
#     assert all(result['hum'].between(0, 1))  # humidity should be normalized between 0 and 1

def test_outlier_handler_data_distribution():
    """Test the impact of OutlierHandler on data distribution"""
    # Given
    df = load_raw_dataset(file_name=config.app_config_.training_data_file)
    handler = OutlierHandler(factor=1.5)
    
    # When
    result = handler.fit(df).transform(df)
    
    # Then
    for col in ['temp', 'atemp', 'hum', 'windspeed']:
        # Calculate percentage of modified values
        modified_values = (df[col] != result[col]).sum()
        total_values = len(df[col])
        modification_rate = modified_values / total_values
        
        # Ensure we're not modifying too much of the data (typically shouldn't modify more than 1-2% for IQR method)
        assert modification_rate < 0.02, f"Too many values modified in {col}: {modification_rate:.2%}"
        
        # Check if the mean and median are relatively preserved
        assert abs(df[col].mean() - result[col].mean()) / df[col].mean() < 0.1
        assert abs(df[col].median() - result[col].median()) / df[col].median() < 0.1

# def test_outlier_handler_preserves_relationships():
#     """Test that OutlierHandler preserves relationships between variables"""
#     # Given
#     df = load_raw_dataset(file_name=config.app_config_.training_data_file)
#     handler = OutlierHandler(factor=1.5)
    
#     # When
#     result = handler.fit(df).transform(df)
    
#     # Then
#     # Test correlation between temp and atemp is preserved
#     original_corr = df['temp'].corr(df['atemp'])
#     transformed_corr = result['temp'].corr(result['atemp'])
#     assert abs(original_corr - transformed_corr) < 0.1
    
#     # Test that casual + registered still equals cnt
#     assert np.allclose(
#         result['casual'] + result['registered'],
#         result['cnt'],
#         rtol=1e-05,
#         atol=1e-08
#     )
