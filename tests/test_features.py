import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
import numpy as np

from bikerental_model.processing.data_manager import load_raw_dataset
from bikerental_model.config.core import config
from bikerental_model.processing.features import WeekdayImputer, WeathersitImputer, WeekdayOneHotEncoder,Mapper

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