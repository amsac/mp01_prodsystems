from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self,):
        self.day_names = None

    def fit(self,df,y=None,dteday='dteday',weekday='weekday'):
        return self

    def transform(self, X):

        """Fill missing 'weekday' values using day names from 'dteday'."""
        X = X.copy()  # Avoid modifying original data

        # Ensure 'dteday' column exists
        if 'dteday' not in X:
            raise ValueError("Column 'dteday' not found in input data.")

        # Convert 'dteday' to datetime format
        X['dteday'] = pd.to_datetime(X['dteday'])

        # Find missing values in 'weekday' column
        missing_indices = X[X['weekday'].isna()].index

        # Extract day name and convert to first 3 letters (e.g., 'Monday' → 'Mon')
        
        X.loc[missing_indices, 'weekday'] = X.loc[missing_indices, 'dteday'].dt.day_name().str[:3]
        X = X.drop(columns=['dteday'])
        print("data info after weekday imputer")
        print(X.info())
        return X    


class WeathersitImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weathersit' column by replacing them with the most frequent category value """

    def __init__(self,):
        self.most_frequent_category = None

    def fit(self,df,y=None,weathersit='weathersit'):
        # YOUR CODE HERE
        self.most_frequent_category = df[weathersit].mode()[0]
        return self

    def transform(self,df,y=None,weathersit='weathersit'):
        # YOUR CODE HERE
        df[weathersit] = df[weathersit].fillna(self.most_frequent_category)
        print("data info after WeathersitImputer imputer")
        print(df.info())
        return df



class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Change the outlier values:
        - to upper-bound, if the value is higher than upper-bound, or
        - to lower-bound, if the value is lower than lower-bound respectively.
    """

    def __init__(self,method='iqr', factor=1.5):
        # YOUR CODE HERE
        self.method = method
        self.factor = factor

    def fit(self,df,y=None):
        # YOUR CODE HERE
        self.columns= df.select_dtypes(include=['int64', 'float64']).columns
        return self

    def transform(self,df,y=None):
        # YOUR CODE HERE
        df = df.copy()

        for col in self.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.factor * IQR
            upper_bound = Q3 + self.factor * IQR

            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

        return df
    
class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    """ One-hot encode weekday column """

    def __init__(self,variable:str):
        # YOUR CODE HERE
        if not isinstance(variable, str):
            raise ValueError("variable name should be a string")
        self.variable = variable
        self.encoder = OneHotEncoder(sparse_output=False)


    def fit(self,df,y=None):
         # we need the fit statement to accomodate the sklearn pipeline
        X = df.copy()
        self.encoder.fit(X[[self.variable]])
        # Get encoded feature names
        self.encoded_features_names = self.encoder.get_feature_names_out([self.variable])
        
        return self

    def transform(self, df):

        X = df.copy()
        encoded_weekdays = self.encoder.transform(X[[self.variable]])
        # Append encoded weekday features to X
        X[self.encoded_features_names] = encoded_weekdays
        # drop 'weekday' column after encoding
        X.drop(self.variable, axis=1, inplace=True)        
        return X


    
class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: str, mappings: dict):

        if not isinstance(variables, str):
            raise ValueError("variables should be a str")

        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.variables] = X[self.variables].map(self.mappings).astype(int)
        return X