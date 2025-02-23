from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class WeekdayImputer(BaseEstimator, TransformerMixin):
    """ Impute missing values in 'weekday' column by extracting dayname from 'dteday' column """

    def __init__(self,):
        self.day_names = None

    def fit(self,df,y=None,dteday='dteday',weekday='weekday'):
        return self

    # def transform(self,df,y=None,dteday='dteday',weekday='weekday'):
    #     print("reached transform")
    #     print(df)
    #     print(df[dteday])
    #     day_names = df[dteday].dt.day_name()[:3]
    #     print("*********&&&&&&&&&&&&&")
    #     print(day_names)
    #     df[weekday] = df[weekday].fillna(day_names)
    #     print("df[weekday].isnull().sum()")
    #     print(df[weekday].isnull().sum())
    #     # df = df.drop(columns=[dteday])
    #     return df
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


# class Mapper(BaseEstimator, TransformerMixin):
#     """
#     Map categories in multiple columns to numerical values.
#     """

#     def __init__(self, mappings):
#         self.mappings = mappings

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X, y=None):
#         X = X.copy()  # Create a copy to avoid modifying the original DataFrame
#         for column, mapping in self.mappings.items():
#             if column in X.columns:  # Check if the column exists in the DataFrame
#                 X[column] = X[column].replace(mapping)
#         return X

# class Mapper(BaseEstimator, TransformerMixin):
#     """Categorical variable mapper."""

#     def __init__(self, variables: str, mappings: dict):

#         if not isinstance(variables, str):
#             raise ValueError("variables should be a str")

#         self.variables = variables
#         self.mappings = mappings

#     def fit(self, X: pd.DataFrame, y: pd.Series = None):
#         # we need the fit statement to accomodate the sklearn pipeline
#         return self

#     def transform(self, X: pd.DataFrame,y=None) -> pd.DataFrame:
#         X = X.copy()
#         X[self.variables] = X[self.variables].map(self.mappings).astype(int)

#         return X

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

    def __init__(self,):
        # YOUR CODE HERE
        self.encoder = None

    def fit(self,df,y=None):
        # YOUR CODE HERE
        return self

    def transform(self, df):
        # Check if 'weekday' column exists before encoding
        # if 'weekday' in df.columns:
        #     # YOUR CODE HERE
            df_encoded = pd.get_dummies(df, columns=['weekday'], prefix='Weekday', dtype=int)
            df = df.drop(columns=['weekday'])
            df = pd.concat([df, df_encoded], axis=1)
        # If 'weekday' column is already encoded, return the DataFrame as is

            return df
    
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