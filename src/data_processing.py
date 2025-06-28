import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
# ---------------------------------------------
# 3. CREATE AGGREGATE FEATURES per CustomerId
# ---------------------------------------------
class AggregateTransactionFeatures(BaseEstimator, TransformerMixin):
    """
    Aggregates total, mean, count, and std of Amount for each CustomerId.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg_df = X.groupby('CustomerId').agg(
            TotalAmount=('Amount', 'sum'),
            AvgAmount=('Amount', 'mean'),
            TransactionCount=('Amount', 'count'),
            StdAmount=('Amount', 'std')
        ).reset_index()
        return agg_df
# ---------------------------------------------
# 4. EXTRACT FEATURES from Transaction Time
# ---------------------------------------------
class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts hour, day, month, and year from the 'TransactionStartTime' column.
    """
    def __init__(self, date_col='TransactionStartTime'):
        self.date_col = date_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.date_col] = pd.to_datetime(X[self.date_col], errors='coerce')
        X['TransactionHour'] = X[self.date_col].dt.hour
        X['TransactionDay'] = X[self.date_col].dt.day
        X['TransactionMonth'] = X[self.date_col].dt.month
        X['TransactionYear'] = X[self.date_col].dt.year
        return X.drop(columns=[self.date_col])
# ---------------------------------------------
# FINAL FEATURE PIPELINE CONSTRUCTION
# ---------------------------------------------
def create_feature_pipeline():
    # Choose columns for transformations
    numeric_features = ['TotalAmount', 'AvgAmount', 'TransactionCount', 'StdAmount']
    categorical_features = ['ProductCategory', 'ChannelId']

    # ---------------------------------------------
    # 6. HANDLE MISSING VALUES + 7. STANDARDIZATION
    # ---------------------------------------------
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),       # Fill missing with mean
        ('scaler', StandardScaler())                       # Standardize features
    ])

    # ---------------------------------------------
    # 5. ENCODE CATEGORICAL VARIABLES (One-Hot)
    # ---------------------------------------------
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing with mode
        ('onehot', OneHotEncoder(handle_unknown='ignore'))     # Convert to binary format
    ])

    # Combine numeric and categorical pipelines
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # ---------------------------------------------
    # Combine All Steps into Full Pipeline
    # ---------------------------------------------
    full_pipeline = Pipeline(steps=[
        ('datetime_features', DateTimeFeatureExtractor()),       # 4. Extract features from date
        ('aggregate_features', AggregateTransactionFeatures()),  # 3. Aggregate features per CustomerId
        ('preprocessing', preprocessor)                          # 5, 6, 7: Encode, Impute, Scale
    ])

    return full_pipeline
