import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Step 1: Extract date-based features
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_col):
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
        return X

# Step 2: Aggregate features at customer level
class CustomerAggregator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        agg_df = X.groupby('CustomerId').agg({
            'Amount': ['sum', 'mean', 'std', 'count'],
            'Value': ['sum', 'mean'],
            'TransactionHour': 'mean',
            'TransactionDay': 'mean',
            'TransactionMonth': 'mean',
            'TransactionYear': 'mean',
            'ProductCategory': 'first',
            'ChannelId': 'first'
        })
        agg_df.columns = ['_'.join(col) for col in agg_df.columns]
        agg_df = agg_df.fillna(0)
        return agg_df.reset_index()

# Step 3: Build transformation pipeline for model

def build_preprocessing_pipeline():
    numerical_features = [
        'Amount_sum', 'Amount_mean', 'Amount_std', 'Amount_count',
        'Value_sum', 'Value_mean',
        'TransactionHour_mean', 'TransactionDay_mean',
        'TransactionMonth_mean', 'TransactionYear_mean'
    ]

    categorical_features = ['ProductCategory_first', 'ChannelId_first']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    return preprocessor

# Step 4: Complete pipeline function

def prepare_model_data(df):
    # Clean garbage columns if they exist
    df = df.drop(columns=["Unnamed: 16", "Unnamed: 17"], errors="ignore")

    # Extract time-based features
    df = DateFeatureExtractor(date_col="TransactionStartTime").fit_transform(df)

    # Aggregate by customer
    df_agg = CustomerAggregator().fit_transform(df)

    return df_agg