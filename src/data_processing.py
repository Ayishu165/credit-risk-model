import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
#Step 2: Custom Transformer for Date/Time Features
# Extract new datetime features from TransactionStartTime
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'], errors='coerce')
        X['transaction_hour'] = X['TransactionStartTime'].dt.hour
        X['transaction_day'] = X['TransactionStartTime'].dt.day
        X['transaction_month'] = X['TransactionStartTime'].dt.month
        X['transaction_year'] = X['TransactionStartTime'].dt.year
        return X.drop(columns=['TransactionStartTime'])
#step 2 create aggregated features
# Aggregate transaction data per CustomerId
def create_aggregates(df):
    aggregates = df.groupby('CustomerId').agg({
        'Amount': ['sum', 'mean', 'count', 'std'],
        'Value': ['sum', 'mean', 'std']
    })
    aggregates.columns = ['_'.join(col) for col in aggregates.columns]
    aggregates.reset_index(inplace=True)
    return aggregates
#3. define feature columns
# Select which columns to transform
numerical_features = ['Amount', 'Value']
categorical_features = ['CurrencyCode', 'CountryCode', 'ProductCategory', 'ChannelId', 'PricingStrategy']
#4  Step 5: Define Transformers for Each Column Type 
# Define transformers for preprocessing
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),       # Handle missing values
    ('scaler', StandardScaler())                         # Standardize numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Handle missing values
    ('onehot', OneHotEncoder(handle_unknown='ignore'))    # Convert to numerical
])
# Step 6: Combine Transformers Using ColumnTransformer
# Combine numerical and categorical pipelines
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
#Step 7: Create Full Pipeline with Custom Date Transformer
# Final pipeline including datetime feature extraction
full_pipeline = Pipeline(steps=[
    ('date_features', DateFeatureExtractor()),  # Custom step for datetime parsing
    ('preprocessor', preprocessor)              # Standard transformations
])
# Step 8: Use Pipeline on Data
def preprocess_data(df):
    df_clean = df.copy()

    # Drop unnecessary columns
    drop_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'ProviderId', 'ProductId', 'Unnamed: 16', 'Unnamed: 17']
    df_clean.drop(columns=drop_cols, inplace=True, errors='ignore')
    
    # Apply pipeline
    processed_array = full_pipeline.fit_transform(df_clean)

    # Return as DataFrame with new column names
    feature_names = full_pipeline.named_steps['preprocessor'].get_feature_names_out()
    processed_df = pd.DataFrame(processed_array, columns=feature_names)

    return processed_df
#
# 






