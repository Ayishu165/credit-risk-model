import pandas as pd
import numpy as np
import pytest
from src.data_processing import DateFeatureExtractor, preprocess_data, create_aggregates

# ---------------------------
# Sample input data for tests
# ---------------------------
data = {
    'TransactionStartTime': ['2024-01-01T10:00:00Z', '2024-02-15T14:30:00Z'],
    'Amount': [100, 250],
    'Value': [100, 250],
    'CurrencyCode': ['UGX', 'UGX'],
    'CountryCode': [256, 256],
    'ProductCategory': ['airtime', 'utility_bill'],
    'ChannelId': ['ChannelId_1', 'ChannelId_2'],
    'PricingStrategy': [2, 3]
}

df_sample = pd.DataFrame(data)

# -----------------------------
# Test 1: Date Feature Extractor
# -----------------------------
def test_date_feature_extractor():
    """
    Test that DateFeatureExtractor extracts expected features correctly.
    """
    extractor = DateFeatureExtractor()
    transformed = extractor.fit_transform(df_sample.copy())
    assert 'transaction_hour' in transformed.columns
    assert 'transaction_day' in transformed.columns
    assert 'transaction_month' in transformed.columns
    assert 'transaction_year' in transformed.columns
    assert 'TransactionStartTime' not in transformed.columns

# ------------------------------------
# Test 2: Output of Preprocess Pipeline
# ------------------------------------
def test_preprocess_data_shape():
    """
    Test that preprocess_data outputs numeric DataFrame without nulls.
    """
    processed = preprocess_data(df_sample.copy())
    assert isinstance(processed, pd.DataFrame)
    assert all(dtype.kind in 'iufc' for dtype in processed.dtypes)  # numeric types
    assert processed.isnull().sum().sum() == 0  # no missing values

# ----------------------------------------
# Test 3: Helper Function - create_aggregates
# ----------------------------------------
def test_create_aggregates_correct_output():
    """
    Test that create_aggregates computes correct aggregates per CustomerId.
    """
    sample_data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, 200, 300],
        'Value': [10, 20, 30]
    })
    result = create_aggregates(sample_data)
    assert result.shape[0] == 2
    assert 'Amount_sum' in result.columns
    assert result[result['CustomerId'] == 'C1']['Amount_sum'].values[0] == 300

# --------------------------------------------------
# Test 4: create_aggregates handles missing values
# --------------------------------------------------
def test_create_aggregates_handles_missing_values():
    """
    Test that create_aggregates does not break with NaN values.
    """
    sample_data = pd.DataFrame({
        'CustomerId': ['C1', 'C1', 'C2'],
        'Amount': [100, np.nan, 300],
        'Value': [10, 20, np.nan]
    })
    result = create_aggregates(sample_data)
    assert result.shape[0] == 2
    assert not result.isnull().all(axis=1).any()  # ensure rows are not all NaN
