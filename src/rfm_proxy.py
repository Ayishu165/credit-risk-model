# src/rfm_proxy.py

import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# ------------------- Step 1: Load Data -------------------
try:
    # __file__ is not available in some environments (like notebooks), so we use a fallback
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../data/raw/data.csv")
    
    # Read the CSV file
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded from: {data_path}")

except FileNotFoundError as e:
    print("❌ File not found. Please make sure 'data.csv' exists in data/raw/")
    raise e

# ------------------- Step 2: Clean Data -------------------
# Remove any unnamed columns (often index leftovers)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Parse transaction date column
df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'], errors='coerce')

# Drop rows missing required fields
df = df.dropna(subset=['CustomerId', 'TransactionStartTime'])

# ------------------- Step 3: RFM Calculation -------------------
snapshot_date = df['TransactionStartTime'].max() + timedelta(days=1)

# Group by CustomerId
rfm = df.groupby('CustomerId').agg({
    'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
    'TransactionId': 'count',                                          # Frequency
    'Amount': 'sum'                                                    # Monetary
}).reset_index()

# Rename columns
rfm.columns = ['CustomerId', 'Recency', 'Frequency', 'Monetary']

# Fill any missing values in Monetary
rfm['Monetary'] = rfm['Monetary'].fillna(0)

# ------------------- Step 4: Scale RFM + Cluster -------------------
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

# Identify high-risk cluster (high Recency, low Frequency & Monetary)
cluster_stats = rfm.groupby('cluster')[['Recency', 'Frequency', 'Monetary']].mean()
high_risk_cluster = cluster_stats.sort_values(by=['Frequency', 'Monetary', 'Recency'], ascending=[True, True, False]).index[0]

# Assign binary label
rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

# ------------------- Step 5: Merge Label to Main Dataset -------------------
df_labeled = df.merge(rfm[['CustomerId', 'is_high_risk']], on='CustomerId', how='left')

# ------------------- Step 6: Save Outputs -------------------
# Define output paths
processed_dir = os.path.join(base_dir, "../data/processed")
os.makedirs(processed_dir, exist_ok=True)

rfm_output_path = os.path.join(processed_dir, "rfm_metrics.csv")
labeled_output_path = os.path.join(processed_dir, "train_data.csv")

rfm.to_csv(rfm_output_path, index=False)
df_labeled.to_csv(labeled_output_path, index=False)

print("✅ RFM metrics and labeled dataset saved successfully.")
print(f"📄 RFM metrics: {rfm_output_path}")
print(f"📄 Labeled data: {labeled_output_path}")
