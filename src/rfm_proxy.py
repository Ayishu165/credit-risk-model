import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Calculate RFM metrics
def compute_rfm(df):
    """
    Compute Recency, Frequency, and Monetary metrics for each customer
    Recency = Days since last transaction (lower = more recent)
    Frequency = Number of transactions
    Monetary = Total transaction amount
    """
    snapshot_date = df["TransactionStartTime"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerId").agg({
        "TransactionStartTime": lambda x: (snapshot_date - x.max()).days,
        "TransactionId": "count",
        "Amount": "sum"
    })
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    return rfm.reset_index()

# Step 2: Perform KMeans clustering on RFM features
def cluster_rfm(rfm_df, n_clusters=3):
    """
    Cluster customers into groups based on RFM using KMeans
    Returns: DataFrame with added 'cluster' and 'is_high_risk'
    """
    features = ["Recency", "Frequency", "Monetary"]
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_df[features])

    # Apply KMeans with fixed random_state for reproducibility
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    rfm_df["cluster"] = kmeans.fit_predict(rfm_scaled)

    # Step 3: Define the high-risk cluster (typically low Frequency and Monetary, high Recency)
    risk_cluster = rfm_df.groupby("cluster")[["Recency", "Frequency", "Monetary"]].mean()
    high_risk_cluster = risk_cluster.sort_values(by=["Recency", "Frequency", "Monetary"], ascending=[False, True, True]).index[0]

    # Assign binary risk label
    rfm_df["is_high_risk"] = (rfm_df["cluster"] == high_risk_cluster).astype(int)

    return rfm_df[["CustomerId", "is_high_risk"]]

# Step 4: Merge high-risk label with processed feature dataset
def merge_with_processed(processed_df, risk_labels_df):
    """
    Merge risk label DataFrame back into the processed dataset by CustomerId
    """
    return processed_df.merge(risk_labels_df, on="CustomerId", how="left").fillna({"is_high_risk": 0})
