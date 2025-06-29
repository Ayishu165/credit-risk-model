import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn

# 1. Load the processed labeled data
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "../data/processed/train_data.csv")
df = pd.read_csv(data_path)

# 2. Prepare features and target
drop_cols = ['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 
             'TransactionStartTime', 'ProductId', 'ProviderId']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')
df = df.dropna(subset=['is_high_risk'])  # drop missing target rows

# Fill missing values
df.fillna({
    'Amount': 0,
    'Value': 0,
    'CurrencyCode': df['CurrencyCode'].mode()[0],
    'ProductCategory': df['ProductCategory'].mode()[0],
    'ChannelId': df['ChannelId'].mode()[0],
    'PricingStrategy': df['PricingStrategy'].mode()[0],
    'CountryCode': df['CountryCode'].mode()[0] if 'CountryCode' in df.columns else 0
}, inplace=True)

# One-hot encode categoricals
df = pd.get_dummies(df, drop_first=True)

# 3. Train-test split
X = df.drop(columns=['is_high_risk'])
y = df['is_high_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Define models with hyperparameter grids
models = {
    'LogisticRegression': {
        'model': LogisticRegression(max_iter=1000),
        'params': {'C': [0.1, 1, 10]}
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [100, 200], 'max_depth': [5, 10, None]}
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    }
}

# 6. Start MLflow experiment
mlflow.set_experiment("credit-risk-model")
best_model = None
best_score = 0

for name, mp in models.items():
    print(f"\n Training: {name}")
    with mlflow.start_run(run_name=name):
        clf = GridSearchCV(mp['model'], mp['params'], cv=3, scoring='roc_auc')
        clf.fit(X_train_scaled, y_train)

        # Predict and evaluate
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)

        # Log metrics and parameters
        mlflow.log_params(clf.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc)

        # Track best model
        if roc > best_score:
            best_score = roc
            best_model = clf.best_estimator_

# 7. Save best model to MLflow
print("\n Registering best model...")
with mlflow.start_run(run_name="best_model"):
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="best_model"  # ✅ This is what enables registry access
    )
    print("Best model logged and registered with MLflow.")
