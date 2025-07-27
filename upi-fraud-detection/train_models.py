import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import joblib

# Load dataset (replace with your dataset, e.g., synthetic_upi_transactions.csv)
data = pd.read_csv('upi-fraud-detection/synthetic_upi_transactions.csv')

# Handle missing values
data.fillna({'amount': data['amount'].mean(), 'device_id': 'unknown'}, inplace=True)

# Feature engineering
data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek

# Encode categorical variables
data = pd.get_dummies(data, columns=['vpa_sender', 'vpa_receiver', 'device_id'], drop_first=True)

# Split features and target
X = data.drop(['timestamp', 'is_fraud'], axis=1)
y = data['is_fraud']

# Scale numerical features
scaler = StandardScaler()
X[['amount', 'hour']] = scaler.fit_transform(X[['amount', 'hour']])

# Handle class imbalance
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Save preprocessed data
pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['is_fraud'])], axis=1).to_csv('preprocessed_data.csv', index=False)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Initialize models
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(random_state=42)
lr = LogisticRegression(random_state=42)

# Hyperparameter tuning for Random Forest
rf_params = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1')
rf_grid.fit(X_train, y_train)

# Train and evaluate models
models = {'Random Forest': rf_grid.best_estimator_, 'XGBoost': xgb, 'Logistic Regression': lr}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name} Performance:")
    print(classification_report(y_test, y_pred))
    joblib.dump(model, f"{name.lower().replace(' ', '_')}_model.pkl")

# Save feature columns for prediction
joblib.dump(X.columns, 'feature_columns.pkl')