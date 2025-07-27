import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

try:
    rf_model = joblib.load('upi-fraud-detection/random_forest_model.pkl')
    xgb_model = joblib.load('upi-fraud-detection/xgboost_model.pkl')
    lr_model = joblib.load('upi-fraud-detection/logistic_regression_model.pkl')
    feature_columns = joblib.load('upi-fraud-detection/feature_columns.pkl')
    scaler = joblib.load('upi-fraud-detection/scaler.pkl')
except Exception as e:
    logging.error(f"Error loading models or feature columns: {str(e)}")
    raise

def predict_fraud(transaction_data):
    try:
        df = pd.DataFrame([transaction_data])
        logging.info(f"Input data: {transaction_data}")
        
        if 'device_id' not in df.columns or pd.isna(df['device_id'].iloc[0]) or df['device_id'].iloc[0] == '':
            df['device_id'] = 'unknown'
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        df = pd.get_dummies(df, columns=['vpa_sender', 'vpa_receiver', 'device_id'], drop_first=True)
        
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
            df = pd.concat([df, missing_df], axis=1)
        df = df[feature_columns]
        logging.info(f"Processed features: {df.columns.tolist()}")
        
        df[['amount', 'hour']] = scaler.transform(df[['amount', 'hour']])
        logging.info(f"Scaled features: amount={df['amount'].iloc[0]:.4f}, hour={df['hour'].iloc[0]:.4f}")
        
        rf_pred = rf_model.predict(df)[0]
        rf_prob = rf_model.predict_proba(df)[0][1] if rf_pred == 1 else rf_model.predict_proba(df)[0][0]
        xgb_pred = xgb_model.predict(df)[0]
        xgb_prob = xgb_model.predict_proba(df)[0][1] if xgb_pred == 1 else xgb_model.predict_proba(df)[0][0]
        lr_pred = lr_model.predict(df)[0]
        lr_prob = lr_model.predict_proba(df)[0][1] if lr_pred == 1 else lr_model.predict_proba(df)[0][0]
        
        logging.info(f"RF pred: {rf_pred}, prob: {rf_prob:.4f}")
        logging.info(f"XGB pred: {xgb_pred}, prob: {xgb_prob:.4f}")
        logging.info(f"LR pred: {lr_pred}, prob: {lr_prob:.4f}")
        
        if any(np.isnan([rf_prob, xgb_prob, lr_prob])):
            raise ValueError("One or more model probabilities are NaN")
        
        votes = [rf_pred, xgb_pred, lr_pred]
        final_pred = 1 if sum(votes) >= 2 else 0
        confidence = (rf_prob + xgb_prob + lr_prob) / 3
        
        logging.info(f"Final prediction: {'Fraud' if final_pred == 1 else 'Legitimate'}, Confidence: {confidence:.4f}")
        return 'Fraud' if final_pred == 1 else 'Legitimate', confidence
    
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise