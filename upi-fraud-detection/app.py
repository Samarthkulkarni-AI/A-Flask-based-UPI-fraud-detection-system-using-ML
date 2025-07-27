from flask import Flask, request, render_template
from predict import predict_fraud
import pandas as pd
import logging
from time import time

logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

last_request_time = 0
request_interval = 0.3

@app.route('/')
def home():
    logging.info("Home page accessed")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global last_request_time
    current_time = time()
    if current_time - last_request_time < request_interval:
        logging.warning("Request rate limit exceeded")
        return render_template('result.html', result="Error", confidence=0, error="Too many requests, please wait")
    last_request_time = current_time

    try:
        amount = request.form['amount']
        if not amount.replace('.', '', 1).isdigit():
            raise ValueError("Amount must be a valid number")
        amount = float(amount)
        
        vpa_sender = request.form['vpa_sender']
        vpa_receiver = request.form['vpa_receiver']
        timestamp = request.form['timestamp']
        
        try:
            pd.to_datetime(timestamp)
        except ValueError:
            raise ValueError("Timestamp must be in YYYY-MM-DD HH:MM:SS format")
        
        device_id = request.form.get('device_id', '')
        
        transaction_data = {
            'amount': amount,
            'vpa_sender': vpa_sender,
            'vpa_receiver': vpa_receiver,
            'timestamp': timestamp,
            'device_id': device_id
        }
        logging.info(f"Received transaction data: {transaction_data}")
        result, confidence = predict_fraud(transaction_data)
        return render_template('result.html', result=result, confidence=confidence)
    except Exception as e:
        logging.error(f"Prediction endpoint error: {str(e)}")
        return render_template('result.html', result="Error", confidence=0, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)