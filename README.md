UPI Fraud Detection System

A Flask-based web application for detecting potentially fraudulent UPI transactions using a machine learning model trained on synthetic data. This project demonstrates the use of machine learning, backend development, and frontend integration to simulate a real-world fraud detection tool.

---

Features

- Predicts whether a UPI transaction is genuine or fraudulent
- Shows model confidence (probability) in the result
- Built using a custom **synthetic dataset**
- Trained model integrated using `pickle`
- User-friendly interface for entering transaction details
- Basic **transaction history** display
- Optional input for device ID

---

## ML Model Overview

- **Dataset**: Custom synthetic data generated for training
- **Training**: Done using Google Colab
- **Model**: Trained and saved as `fraud_model.pkl`
- **Features Used**:
  - Sender UPI ID
  - Receiver UPI ID
  - Amount
  - Time & Date
  - Optional Device ID

---

## Tech Stack

- **Backend**: Flask (Python)
- **ML Libraries**: scikit-learn, pandas, numpy
- **Frontend**: HTML, CSS, JavaScript (with Jinja2 templating)
- **Model Integration**: `pickle`

---

## Folder Structure
upi-fraud-detection/
├── app.py # Flask app
├── fraud_model.pkl # Trained ML model
├── dataset_preprocessed.csv # Preprocessed synthetic dataset
├── templates/
│ ├── index.html # Input form
│ ├── result.html # Result display
│ └── history.html # Transaction history
├── static/
│ ├── style.css # Styling
│ ├── script.js # JavaScript logic
├── requirements.txt # Python dependencies
├── README.md # Project description

## How to Run Locally

1. Clone the repository
2. Install dependencies
   - pip install -r requirements.txt
3. Run the Flask app
   - python app.py

## Example Input

1. Sender UPI:-	sender@upi
2. Receiver UPI:-	receiver@upi
3. Amount:-	10000
4. Date:-	2025-07-27
5. Time:-	14:23
6. Device ID (opt.):-	ABC123XYZ

## Output Example
Result:
- Legitimate or Fraudulent
- Model Confidence: 92.7%

## Notes
This project is for educational/demo purposes.
The dataset is synthetic — no real user data is used.
The model’s accuracy and logic are limited by the simulated dataset.

## Author
- Samarth Kulkarni
- Email: samarthvishwanathkulkarni@gmail.com
- GitHub: @Samarthkulkarni-AI
