# Fraud Detection in Financial Transactions

## Project Overview

This project aims to detect fraudulent activities in financial transactions using machine learning techniques, specifically focusing on anomaly detection methods like Autoencoders. The goal is to identify fraudulent transactions that deviate from normal patterns by analyzing the reconstruction errors of the transactions. The project uses synthetic or Kaggle datasets for demonstration.

## Skills Involved
- Anomaly Detection
- Neural Networks (Autoencoders, CNNs)
- Data Engineering
- Model Evaluation (ROC-AUC, Precision-Recall)
- SQL and Python

## Datasets

- **Source**: [Kaggle - Synthetic Financial Transactions](https://www.kaggle.com/)
- **Synthetic Data**: The project uses synthetic data generated for training and testing the model. Ensure that the dataset contains relevant features like transaction amounts, timestamps, and other transaction details.




## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fraud_detection_project.git
   cd fraud_detection_project

   pip install -r requirements.txt
   python scripts/preprocess_data.py
   python scripts/train_model.py
   python scripts/evaluate_model.py
import pandas as pd
from scripts.evaluate_model import test_new_data

# Load new data
new_data = pd.read_csv('data/new_test_data.csv')
result = test_new_data(new_data)
print(result)


