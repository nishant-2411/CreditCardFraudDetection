# Credit Card Fraud Detection using Logistic Regression

## Introduction
Credit card fraud is a significant concern in financial transactions. This project utilizes a machine learning model based on Logistic Regression to detect fraudulent transactions effectively.

## Dataset
The dataset used in this project is the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). It contains transactions made by European cardholders in September 2013, with highly imbalanced classes (fraud cases are rare).

## Features
- The dataset consists of numerical features (V1-V28) obtained via PCA transformation.
- Two additional columns: `Time` (seconds elapsed between transactions) and `Amount` (transaction amount).
- The target variable `Class` (0 for normal transactions, 1 for fraudulent transactions).

## Model Used
We use **Logistic Regression**, a statistical method for binary classification, to detect fraud cases. It predicts the probability of a transaction being fraudulent based on given features.

## Installation
1. Clone this repository:
   ```sh
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Dependencies
- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Implementation Steps
1. **Data Preprocessing**:
   - Load and explore the dataset.
   - Handle missing values (if any).
   - Scale `Amount` using StandardScaler.
   - Split data into training and testing sets.
2. **Model Training**:
   - Train a Logistic Regression model on the training set.
   - Use class weighting to handle class imbalance.
3. **Model Evaluation**:
   - Assess the model using accuracy, precision, recall, F1-score, and ROC-AUC score.
4. **Prediction**:
   - Use the trained model to predict fraudulent transactions.

## Usage
Run the following command to train and evaluate the model:
```sh
python main.py
```

## Performance Metrics
Since fraud cases are highly imbalanced, accuracy alone is not sufficient. Instead, we evaluate the model using:
- **Precision**: Measures how many predicted fraud cases are actually fraud.
- **Recall (Sensitivity)**: Measures how many actual fraud cases are detected.
- **F1-score**: Balances precision and recall.
- **ROC-AUC Score**: Measures overall model performance.

## Results
- The model achieves a high recall and F1-score, indicating effective fraud detection.
- Performance can be further improved using techniques like SMOTE, feature engineering, or ensemble learning.

## Future Enhancements
- Implement deep learning models (e.g., Neural Networks).
- Try different classification techniques like Random Forest or XGBoost.
- Enhance feature engineering and outlier detection.
