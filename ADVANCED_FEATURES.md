# Advanced Features Documentation

<!--
    Developer: Molla Samser
    Designer & Tester: Rima Khatun
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in, info@rskworld.com
    Phone: +91 93305 39277
    Company: RSK World
    Description: Documentation for advanced features in the fraud detection project
-->

## Overview

This document describes all the advanced features available in the Fraud Detection Dataset project. These features enhance the basic fraud detection capabilities with state-of-the-art machine learning techniques, comprehensive evaluation metrics, and production-ready tools.

## Table of Contents

1. [Advanced Feature Engineering](#advanced-feature-engineering)
2. [Hyperparameter Tuning](#hyperparameter-tuning)
3. [Advanced Model Evaluation](#advanced-model-evaluation)
4. [SHAP Explainability](#shap-explainability)
5. [Prediction Pipeline](#prediction-pipeline)
6. [Multiple Model Support](#multiple-model-support)

---

## 1. Advanced Feature Engineering

**File**: `advanced_feature_engineering.py`

### Overview
The advanced feature engineering module automatically creates 20+ additional features from the basic dataset to improve model performance.

### Features Created

#### Time-Based Features
- `day_of_week`: Day of week (0-6)
- `month`: Month (1-12)
- `quarter`: Quarter (1-4)
- `is_month_end`: Whether transaction is at month end
- `is_month_start`: Whether transaction is at month start
- `day_of_month`: Day of month (1-31)

#### Time Pattern Features
- `is_rush_hour`: Transaction during rush hours (7-9 AM, 5-7 PM)
- `is_off_hours`: Transaction during off hours (12 AM-5 AM, 10 PM-12 AM)
- `is_business_hours`: Transaction during business hours (9 AM-5 PM)

#### Statistical Features
- `amount_zscore`: Z-score of transaction amount
- `amount_percentile`: Percentile rank of transaction amount
- `amount_deviation_from_avg`: Deviation from average transaction amount
- `amount_ratio_to_avg`: Ratio to average transaction amount
- `is_high_value`: Flag for high-value transactions (95th percentile)

#### Rolling Statistics
- `amount_rolling_mean_7d`: 7-day rolling mean of transaction amounts
- `amount_rolling_std_7d`: 7-day rolling standard deviation
- `amount_rolling_max_7d`: 7-day rolling maximum

#### Transaction Velocity
- `transaction_velocity`: Transactions per hour
- `is_high_velocity`: Flag for high transaction velocity

#### Account Features
- `account_age_months`: Account age in months
- `is_new_account`: Flag for new accounts (<30 days)
- `is_mature_account`: Flag for mature accounts (>365 days)

#### Interaction Features
- `amount_x_transaction_count`: Amount multiplied by transaction count
- `amount_x_is_foreign`: Amount multiplied by foreign transaction flag
- `amount_x_hour`: Amount multiplied by hour of day

#### Risk Scoring
- `risk_score`: Composite risk score based on multiple factors

#### User Behavior Patterns
- `user_avg_amount`: User's average transaction amount
- `user_std_amount`: User's standard deviation of transaction amounts
- `user_total_transactions`: Total transactions by user
- `amount_deviation_from_user_avg`: Deviation from user's average
- `user_unique_locations`: Number of unique locations for user
- `location_changed`: Flag if user has used multiple locations
- `user_unique_devices`: Number of unique devices for user
- `device_changed`: Flag if user has used multiple devices
- `user_unique_merchants`: Number of unique merchants for user

### Usage

```python
from advanced_feature_engineering import engineer_advanced_features
import pandas as pd

# Load dataset
df = pd.read_csv('fraud_detection_dataset.csv')

# Engineer features
df_advanced = engineer_advanced_features(df)

# Save enhanced dataset
df_advanced.to_csv('fraud_detection_dataset_advanced.csv', index=False)
```

---

## 2. Hyperparameter Tuning

**File**: `hyperparameter_tuning.py`

### Overview
Automated hyperparameter optimization using RandomizedSearchCV with cross-validation for Random Forest, XGBoost, and LightGBM models.

### Features
- Randomized search for faster optimization
- Stratified K-Fold cross-validation
- ROC-AUC scoring for imbalanced datasets
- Automatic best model selection
- Results export to CSV
- Model persistence

### Usage

```bash
python hyperparameter_tuning.py
```

### Parameters Tuned

#### Random Forest
- `n_estimators`: [100, 200, 300]
- `max_depth`: [10, 20, 30, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- `max_features`: ['sqrt', 'log2', None]
- `class_weight`: ['balanced', 'balanced_subsample']

#### XGBoost
- `n_estimators`: [100, 200, 300]
- `max_depth`: [3, 5, 7, 10]
- `learning_rate`: [0.01, 0.1, 0.2]
- `subsample`: [0.6, 0.8, 1.0]
- `colsample_bytree`: [0.6, 0.8, 1.0]
- `min_child_weight`: [1, 3, 5]
- `gamma`: [0, 0.1, 0.2]

#### LightGBM
- `n_estimators`: [100, 200, 300]
- `max_depth`: [3, 5, 7, 10, -1]
- `learning_rate`: [0.01, 0.1, 0.2]
- `subsample`: [0.6, 0.8, 1.0]
- `colsample_bytree`: [0.6, 0.8, 1.0]
- `min_child_samples`: [20, 30, 50]
- `num_leaves`: [31, 50, 100]

---

## 3. Advanced Model Evaluation

**File**: `model_evaluation_advanced.py`

### Overview
Comprehensive evaluation framework with multiple metrics, visualizations, and comparison tools.

### Evaluation Metrics

#### Basic Metrics
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- Specificity

#### Advanced Metrics
- ROC-AUC Score
- Average Precision (AP)
- Log Loss
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa

### Visualizations

1. **Confusion Matrix**: Heatmap visualization
2. **ROC Curves**: Comparison across multiple models
3. **Precision-Recall Curves**: For imbalanced datasets
4. **Feature Importance**: Comparison across models
5. **Metrics Comparison**: Bar plots for all metrics

### Cross-Validation
- Stratified K-Fold cross-validation
- Multiple scoring metrics
- Mean and standard deviation reporting

### Usage

```python
from model_evaluation_advanced import (
    evaluate_model_comprehensive,
    plot_roc_curve,
    plot_precision_recall_curve,
    cross_validate_model
)

# Comprehensive evaluation
metrics, y_pred, y_pred_proba, cm = evaluate_model_comprehensive(
    model, X_test, y_test, model_name="Random Forest"
)

# Plot ROC curve
plot_roc_curve(y_test, [y_pred_proba], ["Random Forest"], 
               save_path="roc_curve.png")

# Cross-validation
cv_scores = cross_validate_model(model, X, y, cv=5)
```

---

## 4. SHAP Explainability

**File**: `shap_explainability.py`

### Overview
Model explainability using SHAP (SHapley Additive exPlanations) values to understand model predictions.

### Features
- SHAP summary plots
- SHAP summary bar plots
- SHAP waterfall plots for individual predictions
- Feature importance from SHAP values
- Support for tree-based models (XGBoost, LightGBM, Random Forest)
- Kernel SHAP for other models

### Usage

```python
from shap_explainability import (
    explain_model_shap,
    plot_shap_summary,
    plot_shap_waterfall,
    get_feature_importance_shap
)

# Generate SHAP values
explainer, shap_values, X_sample = explain_model_shap(
    model, X_test, feature_names=X_test.columns, sample_size=100
)

# Plot summary
plot_shap_summary(shap_values, X_sample, 
                 feature_names=X_test.columns,
                 save_path="shap_summary.png")

# Feature importance
importance_df = get_feature_importance_shap(shap_values, X_test.columns)
```

---

## 5. Prediction Pipeline

**File**: `predict_pipeline.py`

### Overview
Production-ready prediction pipeline for deploying fraud detection models.

### Features
- Single transaction prediction
- Batch prediction from CSV files
- Automatic feature engineering
- Risk level classification (Low/Medium/High)
- Probability scores
- Model persistence and loading

### Usage

```python
from predict_pipeline import FraudDetectionPipeline

# Initialize pipeline
pipeline = FraudDetectionPipeline(model_path='fraud_detection_model.pkl')

# Single prediction
transaction = {
    'amount': 150.50,
    'merchant_category': 'Online',
    'location': 'US',
    'device_type': 'Mobile',
    'user_age': 35,
    'account_age_days': 365,
    'transaction_count_24h': 2,
    'avg_transaction_amount': 75.25,
    'is_foreign_transaction': 0,
    'is_weekend': 0,
    'hour_of_day': 14
}

result = pipeline.predict_single(transaction)
print(result)
# Output: {'is_fraud': 0, 'fraud_probability': 0.123, 'risk_level': 'Low'}

# Batch prediction
results = pipeline.batch_predict(
    'test_transactions.csv', 
    'predictions.csv'
)
```

---

## 6. Multiple Model Support

**File**: `train_model.py`

### Supported Models

#### Random Forest
- Robust ensemble method
- Handles non-linear relationships
- Feature importance analysis
- Balanced class weights

#### XGBoost
- Gradient boosting framework
- Advanced regularization
- Handles missing values
- Scale-aware for imbalanced data

#### LightGBM
- Fast gradient boosting
- Leaf-wise tree growth
- Low memory usage
- GPU support

### Model Comparison
- Automatic comparison of all trained models
- Best model selection based on ROC-AUC
- Performance metrics for each model
- Visual comparison plots

---

## Quick Start Guide

### 1. Generate Dataset
```bash
python generate_data.py
```

### 2. Engineer Advanced Features
```bash
python advanced_feature_engineering.py
```

### 3. Train Models
```bash
python train_model.py
```

### 4. Hyperparameter Tuning (Optional)
```bash
python hyperparameter_tuning.py
```

### 5. Use Prediction Pipeline
```python
from predict_pipeline import FraudDetectionPipeline
pipeline = FraudDetectionPipeline('fraud_detection_model.pkl')
result = pipeline.predict_single(transaction_dict)
```

---

## Best Practices

1. **Feature Engineering**: Always use advanced feature engineering for better model performance
2. **Hyperparameter Tuning**: Perform hyperparameter tuning for production models
3. **Cross-Validation**: Use cross-validation to get robust performance estimates
4. **Model Explainability**: Use SHAP values to understand and explain model predictions
5. **Evaluation Metrics**: Focus on ROC-AUC and Precision-Recall for imbalanced datasets
6. **Risk Levels**: Use probability thresholds to classify risk levels appropriately

---

## Contact

For questions or support:
- **Website**: https://rskworld.in
- **Email**: help@rskworld.in, support@rskworld.in
- **Phone**: +91 93305 39277

---

**Developer**: Molla Samser  
**Designer & Tester**: Rima Khatun  
**Company**: RSK World

