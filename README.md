# Fraud Detection Dataset

<!--
    Developer: Molla Samser
    Designer & Tester: Rima Khatun
    Website: https://rskworld.in
    Email: help@rskworld.in, support@rskworld.in, info@rskworld.com
    Phone: +91 93305 39277
    Company: RSK World
    Description: Financial fraud detection dataset with transaction records, user behavior patterns, and fraud labels for building anti-fraud ML models.
-->

Financial fraud detection dataset with transaction records, user behavior patterns, and fraud labels for building anti-fraud ML models.

## Description

This dataset contains transaction records with features like transaction amount, location, time, merchant information, and fraud labels. Perfect for building fraud detection models, anomaly detection, and financial security applications.

## Features

- **Transaction records** - Comprehensive transaction data
- **User behavior features** - Advanced behavioral patterns
- **Fraud labels** - Binary classification labels
- **Imbalanced dataset handling** - Realistic 5% fraud ratio
- **Ready for classification models** - Preprocessed and structured
- **Advanced feature engineering** - Time-based, statistical, and interaction features
- **Multiple ML models** - Random Forest, XGBoost, LightGBM support

## Technologies

- CSV, Excel
- Pandas, NumPy
- Scikit-learn, XGBoost, LightGBM
- SHAP (for model explainability)
- SMOTE (for imbalanced data handling)

## Dataset Structure

### Basic Features
- `transaction_id`: Unique identifier for each transaction
- `user_id`: User identifier
- `amount`: Transaction amount
- `merchant_category`: Category of the merchant
- `location`: Transaction location
- `timestamp`: Transaction timestamp
- `device_type`: Device used for transaction
- `user_age`: Age of the user
- `account_age_days`: Age of the account in days
- `transaction_count_24h`: Number of transactions in last 24 hours
- `avg_transaction_amount`: Average transaction amount
- `is_foreign_transaction`: Whether transaction is from foreign location
- `is_weekend`: Whether transaction occurred on weekend
- `hour_of_day`: Hour of the day (0-23)
- `is_fraud`: Fraud label (0 = Normal, 1 = Fraud)

### Advanced Features (Auto-generated)
- **Time-based features**: day_of_week, month, is_month_end, is_month_start, quarter
- **Transaction velocity**: time_since_last_transaction, transactions_per_hour, transactions_per_day
- **Statistical features**: amount_zscore, amount_percentile, amount_deviation_from_avg
- **Rolling statistics**: amount_rolling_mean_7d, amount_rolling_std_7d, amount_rolling_max_7d
- **Change indicators**: location_changed, device_changed, merchant_category_changed
- **Risk scoring**: risk_score (composite risk indicator)
- **Interaction features**: amount_x_transaction_count, amount_x_is_foreign, etc.
- **Time patterns**: is_rush_hour, is_off_hours, is_business_hours
- **Behavioral patterns**: avg_amount_ratio, transaction_velocity
- **Account features**: account_age_months, is_new_account, is_mature_account

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate the dataset:
```bash
python generate_data.py
```

### 2. Engineer advanced features (optional):
```bash
python advanced_feature_engineering.py
```

### 3. Explore the data:
```bash
jupyter notebook fraud_detection_analysis.ipynb
```

### 4. Train models:
```bash
# Basic training
python train_model.py

# The script will automatically:
# - Use advanced feature engineering
# - Train multiple models (Random Forest, XGBoost, LightGBM)
# - Compare model performance
# - Generate evaluation plots
```

## Advanced Features

### 1. Advanced Feature Engineering (`advanced_feature_engineering.py`)
Comprehensive feature engineering module that creates:
- **Time-based features**: day_of_week, month, quarter, is_month_end, is_month_start
- **Time patterns**: is_rush_hour, is_off_hours, is_business_hours
- **Statistical features**: amount_zscore, amount_percentile, amount_deviation_from_avg
- **Rolling statistics**: amount_rolling_mean_7d, amount_rolling_std_7d, amount_rolling_max_7d
- **Transaction velocity**: transaction_velocity, is_high_velocity
- **Change indicators**: location_changed, device_changed
- **Risk scoring**: risk_score (composite risk indicator)
- **Interaction features**: amount_x_transaction_count, amount_x_is_foreign, amount_x_hour
- **Account features**: account_age_months, is_new_account, is_mature_account
- **User behavior patterns**: user_avg_amount, user_std_amount, amount_deviation_from_user_avg

### 2. Hyperparameter Tuning (`hyperparameter_tuning.py`)
- Automated hyperparameter optimization using RandomizedSearchCV
- Supports Random Forest, XGBoost, and LightGBM
- Cross-validation for robust parameter selection
- Automatic best model selection and saving
- Results exported to CSV for analysis

### 3. Advanced Model Evaluation (`model_evaluation_advanced.py`)
Comprehensive evaluation with:
- Multiple metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision
- Additional metrics: Log Loss, Matthews Correlation Coefficient, Cohen's Kappa
- Cross-validation with multiple scoring metrics
- ROC curve comparison across models
- Precision-Recall curve analysis
- Feature importance comparison
- Metrics visualization and comparison plots

### 4. Model Explainability (`shap_explainability.py`)
SHAP (SHapley Additive exPlanations) integration:
- SHAP summary plots
- SHAP waterfall plots for individual predictions
- Feature importance from SHAP values
- Model interpretation and explainability

### 5. Prediction Pipeline (`predict_pipeline.py`)
Production-ready prediction system:
- Batch prediction from CSV files
- Single transaction prediction
- Risk level classification (Low/Medium/High)
- Probability scores
- Ready for deployment

### 6. Model Training (`train_model.py`)
Training script with:
- **Random Forest**: Robust ensemble method with balanced class weights
- **XGBoost**: Gradient boosting with advanced regularization
- **LightGBM**: Fast gradient boosting framework
- Automatic model comparison and best model selection
- SMOTE for handling imbalanced data
- Comprehensive evaluation metrics

## Difficulty Level

Intermediate to Advanced

## Project Structure

```
fraud-detection/
├── fraud_detection_dataset.csv           # Generated dataset
├── generate_data.py                      # Dataset generation script
├── advanced_feature_engineering.py       # Advanced feature engineering
├── train_model.py                        # Model training script
├── hyperparameter_tuning.py              # Hyperparameter optimization
├── model_evaluation_advanced.py          # Advanced evaluation metrics
├── shap_explainability.py                # SHAP model explainability
├── predict_pipeline.py                   # Production prediction pipeline
├── fraud_detection_analysis.ipynb        # Jupyter notebook for analysis
├── verify_dataset.py                     # Dataset verification script
├── requirements.txt                      # Python dependencies
├── README.md                             # This file
├── index.html                            # Demo page
└── .gitignore                            # Git ignore file
```

## Model Outputs

After training and evaluation, the following files may be generated:
- `fraud_detection_model.pkl` - Best trained model
- `best_tuned_model_*.pkl` - Best tuned model from hyperparameter optimization
- `confusion_matrix.png` - Confusion matrix visualization
- `roc_curve.png` - ROC curve comparison
- `feature_importance.png` - Feature importance plot
- `hyperparameter_tuning_results.csv` - Hyperparameter tuning results
- `fraud_detection_dataset_advanced.csv` - Dataset with advanced features
- `shap_summary.png` - SHAP summary plot (if SHAP is used)
- `shap_summary_bar.png` - SHAP summary bar plot

## Contact

For questions or support, please contact:
- **Website**: https://rskworld.in
- **Email**: help@rskworld.in, support@rskworld.in
- **Phone**: +91 93305 39277

## License

Content used for educational purposes only.

## Credits

- **Developer**: Molla Samser
- **Designer & Tester**: Rima Khatun
- **Company**: RSK World
