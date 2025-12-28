"""
Advanced Feature Engineering for Fraud Detection

Developer: Molla Samser
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in, info@rskworld.com
Phone: +91 93305 39277
Company: RSK World
Description: Advanced feature engineering including time-based, statistical, and interaction features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def engineer_advanced_features(df):
    """
    Create advanced features for fraud detection
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset with basic features
    
    Returns:
    --------
    pd.DataFrame
        Dataset with additional advanced features
    """
    print("Engineering advanced features...")
    df_eng = df.copy()
    
    # Convert timestamp to datetime if needed
    if 'timestamp' in df_eng.columns:
        if not pd.api.types.is_datetime64_any_dtype(df_eng['timestamp']):
            df_eng['timestamp'] = pd.to_datetime(df_eng['timestamp'])
        
        # Time-based features
        df_eng['day_of_week'] = df_eng['timestamp'].dt.dayofweek
        df_eng['month'] = df_eng['timestamp'].dt.month
        df_eng['quarter'] = df_eng['timestamp'].dt.quarter
        df_eng['is_month_end'] = df_eng['timestamp'].dt.is_month_end.astype(int)
        df_eng['is_month_start'] = df_eng['timestamp'].dt.is_month_start.astype(int)
        df_eng['day_of_month'] = df_eng['timestamp'].dt.day
        
        # Time patterns
        df_eng['is_rush_hour'] = ((df_eng['hour_of_day'] >= 7) & (df_eng['hour_of_day'] <= 9) | 
                                  (df_eng['hour_of_day'] >= 17) & (df_eng['hour_of_day'] <= 19)).astype(int)
        df_eng['is_off_hours'] = ((df_eng['hour_of_day'] >= 0) & (df_eng['hour_of_day'] <= 5) | 
                                  (df_eng['hour_of_day'] >= 22)).astype(int)
        df_eng['is_business_hours'] = ((df_eng['hour_of_day'] >= 9) & (df_eng['hour_of_day'] <= 17)).astype(int)
    
    # Statistical features for amounts
    if 'amount' in df_eng.columns:
        df_eng['amount_zscore'] = (df_eng['amount'] - df_eng['amount'].mean()) / df_eng['amount'].std()
        df_eng['amount_percentile'] = df_eng['amount'].rank(pct=True)
        
        # Amount deviation from user average
        if 'user_id' in df_eng.columns and 'avg_transaction_amount' in df_eng.columns:
            df_eng['amount_deviation_from_avg'] = df_eng['amount'] - df_eng['avg_transaction_amount']
            df_eng['amount_ratio_to_avg'] = df_eng['amount'] / (df_eng['avg_transaction_amount'] + 1e-6)
        
        # High-value transaction flag
        amount_95th = df_eng['amount'].quantile(0.95)
        df_eng['is_high_value'] = (df_eng['amount'] > amount_95th).astype(int)
    
    # Account age features
    if 'account_age_days' in df_eng.columns:
        df_eng['account_age_months'] = df_eng['account_age_days'] / 30.0
        df_eng['is_new_account'] = (df_eng['account_age_days'] < 30).astype(int)
        df_eng['is_mature_account'] = (df_eng['account_age_days'] > 365).astype(int)
    
    # Transaction velocity features
    if 'transaction_count_24h' in df_eng.columns:
        df_eng['transaction_velocity'] = df_eng['transaction_count_24h'] / 24.0
        df_eng['is_high_velocity'] = (df_eng['transaction_count_24h'] > df_eng['transaction_count_24h'].quantile(0.90)).astype(int)
    
    # Interaction features
    if 'amount' in df_eng.columns:
        if 'transaction_count_24h' in df_eng.columns:
            df_eng['amount_x_transaction_count'] = df_eng['amount'] * df_eng['transaction_count_24h']
        if 'is_foreign_transaction' in df_eng.columns:
            df_eng['amount_x_is_foreign'] = df_eng['amount'] * df_eng['is_foreign_transaction']
        if 'hour_of_day' in df_eng.columns:
            df_eng['amount_x_hour'] = df_eng['amount'] * df_eng['hour_of_day']
    
    # Risk scoring (composite feature)
    risk_score = 0
    
    if 'is_foreign_transaction' in df_eng.columns:
        risk_score += df_eng['is_foreign_transaction'] * 0.2
    if 'is_off_hours' in df_eng.columns:
        risk_score += df_eng['is_off_hours'] * 0.15
    if 'is_high_value' in df_eng.columns:
        risk_score += df_eng['is_high_value'] * 0.15
    if 'is_new_account' in df_eng.columns:
        risk_score += df_eng['is_new_account'] * 0.1
    if 'is_high_velocity' in df_eng.columns:
        risk_score += df_eng['is_high_velocity'] * 0.2
    if 'amount_zscore' in df_eng.columns:
        risk_score += np.abs(df_eng['amount_zscore']) * 0.1
    
    df_eng['risk_score'] = risk_score
    
    # User behavior patterns
    if 'user_id' in df_eng.columns and 'amount' in df_eng.columns:
        user_stats = df_eng.groupby('user_id')['amount'].agg(['mean', 'std', 'count']).reset_index()
        user_stats.columns = ['user_id', 'user_avg_amount', 'user_std_amount', 'user_total_transactions']
        df_eng = df_eng.merge(user_stats, on='user_id', how='left')
        
        # Fill NaN values
        df_eng['user_std_amount'] = df_eng['user_std_amount'].fillna(0)
        
        # Deviation from user's normal behavior
        if 'user_avg_amount' in df_eng.columns:
            df_eng['amount_deviation_from_user_avg'] = df_eng['amount'] - df_eng['user_avg_amount']
    
    # Location and device stability
    if 'user_id' in df_eng.columns:
        if 'location' in df_eng.columns:
            user_locations = df_eng.groupby('user_id')['location'].nunique().reset_index()
            user_locations.columns = ['user_id', 'user_unique_locations']
            df_eng = df_eng.merge(user_locations, on='user_id', how='left')
            df_eng['location_changed'] = (df_eng['user_unique_locations'] > 1).astype(int)
        
        if 'device_type' in df_eng.columns:
            user_devices = df_eng.groupby('user_id')['device_type'].nunique().reset_index()
            user_devices.columns = ['user_id', 'user_unique_devices']
            df_eng = df_eng.merge(user_devices, on='user_id', how='left')
            df_eng['device_changed'] = (df_eng['user_unique_devices'] > 1).astype(int)
        
        if 'merchant_category' in df_eng.columns:
            user_merchants = df_eng.groupby('user_id')['merchant_category'].nunique().reset_index()
            user_merchants.columns = ['user_id', 'user_unique_merchants']
            df_eng = df_eng.merge(user_merchants, on='user_id', how='left')
    
    # Rolling statistics (if timestamp is available and sorted)
    if 'timestamp' in df_eng.columns and 'amount' in df_eng.columns and 'user_id' in df_eng.columns:
        try:
            df_eng_sorted = df_eng.sort_values(['user_id', 'timestamp']).copy()
            df_eng_sorted['amount_rolling_mean_7d'] = df_eng_sorted.groupby('user_id')['amount'].transform(
                lambda x: x.rolling(window=7, min_periods=1).mean()
            )
            df_eng_sorted['amount_rolling_std_7d'] = df_eng_sorted.groupby('user_id')['amount'].transform(
                lambda x: x.rolling(window=7, min_periods=1).std()
            )
            df_eng_sorted['amount_rolling_max_7d'] = df_eng_sorted.groupby('user_id')['amount'].transform(
                lambda x: x.rolling(window=7, min_periods=1).max()
            )
            
            # Merge back
            df_eng = df_eng_sorted.sort_index()
        except:
            print("Warning: Could not compute rolling statistics")
    
    print(f"Advanced features created. Total features: {df_eng.shape[1]}")
    return df_eng

def get_feature_importance_categories():
    """
    Return feature categories for analysis
    
    Returns:
    --------
    dict
        Dictionary mapping feature categories to feature names
    """
    return {
        'basic': ['amount', 'user_age', 'account_age_days', 'transaction_count_24h', 
                 'avg_transaction_amount', 'is_foreign_transaction', 'is_weekend', 'hour_of_day'],
        'time_based': ['day_of_week', 'month', 'quarter', 'is_month_end', 'is_month_start',
                      'is_rush_hour', 'is_off_hours', 'is_business_hours'],
        'statistical': ['amount_zscore', 'amount_percentile', 'amount_deviation_from_avg',
                       'amount_ratio_to_avg', 'amount_rolling_mean_7d', 'amount_rolling_std_7d'],
        'interaction': ['amount_x_transaction_count', 'amount_x_is_foreign', 'amount_x_hour'],
        'behavioral': ['transaction_velocity', 'is_high_velocity', 'risk_score',
                      'amount_deviation_from_user_avg'],
        'account': ['account_age_months', 'is_new_account', 'is_mature_account'],
        'user_patterns': ['user_avg_amount', 'user_std_amount', 'user_total_transactions',
                         'location_changed', 'device_changed']
    }

if __name__ == "__main__":
    # Test the feature engineering
    import pandas as pd
    
    print("Testing advanced feature engineering...")
    print("Developer: Molla Samser")
    print("Website: https://rskworld.in")
    
    # Load sample data
    df = pd.read_csv('fraud_detection_dataset.csv')
    print(f"\nOriginal shape: {df.shape}")
    
    # Engineer features
    df_eng = engineer_advanced_features(df)
    print(f"Engineered shape: {df_eng.shape}")
    print(f"\nNew features: {df_eng.shape[1] - df.shape[1]}")
    print(f"\nNew columns: {[col for col in df_eng.columns if col not in df.columns]}")
    
    # Save engineered dataset
    df_eng.to_csv('fraud_detection_dataset_advanced.csv', index=False)
    print("\nEngineered dataset saved to fraud_detection_dataset_advanced.csv")

