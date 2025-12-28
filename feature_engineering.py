"""
Advanced Feature Engineering for Fraud Detection

Developer: Molla Samser
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in, info@rskworld.com
Phone: +91 93305 39277
Company: RSK World
Description: Advanced feature engineering functions for fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

def engineer_advanced_features(df):
    """
    Create advanced features from raw transaction data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw transaction dataframe with timestamp, user_id, amount, etc.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional engineered features
    """
    df = df.copy()
    
    # Ensure timestamp is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    print("Engineering advanced features...")
    print(f"Starting with {df.shape[1]} features")
    
    # Time-based features
    if 'timestamp' in df.columns:
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['day_of_month'] = df['timestamp'].dt.day
        df['is_month_end'] = (df['timestamp'].dt.day > 25).astype(int)
        df['is_month_start'] = (df['timestamp'].dt.day <= 5).astype(int)
        df['quarter'] = df['timestamp'].dt.quarter
    
    # Transaction velocity features
    if 'user_id' in df.columns and 'timestamp' in df.columns:
        df['time_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
        df['time_since_last_transaction'] = df['time_since_last_transaction'].fillna(24)
        
        # Transaction frequency
        df['transactions_per_hour'] = 1 / (df['time_since_last_transaction'] + 0.1)
        df['transactions_per_day'] = df['transactions_per_hour'] * 24
    
    # Rolling window features
    if 'user_id' in df.columns and 'amount' in df.columns:
        # Amount statistics over rolling windows
        for window in [3, 7, 14, 30]:
            df[f'amount_rolling_mean_{window}d'] = df.groupby('user_id')['amount'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'amount_rolling_std_{window}d'] = df.groupby('user_id')['amount'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            df[f'amount_rolling_max_{window}d'] = df.groupby('user_id')['amount'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
    
    # Z-scores and percentiles
    if 'user_id' in df.columns and 'amount' in df.columns:
        df['amount_zscore'] = df.groupby('user_id')['amount'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        df['amount_zscore'] = df['amount_zscore'].fillna(0)
        
        df['amount_percentile'] = df.groupby('user_id')['amount'].transform(lambda x: x.rank(pct=True))
        df['amount_percentile'] = df['amount_percentile'].fillna(0.5)
        
        # Deviation from average
        if 'avg_transaction_amount' in df.columns:
            df['amount_deviation_from_avg'] = (df['amount'] - df['avg_transaction_amount']) / (df['avg_transaction_amount'] + 1)
    
    # Change indicators
    change_cols = ['location', 'device_type', 'merchant_category']
    for col in change_cols:
        if col in df.columns and 'user_id' in df.columns:
            df[f'{col}_changed'] = (df.groupby('user_id')[col].shift() != df[col]).astype(int)
            df[f'{col}_changed'] = df[f'{col}_changed'].fillna(0)
            
            # Count of unique values in recent transactions
            df[f'{col}_unique_count_7d'] = df.groupby('user_id')[col].transform(
                lambda x: x.rolling(window=7, min_periods=1).nunique()
            )
    
    # Risk score calculation
    risk_factors = []
    if 'is_foreign_transaction' in df.columns:
        risk_factors.append(df['is_foreign_transaction'] * 0.15)
    if 'transaction_count_24h' in df.columns:
        risk_factors.append((df['transaction_count_24h'] > 5).astype(int) * 0.15)
    if 'amount' in df.columns and 'user_id' in df.columns:
        amount_threshold = df.groupby('user_id')['amount'].transform('quantile', 0.95)
        risk_factors.append((df['amount'] > amount_threshold).astype(int) * 0.2)
    if 'account_age_days' in df.columns:
        risk_factors.append((df['account_age_days'] < 30).astype(int) * 0.15)
    if 'hour_of_day' in df.columns:
        risk_factors.append((df['hour_of_day'].isin([0, 1, 2, 3, 4, 5, 22, 23])).astype(int) * 0.1)
    if 'time_since_last_transaction' in df.columns:
        risk_factors.append((df['time_since_last_transaction'] < 1).astype(int) * 0.1)
    if 'location_changed' in df.columns:
        risk_factors.append(df['location_changed'] * 0.15)
    
    if risk_factors:
        df['risk_score'] = pd.concat(risk_factors, axis=1).sum(axis=1)
        df['risk_score'] = df['risk_score'].clip(0, 1)
    
    # Interaction features
    if 'amount' in df.columns and 'transaction_count_24h' in df.columns:
        df['amount_x_transaction_count'] = df['amount'] * df['transaction_count_24h']
    if 'amount' in df.columns and 'is_foreign_transaction' in df.columns:
        df['amount_x_is_foreign'] = df['amount'] * df['is_foreign_transaction']
    if 'transaction_count_24h' in df.columns and 'is_foreign_transaction' in df.columns:
        df['transaction_count_x_is_foreign'] = df['transaction_count_24h'] * df['is_foreign_transaction']
    
    # Time pattern features
    if 'hour_of_day' in df.columns:
        df['is_rush_hour'] = df['hour_of_day'].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df['is_off_hours'] = df['hour_of_day'].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
        df['is_business_hours'] = df['hour_of_day'].isin(range(9, 18)).astype(int)
    
    # User behavior patterns
    if 'amount' in df.columns and 'avg_transaction_amount' in df.columns:
        df['avg_amount_ratio'] = df['amount'] / (df['avg_transaction_amount'] + 1)
    if 'transaction_count_24h' in df.columns and 'time_since_last_transaction' in df.columns:
        df['transaction_velocity'] = df['transaction_count_24h'] / (df['time_since_last_transaction'] + 0.1)
    
    # Account age features
    if 'account_age_days' in df.columns:
        df['account_age_months'] = df['account_age_days'] / 30
        df['is_new_account'] = (df['account_age_days'] < 30).astype(int)
        df['is_mature_account'] = (df['account_age_days'] > 365).astype(int)
    
    # Fill NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Fill object columns with 'unknown'
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        if col not in ['transaction_id', 'user_id', 'timestamp']:
            df[col] = df[col].fillna('unknown')
    
    print(f"Finished with {df.shape[1]} features")
    print(f"Added {df.shape[1] - len([c for c in df.columns if c not in df.columns])} new features")
    
    return df

def scale_features(X, scaler_type='standard'):
    """
    Scale features for better model performance
    
    Parameters:
    -----------
    X : pd.DataFrame or np.array
        Features to scale
    scaler_type : str
        Type of scaler ('standard' or 'robust')
    
    Returns:
    --------
    tuple
        Scaled features and scaler object
    """
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("scaler_type must be 'standard' or 'robust'")
    
    if isinstance(X, pd.DataFrame):
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    else:
        X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

if __name__ == "__main__":
    # Example usage
    print("Advanced Feature Engineering Module")
    print("Developer: Molla Samser")
    print("Website: https://rskworld.in")
    print("-" * 50)
    
    # Load sample data
    try:
        df = pd.read_csv('fraud_detection_dataset.csv')
        print(f"Loaded dataset with {df.shape[0]} records and {df.shape[1]} features")
        
        # Engineer features
        df_engineered = engineer_advanced_features(df)
        print(f"\nEngineered dataset has {df_engineered.shape[1]} features")
        print(f"\nNew features added:")
        new_features = [c for c in df_engineered.columns if c not in df.columns]
        print(f"Total new features: {len(new_features)}")
        for feat in new_features[:20]:  # Show first 20
            print(f"  - {feat}")
        if len(new_features) > 20:
            print(f"  ... and {len(new_features) - 20} more")
    except FileNotFoundError:
        print("Please generate the dataset first using generate_data.py")

