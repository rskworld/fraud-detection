"""
Fraud Detection Dataset Generator

Developer: Molla Samser
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in, info@rskworld.com
Phone: +91 93305 39277
Company: RSK World
Description: Generates synthetic fraud detection dataset with transaction records
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_fraud_dataset(n_samples=10000, fraud_ratio=0.05):
    """
    Generate synthetic fraud detection dataset
    
    Parameters:
    -----------
    n_samples : int
        Total number of transactions to generate
    fraud_ratio : float
        Ratio of fraudulent transactions (for imbalanced dataset)
    
    Returns:
    --------
    pd.DataFrame
        Dataset with transaction features and fraud labels
    """
    
    n_fraud = int(n_samples * fraud_ratio)
    n_normal = n_samples - n_fraud
    
    # Merchant categories
    merchant_categories = ['Retail', 'Grocery', 'Restaurant', 'Gas', 'Online', 
                          'Pharmacy', 'Travel', 'Entertainment', 'Utilities', 'Other']
    
    # Device types
    device_types = ['Mobile', 'Desktop', 'Tablet', 'ATM', 'POS']
    
    # Location codes
    locations = ['US', 'UK', 'CA', 'AU', 'FR', 'DE', 'IT', 'ES', 'NL', 'BE']
    
    data = []
    
    # Generate normal transactions
    for i in range(n_normal):
        user_id = np.random.randint(1, 5000)
        amount = np.random.lognormal(mean=3.5, sigma=1.2)
        amount = min(amount, 10000)  # Cap at 10000
        
        transaction = {
            'transaction_id': f'TXN_{i+1:06d}',
            'user_id': f'USER_{user_id:04d}',
            'amount': round(amount, 2),
            'merchant_category': np.random.choice(merchant_categories),
            'location': np.random.choice(locations),
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365), 
                                                    hours=np.random.randint(0, 24),
                                                    minutes=np.random.randint(0, 60)),
            'device_type': np.random.choice(device_types),
            'user_age': np.random.randint(18, 80),
            'account_age_days': np.random.randint(1, 3650),
            'transaction_count_24h': np.random.poisson(2),
            'avg_transaction_amount': round(np.random.lognormal(mean=3.5, sigma=1.0), 2),
            'is_foreign_transaction': np.random.choice([0, 1], p=[0.7, 0.3]),
            'is_weekend': np.random.choice([0, 1], p=[0.7, 0.3]),
            'hour_of_day': np.random.randint(0, 24),
            'is_fraud': 0
        }
        data.append(transaction)
    
    # Generate fraudulent transactions (with different patterns)
    for i in range(n_fraud):
        user_id = np.random.randint(1, 5000)
        # Fraud transactions tend to be larger
        amount = np.random.lognormal(mean=4.5, sigma=1.5)
        amount = min(amount, 50000)  # Higher cap for fraud
        
        # Fraud more likely at unusual hours
        unusual_hours = [0, 1, 2, 3, 4, 5, 22, 23]
        normal_hours = list(range(6, 22))
        all_hours = unusual_hours + normal_hours
        # Create probabilities: 0.12 for unusual hours, rest distributed among normal hours
        p_unusual = 0.12
        total_unusual_prob = len(unusual_hours) * p_unusual
        p_normal = (1 - total_unusual_prob) / len(normal_hours)
        probs = [p_unusual] * len(unusual_hours) + [p_normal] * len(normal_hours)
        hour = np.random.choice(all_hours, p=probs)
        
        transaction = {
            'transaction_id': f'TXN_{n_normal+i+1:06d}',
            'user_id': f'USER_{user_id:04d}',
            'amount': round(amount, 2),
            'merchant_category': np.random.choice(merchant_categories),
            'location': np.random.choice(locations),  # Fraud more likely foreign
            'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365), 
                                                    hours=int(hour),
                                                    minutes=np.random.randint(0, 60)),
            'device_type': np.random.choice(device_types),
            'user_age': np.random.randint(18, 80),
            'account_age_days': np.random.randint(1, 365),  # Newer accounts more risky
            'transaction_count_24h': np.random.poisson(5),  # More transactions
            'avg_transaction_amount': round(np.random.lognormal(mean=4.0, sigma=1.2), 2),
            'is_foreign_transaction': np.random.choice([0, 1], p=[0.3, 0.7]),  # More foreign
            'is_weekend': np.random.choice([0, 1], p=[0.6, 0.4]),
            'hour_of_day': hour,
            'is_fraud': 1
        }
        data.append(transaction)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    
    # Advanced Feature Engineering
    print("Generating advanced features...")
    
    # Time-based features
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_month_end'] = (df['timestamp'].dt.day > 25).astype(int)
    df['is_month_start'] = (df['timestamp'].dt.day <= 5).astype(int)
    
    # Transaction velocity features
    df['time_since_last_transaction'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600  # hours
    df['time_since_last_transaction'] = df['time_since_last_transaction'].fillna(24)  # First transaction assumed 24h ago
    
    # Transaction count features (simplified - count previous transactions)
    df['transaction_count_7d'] = df.groupby('user_id').cumcount() + 1
    df['transaction_count_30d'] = df.groupby('user_id').cumcount() + 1
    
    # Amount-based features
    df['amount_zscore'] = df.groupby('user_id')['amount'].transform(lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0)
    df['amount_zscore'] = df['amount_zscore'].fillna(0)
    df['amount_percentile'] = df.groupby('user_id')['amount'].transform(lambda x: x.rank(pct=True))
    df['amount_percentile'] = df['amount_percentile'].fillna(0.5)
    
    # Rolling statistics (using expanding window for simplicity)
    df['amount_rolling_mean_7d'] = df.groupby('user_id')['amount'].transform(lambda x: x.expanding(min_periods=1).mean())
    df['amount_rolling_std_7d'] = df.groupby('user_id')['amount'].transform(lambda x: x.expanding(min_periods=1).std())
    df['amount_rolling_std_7d'] = df['amount_rolling_std_7d'].fillna(0)
    
    # Location change indicator
    df['location_changed'] = (df.groupby('user_id')['location'].shift() != df['location']).astype(int)
    df['location_changed'] = df['location_changed'].fillna(0)
    
    # Device change indicator
    df['device_changed'] = (df.groupby('user_id')['device_type'].shift() != df['device_type']).astype(int)
    df['device_changed'] = df['device_changed'].fillna(0)
    
    # Merchant category change
    df['merchant_category_changed'] = (df.groupby('user_id')['merchant_category'].shift() != df['merchant_category']).astype(int)
    df['merchant_category_changed'] = df['merchant_category_changed'].fillna(0)
    
    # Risk score based on multiple factors
    df['risk_score'] = (
        (df['is_foreign_transaction'] * 0.2) +
        (df['transaction_count_24h'] > 5).astype(int) * 0.15 +
        (df['amount'] > df.groupby('user_id')['amount'].transform('quantile', 0.95)).astype(int) * 0.2 +
        (df['account_age_days'] < 30).astype(int) * 0.15 +
        (df['hour_of_day'].isin([0, 1, 2, 3, 4, 5, 22, 23])).astype(int) * 0.1 +
        (df['time_since_last_transaction'] < 1).astype(int) * 0.1 +
        (df['location_changed'] * 0.1)
    )
    
    # Interaction features
    df['amount_x_transaction_count'] = df['amount'] * df['transaction_count_24h']
    df['amount_x_is_foreign'] = df['amount'] * df['is_foreign_transaction']
    df['transaction_count_x_is_foreign'] = df['transaction_count_24h'] * df['is_foreign_transaction']
    
    # Time patterns
    df['is_rush_hour'] = df['hour_of_day'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    df['is_off_hours'] = df['hour_of_day'].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
    
    # User behavior patterns
    df['avg_amount_ratio'] = df['amount'] / (df['avg_transaction_amount'] + 1)
    df['transaction_velocity'] = df['transaction_count_24h'] / (df['time_since_last_transaction'] + 0.1)
    
    # Fill NaN values
    df = df.fillna(0)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("Generating fraud detection dataset...")
    print("Developer: Molla Samser")
    print("Website: https://rskworld.in")
    print("-" * 50)
    
    # Generate dataset
    df = generate_fraud_dataset(n_samples=10000, fraud_ratio=0.05)
    
    # Save to CSV
    df.to_csv('fraud_detection_dataset.csv', index=False)
    print(f"Dataset saved to fraud_detection_dataset.csv")
    print(f"Total transactions: {len(df)}")
    print(f"Fraudulent transactions: {df['is_fraud'].sum()}")
    print(f"Normal transactions: {(df['is_fraud'] == 0).sum()}")
    print(f"Fraud ratio: {df['is_fraud'].mean():.2%}")
    
    # Save to Excel (optional)
    try:
        df.to_excel('fraud_detection_dataset.xlsx', index=False)
        print(f"Dataset saved to fraud_detection_dataset.xlsx")
    except Exception as e:
        print(f"Note: Excel file not generated ({str(e)}). CSV file is available.")
    
    # Display sample
    print("\nSample data:")
    print(df.head(10))
    print("\nDataset statistics:")
    print(df.describe())

