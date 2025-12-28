"""
Dataset Verification Script

Developer: Molla Samser
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in, info@rskworld.com
Phone: +91 93305 39277
Company: RSK World
Description: Verifies the generated fraud detection dataset
"""

import pandas as pd

# Load dataset
df = pd.read_csv('fraud_detection_dataset.csv')

print("="*50)
print("DATASET VERIFICATION")
print("Developer: Molla Samser")
print("Website: https://rskworld.in")
print("="*50)
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFraud ratio: {df['is_fraud'].mean():.2%}")
print(f"Fraudulent transactions: {df['is_fraud'].sum()}")
print(f"Normal transactions: {(df['is_fraud'] == 0).sum()}")
print("\nSample data:")
print(df.head(5))
print("\nDataset info:")
print(df.info())
print("\nDataset statistics:")
print(df.describe())
print(f"\nTotal features: {len(df.columns)}")
print("\nAll features:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

