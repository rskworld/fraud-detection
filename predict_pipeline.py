"""
Fraud Detection Prediction Pipeline

Developer: Molla Samser
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in, info@rskworld.com
Phone: +91 93305 39277
Company: RSK World
Description: Production-ready prediction pipeline for fraud detection
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionPipeline:
    """
    Production-ready fraud detection pipeline
    """
    
    def __init__(self, model_path=None, use_advanced_features=True):
        """
        Initialize the pipeline
        
        Parameters:
        -----------
        model_path : str
            Path to saved model file
        use_advanced_features : bool
            Whether to use advanced feature engineering
        """
        self.model = None
        self.label_encoders = {}
        self.use_advanced_features = use_advanced_features
        self.feature_names = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load a trained model"""
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        print("Model loaded successfully!")
    
    def _engineer_features(self, df):
        """Engineer features for prediction"""
        from advanced_feature_engineering import engineer_advanced_features
        
        if self.use_advanced_features:
            df = engineer_advanced_features(df)
        
        return df
    
    def _preprocess(self, df):
        """Preprocess input data"""
        df_processed = df.copy()
        
        # Encode categorical variables (same as training)
        categorical_cols = ['merchant_category', 'location', 'device_type', 'user_id']
        
        for col in categorical_cols:
            if col in df_processed.columns:
                if col in self.label_encoders:
                    # Use existing encoder
                    df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))
                else:
                    # Create new encoder (for testing)
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
        
        # Drop unnecessary columns
        drop_cols = ['transaction_id', 'timestamp', 'is_fraud']
        df_processed = df_processed.drop([col for col in drop_cols if col in df_processed.columns], axis=1)
        
        return df_processed
    
    def predict(self, df):
        """
        Predict fraud for given transactions
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with transaction data
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Make a copy to avoid modifying original
        df_pred = df.copy()
        
        # Engineer features
        df_pred = self._engineer_features(df_pred)
        
        # Preprocess
        X = self._preprocess(df_pred)
        
        # Ensure feature order matches training
        if self.feature_names is not None:
            X = X.reindex(columns=self.feature_names, fill_value=0)
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        # Add predictions to dataframe
        result = df.copy()
        result['is_fraud_predicted'] = predictions
        result['fraud_probability'] = probabilities
        result['risk_level'] = pd.cut(probabilities, 
                                      bins=[0, 0.3, 0.7, 1.0],
                                      labels=['Low', 'Medium', 'High'])
        
        return result
    
    def predict_single(self, transaction_dict):
        """
        Predict fraud for a single transaction
        
        Parameters:
        -----------
        transaction_dict : dict
            Dictionary with transaction features
        
        Returns:
        --------
        dict
            Prediction results
        """
        df = pd.DataFrame([transaction_dict])
        result = self.predict(df)
        
        return {
            'is_fraud': int(result['is_fraud_predicted'].iloc[0]),
            'fraud_probability': float(result['fraud_probability'].iloc[0]),
            'risk_level': str(result['risk_level'].iloc[0])
        }
    
    def batch_predict(self, csv_path, output_path=None):
        """
        Batch prediction from CSV file
        
        Parameters:
        -----------
        csv_path : str
            Path to input CSV file
        output_path : str
            Path to save predictions (optional)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with predictions
        """
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} transactions")
        
        results = self.predict(df)
        
        if output_path:
            results.to_csv(output_path, index=False)
            print(f"Predictions saved to {output_path}")
        
        return results

def evaluate_predictions(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate prediction results
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like
        Prediction probabilities (optional)
    
    Returns:
    --------
    dict
        Evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    print("Evaluation Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
    
    return metrics

def main():
    """Example usage"""
    print("="*50)
    print("FRAUD DETECTION PREDICTION PIPELINE")
    print("Developer: Molla Samser")
    print("Website: https://rskworld.in")
    print("="*50)
    
    # Example usage
    print("\nExample usage:")
    print("""
    # Load pipeline with model
    pipeline = FraudDetectionPipeline(model_path='fraud_detection_model.pkl')
    
    # Predict single transaction
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
    
    # Batch prediction
    results = pipeline.batch_predict('test_transactions.csv', 'predictions.csv')
    """)

if __name__ == "__main__":
    main()

