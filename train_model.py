"""
Fraud Detection Model Training Script

Developer: Molla Samser
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in, info@rskworld.com
Phone: +91 93305 39277
Company: RSK World
Description: Trains machine learning models for fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

# Set style for plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

def load_data(file_path='fraud_detection_dataset.csv'):
    """
    Load the fraud detection dataset
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset
    """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {len(df)} records")
    return df

def preprocess_data(df, use_advanced_features=True):
    """
    Preprocess the data for machine learning with advanced feature engineering
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
    use_advanced_features : bool
        Whether to use advanced feature engineering
    
    Returns:
    --------
    tuple
        Features (X) and target (y)
    """
    print("\nPreprocessing data...")
    
    # Import feature engineering module
    if use_advanced_features:
        try:
            from advanced_feature_engineering import engineer_advanced_features
            print("Using advanced feature engineering...")
            df_processed = engineer_advanced_features(df)
        except ImportError:
            print("Advanced feature engineering module not found, using basic preprocessing...")
            df_processed = df.copy()
    else:
        df_processed = df.copy()
    
    # Convert timestamp to datetime if it's string
    if 'timestamp' in df_processed.columns:
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
        if 'day_of_week' not in df_processed.columns:
            df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
        if 'month' not in df_processed.columns:
            df_processed['month'] = df_processed['timestamp'].dt.month
        df_processed = df_processed.drop('timestamp', axis=1)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['merchant_category', 'location', 'device_type', 'user_id']
    
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Drop transaction_id (not useful for prediction)
    if 'transaction_id' in df_processed.columns:
        df_processed = df_processed.drop('transaction_id', axis=1)
    
    # Separate features and target
    X = df_processed.drop('is_fraud', axis=1)
    y = df_processed['is_fraud']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y, label_encoders

def handle_imbalance(X, y):
    """
    Handle class imbalance using SMOTE
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    
    Returns:
    --------
    tuple
        Balanced X and y
    """
    print("\nHandling class imbalance with SMOTE...")
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    print(f"After SMOTE - Features shape: {X_balanced.shape}")
    print(f"After SMOTE - Target distribution:\n{pd.Series(y_balanced).value_counts()}")
    return X_balanced, y_balanced

def train_advanced_models(X, y, use_smote=True):
    """
    Train multiple advanced models for fraud detection
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    use_smote : bool
        Whether to use SMOTE for balancing
    
    Returns:
    --------
    dict
        Dictionary of trained models and their results
    """
    print("\nTraining advanced models...")
    
    # Handle imbalance if requested
    if use_smote:
        X_train, y_train = handle_imbalance(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    
    models = {}
    results = {}
    
    # 1. Random Forest
    print("\n" + "="*50)
    print("Training Random Forest...")
    print("="*50)
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    models['RandomForest'] = rf_model
    
    # 2. XGBoost
    if XGBOOST_AVAILABLE:
        print("\n" + "="*50)
        print("Training XGBoost...")
        print("="*50)
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
        )
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model
    
    # 3. LightGBM
    if LIGHTGBM_AVAILABLE:
        print("\n" + "="*50)
        print("Training LightGBM...")
        print("="*50)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )
        lgb_model.fit(X_train, y_train)
        models['LightGBM'] = lgb_model
    
    # Evaluate all models
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    for model_name, model in models.items():
        print(f"\n{model_name} Evaluation:")
        print("-" * 50)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Average Precision: {avg_precision:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        results[model_name] = {
            'model': model,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision
        }
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    print(f"\n{'='*70}")
    print(f"Best Model: {best_model_name} (ROC AUC: {results[best_model_name]['roc_auc']:.4f})")
    print(f"{'='*70}")
    
    # Plot comparison
    plot_model_comparison(results, y_test, X.columns)
    
    return models, results, X_test, y_test, best_model_name

def plot_model_comparison(results, y_test, feature_names):
    """Plot comparison of different models"""
    
    # ROC Curves Comparison
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for model_name, result in results.items():
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {result['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Precision-Recall Curves
    plt.subplot(1, 2, 2)
    for model_name, result in results.items():
        precision, recall, _ = precision_recall_curve(y_test, result['y_pred_proba'])
        plt.plot(recall, precision, label=f"{model_name} (AP = {result['avg_precision']:.3f})")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nModel comparison plots saved to model_comparison.png")
    
    # Feature importance for best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['roc_auc'])
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 8))
        top_features = feature_importance.head(20)
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title(f'Top 20 Feature Importances - {best_model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.savefig('feature_importance_advanced.png', dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to feature_importance_advanced.png")
        
        print(f"\nTop 15 Most Important Features ({best_model_name}):")
        print(feature_importance.head(15))

def train_model(X, y, use_smote=True):
    """
    Train Random Forest model for fraud detection (backward compatibility)
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target variable
    use_smote : bool
        Whether to use SMOTE for balancing
    
    Returns:
    --------
    sklearn.estimator
        Trained model
    """
    models, results, X_test, y_test, best_model_name = train_advanced_models(X, y, use_smote)
    return results[best_model_name]['model'], X_test, y_test

def save_model(model, filename='fraud_detection_model.pkl'):
    """
    Save the trained model
    
    Parameters:
    -----------
    model : sklearn.estimator
        Trained model
    filename : str
        Output filename
    """
    joblib.dump(model, filename)
    print(f"\nModel saved to {filename}")

def main():
    """Main function"""
    print("="*50)
    print("FRAUD DETECTION MODEL TRAINING")
    print("Developer: Molla Samser")
    print("Website: https://rskworld.in")
    print("="*50)
    
    # Load data
    df = load_data()
    
    # Preprocess with advanced features
    X, y, label_encoders = preprocess_data(df, use_advanced_features=True)
    
    # Train advanced models
    models, results, X_test, y_test, best_model_name = train_advanced_models(X, y, use_smote=True)
    model = results[best_model_name]['model']
    
    # Save model
    save_model(model)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print("="*50)

if __name__ == "__main__":
    main()

