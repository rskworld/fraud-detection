"""
Hyperparameter Tuning for Fraud Detection Models

Developer: Molla Samser
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in, info@rskworld.com
Phone: +91 93305 39277
Company: RSK World
Description: Hyperparameter tuning using GridSearchCV and RandomizedSearchCV
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Advanced models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

def load_and_preprocess_data(file_path='fraud_detection_dataset.csv', use_advanced_features=False):
    """
    Load and preprocess data
    
    Parameters:
    -----------
    file_path : str
        Path to dataset
    use_advanced_features : bool
        Whether to use advanced feature engineering
    
    Returns:
    --------
    tuple
        X, y
    """
    print("Loading and preprocessing data...")
    df = pd.read_csv(file_path)
    
    if use_advanced_features:
        from advanced_feature_engineering import engineer_advanced_features
        df = engineer_advanced_features(df)
    
    # Preprocess
    df_model = df.copy()
    
    # Encode categorical variables
    categorical_cols = ['merchant_category', 'location', 'device_type', 'user_id']
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_model.columns:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            label_encoders[col] = le
    
    # Drop unnecessary columns
    drop_cols = ['transaction_id', 'is_fraud', 'timestamp']
    X = df_model.drop([col for col in drop_cols if col in df_model.columns], axis=1)
    y = df_model['is_fraud']
    
    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    
    print(f"Data shape: {X_balanced.shape}")
    return X_balanced, y_balanced

def tune_random_forest(X, y, cv=3, n_iter=50):
    """
    Tune Random Forest hyperparameters
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    cv : int
        Number of CV folds
    n_iter : int
        Number of iterations for RandomizedSearch
    
    Returns:
    --------
    sklearn.model_selection.RandomizedSearchCV
        Best model
    """
    print("\n" + "="*50)
    print("Tuning Random Forest...")
    print("="*50)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Use RandomizedSearchCV for faster tuning
    random_search = RandomizedSearchCV(
        rf, param_grid, 
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X, y)
    
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    return random_search

def tune_xgboost(X, y, cv=3, n_iter=50):
    """
    Tune XGBoost hyperparameters
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    cv : int
        Number of CV folds
    n_iter : int
        Number of iterations
    
    Returns:
    --------
    sklearn.model_selection.RandomizedSearchCV
        Best model
    """
    if not XGBOOST_AVAILABLE:
        print("XGBoost not available. Skipping...")
        return None
    
    print("\n" + "="*50)
    print("Tuning XGBoost...")
    print("="*50)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=(y == 0).sum() / (y == 1).sum()
    )
    
    random_search = RandomizedSearchCV(
        xgb_model, param_grid,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X, y)
    
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    return random_search

def tune_lightgbm(X, y, cv=3, n_iter=50):
    """
    Tune LightGBM hyperparameters
    
    Parameters:
    -----------
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    cv : int
        Number of CV folds
    n_iter : int
        Number of iterations
    
    Returns:
    --------
    sklearn.model_selection.RandomizedSearchCV
        Best model
    """
    if not LIGHTGBM_AVAILABLE:
        print("LightGBM not available. Skipping...")
        return None
    
    print("\n" + "="*50)
    print("Tuning LightGBM...")
    print("="*50)
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 10, -1],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_samples': [20, 30, 50],
        'num_leaves': [31, 50, 100]
    }
    
    lgb_model = lgb.LGBMClassifier(
        random_state=42,
        class_weight='balanced',
        verbose=-1
    )
    
    random_search = RandomizedSearchCV(
        lgb_model, param_grid,
        n_iter=n_iter,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    
    random_search.fit(X, y)
    
    print(f"\nBest parameters: {random_search.best_params_}")
    print(f"Best CV score: {random_search.best_score_:.4f}")
    
    return random_search

def compare_tuned_models(models_dict, X_test, y_test):
    """
    Compare tuned models on test set
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of model names and tuned models
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    """
    print("\n" + "="*50)
    print("MODEL COMPARISON ON TEST SET")
    print("="*50)
    
    results = []
    
    for name, model in models_dict.items():
        if model is None:
            continue
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        results.append({
            'Model': name,
            'ROC-AUC': auc,
            'F1-Score': f1,
            'Precision': precision,
            'Recall': recall
        })
        
        print(f"\n{name}:")
        print(f"  ROC-AUC: {auc:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ROC-AUC', ascending=False)
    
    print("\n" + "="*50)
    print("RANKED MODELS:")
    print("="*50)
    print(results_df.to_string(index=False))
    
    return results_df

def main():
    """Main function"""
    print("="*50)
    print("HYPERPARAMETER TUNING FOR FRAUD DETECTION")
    print("Developer: Molla Samser")
    print("Website: https://rskworld.in")
    print("="*50)
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(use_advanced_features=True)
    
    # Split into train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Tune models
    tuned_models = {}
    
    tuned_models['RandomForest'] = tune_random_forest(X_train, y_train, cv=3, n_iter=30)
    tuned_models['XGBoost'] = tune_xgboost(X_train, y_train, cv=3, n_iter=30)
    tuned_models['LightGBM'] = tune_lightgbm(X_train, y_train, cv=3, n_iter=30)
    
    # Compare models
    results_df = compare_tuned_models(tuned_models, X_test, y_test)
    
    # Save best model
    best_model_name = results_df.iloc[0]['Model']
    best_model = tuned_models[best_model_name].best_estimator_
    
    joblib.dump(best_model, f'best_tuned_model_{best_model_name.lower()}.pkl')
    print(f"\nBest model ({best_model_name}) saved to best_tuned_model_{best_model_name.lower()}.pkl")
    
    # Save results
    results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
    print("Results saved to hyperparameter_tuning_results.csv")

if __name__ == "__main__":
    main()

