"""
SHAP Model Explainability for Fraud Detection

Developer: Molla Samser
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in, info@rskworld.com
Phone: +91 93305 39277
Company: RSK World
Description: SHAP values for model interpretability and explainability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

def explain_model_shap(model, X, feature_names=None, sample_size=100):
    """
    Generate SHAP explanations for model predictions
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X : pd.DataFrame or np.array
        Feature data
    feature_names : list
        List of feature names
    sample_size : int
        Number of samples to use for SHAP (for faster computation)
    
    Returns:
    --------
    shap.Explainer
        SHAP explainer object
    """
    if not SHAP_AVAILABLE:
        print("SHAP is not installed. Please install it first: pip install shap")
        return None
    
    print("Computing SHAP values...")
    print("Developer: Molla Samser | Website: https://rskworld.in")
    
    # Sample data if too large
    if len(X) > sample_size:
        print(f"Sampling {sample_size} instances for faster computation...")
        X_sample = X.sample(n=sample_size, random_state=42) if isinstance(X, pd.DataFrame) else X[:sample_size]
    else:
        X_sample = X
    
    # Create explainer based on model type
    model_type = type(model).__name__
    
    if 'XGBoost' in model_type or 'XGBClassifier' in model_type:
        explainer = shap.TreeExplainer(model)
    elif 'LightGBM' in model_type or 'LGBMClassifier' in model_type:
        explainer = shap.TreeExplainer(model)
    elif 'RandomForest' in model_type or 'RandomForestClassifier' in model_type:
        explainer = shap.TreeExplainer(model)
    else:
        # For other models, use KernelExplainer
        explainer = shap.KernelExplainer(model.predict_proba, X_sample[:100])
    
    shap_values = explainer.shap_values(X_sample)
    
    # Handle multi-class output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use fraud class (index 1)
    
    return explainer, shap_values, X_sample

def plot_shap_summary(shap_values, X_sample, feature_names=None, save_path=None):
    """Plot SHAP summary plot"""
    if not SHAP_AVAILABLE:
        return
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary plot saved to {save_path}")
    plt.show()

def plot_shap_summary_bar(shap_values, X_sample, feature_names=None, save_path=None):
    """Plot SHAP summary bar plot"""
    if not SHAP_AVAILABLE:
        return
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", feature_names=feature_names, show=False)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP summary bar plot saved to {save_path}")
    plt.show()

def plot_shap_waterfall(explainer, X_sample, instance_idx=0, feature_names=None, save_path=None):
    """Plot SHAP waterfall plot for a single instance"""
    if not SHAP_AVAILABLE:
        return
    
    shap_values = explainer.shap_values(X_sample.iloc[[instance_idx]] if isinstance(X_sample, pd.DataFrame) else X_sample[[instance_idx]])
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(values=shap_values[0],
                                         base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                                         data=X_sample.iloc[instance_idx] if isinstance(X_sample, pd.DataFrame) else X_sample[instance_idx],
                                         feature_names=feature_names), show=False)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"SHAP waterfall plot saved to {save_path}")
    plt.show()

def get_feature_importance_shap(shap_values, feature_names):
    """
    Get feature importance from SHAP values
    
    Parameters:
    -----------
    shap_values : np.array
        SHAP values
    feature_names : list
        Feature names
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature importance
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('shap_importance', ascending=False)
    
    return importance_df

def explain_prediction(explainer, X_instance, feature_names=None):
    """
    Explain a single prediction
    
    Parameters:
    -----------
    explainer : shap.Explainer
        SHAP explainer
    X_instance : pd.DataFrame or np.array
        Single instance to explain
    feature_names : list
        Feature names
    
    Returns:
    --------
    dict
        Explanation results
    """
    if not SHAP_AVAILABLE:
        return None
    
    shap_values = explainer.shap_values(X_instance)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    if isinstance(X_instance, pd.DataFrame):
        X_array = X_instance.values[0]
    else:
        X_array = X_instance[0]
    
    explanation = {
        'base_value': explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
        'feature_values': dict(zip(feature_names, X_array)),
        'shap_values': dict(zip(feature_names, shap_values[0])),
        'prediction': explainer.expected_value[1] + shap_values[0].sum() if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value + shap_values[0].sum()
    }
    
    return explanation

def main():
    """Example usage"""
    print("="*50)
    print("SHAP MODEL EXPLAINABILITY")
    print("Developer: Molla Samser")
    print("Website: https://rskworld.in")
    print("="*50)
    
    if not SHAP_AVAILABLE:
        print("\nSHAP is not installed. Please install it first:")
        print("pip install shap")
        return
    
    print("\nExample usage:")
    print("""
    # After training a model
    import joblib
    from shap_explainability import explain_model_shap, plot_shap_summary
    
    # Load model
    model = joblib.load('fraud_detection_model.pkl')
    
    # Get SHAP values
    explainer, shap_values, X_sample = explain_model_shap(
        model, X_test, feature_names=X_test.columns, sample_size=100
    )
    
    # Plot summary
    plot_shap_summary(shap_values, X_sample, feature_names=X_test.columns, 
                     save_path='shap_summary.png')
    
    # Plot summary bar
    plot_shap_summary_bar(shap_values, X_sample, feature_names=X_test.columns,
                         save_path='shap_summary_bar.png')
    """)

if __name__ == "__main__":
    main()

