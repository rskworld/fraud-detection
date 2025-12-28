"""
Advanced Model Evaluation and Visualization

Developer: Molla Samser
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in, info@rskworld.com
Phone: +91 93305 39277
Company: RSK World
Description: Comprehensive model evaluation with advanced metrics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, f1_score,
    precision_score, recall_score, accuracy_score, log_loss,
    matthews_corrcoef, cohen_kappa_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")

def evaluate_model_comprehensive(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive model evaluation with multiple metrics
    
    Parameters:
    -----------
    model : sklearn estimator
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    model_name : str
        Name of the model
    
    Returns:
    --------
    dict
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE EVALUATION: {model_name}")
    print(f"{'='*60}")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'average_precision': average_precision_score(y_test, y_pred_proba),
        'log_loss': log_loss(y_test, y_pred_proba),
        'matthews_corrcoef': matthews_corrcoef(y_test, y_pred),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred)
    }
    
    # Print metrics
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print(f"Matthews Correlation Coefficient: {metrics['matthews_corrcoef']:.4f}")
    print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    metrics['true_positive'] = tp
    metrics['true_negative'] = tn
    metrics['false_positive'] = fp
    metrics['false_negative'] = fn
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\nSpecificity: {metrics['specificity']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    
    return metrics, y_pred, y_pred_proba, cm

def plot_confusion_matrix(cm, model_name="Model", save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'],
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_roc_curve(y_test, y_pred_proba_list, model_names, save_path=None):
    """Plot ROC curves for multiple models"""
    plt.figure(figsize=(10, 8))
    
    for y_pred_proba, name in zip(y_pred_proba_list, model_names):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    plt.show()

def plot_precision_recall_curve(y_test, y_pred_proba_list, model_names, save_path=None):
    """Plot Precision-Recall curves for multiple models"""
    plt.figure(figsize=(10, 8))
    
    for y_pred_proba, name in zip(y_pred_proba_list, model_names):
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        ap = average_precision_score(y_test, y_pred_proba)
        plt.plot(recall, precision, label=f'{name} (AP = {ap:.4f})', linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curves saved to {save_path}")
    plt.show()

def plot_feature_importance_comparison(models_dict, feature_names, top_n=20, save_path=None):
    """Compare feature importance across models"""
    n_models = len(models_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 8))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(models_dict.items()):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            continue
        
        feature_imp_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        sns.barplot(data=feature_imp_df, y='feature', x='importance', ax=axes[idx])
        axes[idx].set_title(f'Top {top_n} Features - {name}', fontweight='bold')
        axes[idx].set_xlabel('Importance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance comparison saved to {save_path}")
    plt.show()

def cross_validate_model(model, X, y, cv=5, scoring_metrics=None):
    """
    Perform cross-validation with multiple metrics
    
    Parameters:
    -----------
    model : sklearn estimator
        Model to validate
    X : pd.DataFrame
        Features
    y : pd.Series
        Target
    cv : int
        Number of CV folds
    scoring_metrics : list
        List of scoring metrics
    
    Returns:
    --------
    dict
        Dictionary of CV scores
    """
    if scoring_metrics is None:
        scoring_metrics = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']
    
    cv_scores = {}
    cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, cv=cv_fold, scoring=metric, n_jobs=-1)
        cv_scores[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    for metric, scores in cv_scores.items():
        print(f"{metric.upper()}: {scores['mean']:.4f} (+/- {scores['std']*2:.4f})")
    
    return cv_scores

def plot_metrics_comparison(metrics_list, save_path=None):
    """Plot comparison of metrics across models"""
    df = pd.DataFrame(metrics_list)
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'average_precision']
    available_metrics = [m for m in metrics_to_plot if m in df.columns]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(available_metrics[:6]):
        df_plot = df[['model_name', metric]].sort_values(metric, ascending=False)
        sns.barplot(data=df_plot, y='model_name', x=metric, ax=axes[idx])
        axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
        axes[idx].set_xlabel('Score')
        axes[idx].set_ylabel('')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison saved to {save_path}")
    plt.show()

def main():
    """Example usage"""
    print("="*50)
    print("ADVANCED MODEL EVALUATION")
    print("Developer: Molla Samser")
    print("Website: https://rskworld.in")
    print("="*50)
    print("\nThis module provides comprehensive evaluation functions.")
    print("Import and use the functions in your model training scripts.")

if __name__ == "__main__":
    main()

