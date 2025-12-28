"""
Test all imports to check for errors

Developer: Molla Samser
Designer & Tester: Rima Khatun
Website: https://rskworld.in
Email: help@rskworld.in, support@rskworld.in, info@rskworld.com
Phone: +91 93305 39277
Company: RSK World
"""

import sys

def test_imports():
    """Test all module imports"""
    errors = []
    
    print("Testing module imports...")
    print("="*50)
    
    # Test basic imports
    modules = [
        'pandas',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'joblib'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"[OK] {module}")
        except ImportError as e:
            errors.append(f"{module}: {e}")
            print(f"[ERROR] {module}: {e}")
    
    # Test optional imports
    print("\nTesting optional imports...")
    optional_modules = [
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm'),
        ('shap', 'shap'),
        ('imblearn', 'imblearn')
    ]
    
    for display_name, module_name in optional_modules:
        try:
            __import__(module_name)
            print(f"[OK] {display_name}")
        except ImportError:
            print(f"[OPTIONAL] {display_name} (not installed - optional)")
    
    # Test project modules
    print("\nTesting project modules...")
    project_modules = [
        'advanced_feature_engineering',
        'model_evaluation_advanced',
        'predict_pipeline',
        'shap_explainability'
    ]
    
    for module in project_modules:
        try:
            __import__(module)
            print(f"[OK] {module}")
        except ImportError as e:
            errors.append(f"{module}: {e}")
            print(f"[ERROR] {module}: {e}")
    
    # Test hyperparameter_tuning separately (requires imblearn)
    print("\nTesting hyperparameter_tuning (requires imblearn)...")
    try:
        import imblearn
        try:
            import hyperparameter_tuning
            print("[OK] hyperparameter_tuning")
        except ImportError as e:
            errors.append(f"hyperparameter_tuning: {e}")
            print(f"[ERROR] hyperparameter_tuning: {e}")
    except ImportError:
        print("[OPTIONAL] hyperparameter_tuning (imblearn not installed - optional)")
    
    print("\n" + "="*50)
    if errors:
        print(f"Found {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("All imports successful!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)

