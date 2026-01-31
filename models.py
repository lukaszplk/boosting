"""
Boosting model implementations and utilities.
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, log_loss
)
import numpy as np
from typing import Dict, Any, List, Optional
import warnings


def create_adaboost_model(config: Dict[str, Any]) -> AdaBoostClassifier:
    """
    Create an AdaBoost classifier.
    
    Args:
        config: Dictionary containing AdaBoost parameters
        
    Returns:
        Configured AdaBoostClassifier instance
    """
    # Handle deprecated base_estimator parameter
    config_copy = config.copy()
    base_estimator = None
    
    if 'base_estimator' in config_copy:
        base_estimator = config_copy.pop('base_estimator')
    
    if base_estimator is None:
        base_estimator = DecisionTreeClassifier(max_depth=1)
    
    # Use 'estimator' instead of deprecated 'base_estimator'
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AdaBoostClassifier(estimator=base_estimator, **config_copy)


def create_gradient_boosting_model(config: Dict[str, Any]) -> GradientBoostingClassifier:
    """
    Create a Gradient Boosting classifier.
    
    Args:
        config: Dictionary containing Gradient Boosting parameters
        
    Returns:
        Configured GradientBoostingClassifier instance
    """
    return GradientBoostingClassifier(**config)


def create_xgboost_model(config: Dict[str, Any]):
    """
    Create an XGBoost classifier.
    
    Args:
        config: Dictionary containing XGBoost parameters
        
    Returns:
        Configured XGBClassifier instance
    """
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(**config)
    except ImportError:
        raise ImportError("XGBoost not installed. Install with: pip install xgboost")


def create_lightgbm_model(config: Dict[str, Any]):
    """
    Create a LightGBM classifier.
    
    Args:
        config: Dictionary containing LightGBM parameters
        
    Returns:
        Configured LGBMClassifier instance
    """
    try:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**config)
    except ImportError:
        raise ImportError("LightGBM not installed. Install with: pip install lightgbm")


def train_model(model: Any, x_train: np.ndarray, y_train: np.ndarray) -> Any:
    """
    Train a model on the provided training data.
    
    Args:
        model: Scikit-learn compatible model instance
        x_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    average: str = 'weighted'
) -> Dict[str, Any]:
    """
    Evaluate model performance on train and test sets.
    
    Args:
        model: Trained model
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        average: Averaging method for multi-class metrics
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Predictions
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # Probabilities for ROC-AUC
    try:
        y_train_proba = model.predict_proba(x_train)
        y_test_proba = model.predict_proba(x_test)
        
        # Calculate ROC-AUC
        if len(np.unique(y_test)) == 2:
            train_auc = roc_auc_score(y_train, y_train_proba[:, 1])
            test_auc = roc_auc_score(y_test, y_test_proba[:, 1])
        else:
            train_auc = roc_auc_score(y_train, y_train_proba, multi_class='ovr', average=average)
            test_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average=average)
        
        train_logloss = log_loss(y_train, y_train_proba)
        test_logloss = log_loss(y_test, y_test_proba)
    except (AttributeError, ValueError):
        train_auc = test_auc = None
        train_logloss = test_logloss = None
    
    # Calculate metrics
    results = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, average=average, zero_division=0),
            'recall': recall_score(y_train, y_train_pred, average=average, zero_division=0),
            'f1': f1_score(y_train, y_train_pred, average=average, zero_division=0),
            'auc': train_auc,
            'logloss': train_logloss
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, average=average, zero_division=0),
            'recall': recall_score(y_test, y_test_pred, average=average, zero_division=0),
            'f1': f1_score(y_test, y_test_pred, average=average, zero_division=0),
            'auc': test_auc,
            'logloss': test_logloss
        },
        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
        'classification_report': classification_report(y_test, y_test_pred, zero_division=0)
    }
    
    return results


def compare_boosting_algorithms(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    configs: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple boosting algorithms.
    
    Args:
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        configs: Dictionary mapping algorithm names to their configs
        
    Returns:
        Dictionary containing results for all algorithms
    """
    results = {}
    
    # AdaBoost
    if 'adaboost' in configs:
        print("Training AdaBoost...")
        ada_model = create_adaboost_model(configs['adaboost'])
        train_model(ada_model, x_train, y_train)
        results['adaboost'] = {
            'model': ada_model,
            'metrics': evaluate_model(ada_model, x_train, y_train, x_test, y_test)
        }
    
    # Gradient Boosting
    if 'gradient_boosting' in configs:
        print("Training Gradient Boosting...")
        gb_model = create_gradient_boosting_model(configs['gradient_boosting'])
        train_model(gb_model, x_train, y_train)
        results['gradient_boosting'] = {
            'model': gb_model,
            'metrics': evaluate_model(gb_model, x_train, y_train, x_test, y_test)
        }
    
    # XGBoost
    if 'xgboost' in configs:
        try:
            print("Training XGBoost...")
            xgb_model = create_xgboost_model(configs['xgboost'])
            train_model(xgb_model, x_train, y_train)
            results['xgboost'] = {
                'model': xgb_model,
                'metrics': evaluate_model(xgb_model, x_train, y_train, x_test, y_test)
            }
        except ImportError as e:
            print(f"Skipping XGBoost: {e}")
    
    # LightGBM
    if 'lightgbm' in configs:
        try:
            print("Training LightGBM...")
            lgb_model = create_lightgbm_model(configs['lightgbm'])
            train_model(lgb_model, x_train, y_train)
            results['lightgbm'] = {
                'model': lgb_model,
                'metrics': evaluate_model(lgb_model, x_train, y_train, x_test, y_test)
            }
        except ImportError as e:
            print(f"Skipping LightGBM: {e}")
    
    return results


def get_feature_importance(
    model: Any,
    feature_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: Names of features
        
    Returns:
        Dictionary mapping feature names to importance scores
    """
    if not hasattr(model, 'feature_importances_'):
        raise AttributeError("Model does not have feature_importances_ attribute")
    
    importances = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    
    return dict(zip(feature_names, importances))
