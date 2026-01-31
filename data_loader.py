"""
Data loading and preprocessing module for boosting experiments.
"""
from sklearn.datasets import load_breast_cancer, load_wine, make_classification
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any
import numpy as np


def load_data(
    dataset: str = 'breast_cancer',
    test_size: float = 0.25,
    random_state: int = 23
) -> Tuple[np.ndarray, ...]:
    """
    Load a dataset and split into train/test sets.
    
    Args:
        dataset: Dataset name ('breast_cancer', 'wine', or 'synthetic')
        test_size: Proportion of dataset for test split
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (x_train, x_test, y_train, y_test)
    """
    # Load dataset
    if dataset == 'breast_cancer':
        data = load_breast_cancer()
        x, y = data.data, data.target
    elif dataset == 'wine':
        data = load_wine()
        x, y = data.data, data.target
    elif dataset == 'synthetic':
        x, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Split into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return x_train, x_test, y_train, y_test


def get_dataset_info(dataset: str = 'breast_cancer') -> Dict[str, Any]:
    """
    Get information about the dataset.
    
    Args:
        dataset: Dataset name
        
    Returns:
        Dictionary containing dataset metadata
    """
    if dataset == 'breast_cancer':
        data = load_breast_cancer()
        return {
            'name': 'Breast Cancer Wisconsin',
            'n_samples': data.data.shape[0],
            'n_features': data.data.shape[1],
            'n_classes': len(data.target_names),
            'feature_names': data.feature_names.tolist(),
            'target_names': data.target_names.tolist(),
            'description': 'Diagnostic classification task'
        }
    elif dataset == 'wine':
        data = load_wine()
        return {
            'name': 'Wine Recognition',
            'n_samples': data.data.shape[0],
            'n_features': data.data.shape[1],
            'n_classes': len(data.target_names),
            'feature_names': data.feature_names.tolist(),
            'target_names': data.target_names.tolist(),
            'description': 'Wine cultivar classification'
        }
    elif dataset == 'synthetic':
        return {
            'name': 'Synthetic Classification',
            'n_samples': 1000,
            'n_features': 20,
            'n_classes': 2,
            'feature_names': [f'feature_{i}' for i in range(20)],
            'target_names': ['class_0', 'class_1'],
            'description': 'Synthetically generated classification data'
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
