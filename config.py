"""
Configuration file for boosting ensemble experiments.
"""

# Data splitting configuration
TEST_SIZE = 0.25
RANDOM_STATE = 23

# Base estimator configuration (for AdaBoost)
BASE_ESTIMATOR_CONFIG = {
    'max_depth': 2,
    'random_state': RANDOM_STATE
}

# AdaBoost configuration
ADABOOST_CONFIG = {
    'n_estimators': 50,
    'learning_rate': 1.0,
    'random_state': RANDOM_STATE
}

# Gradient Boosting configuration
GRADIENT_BOOSTING_CONFIG = {
    'n_estimators': 50,
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'random_state': RANDOM_STATE
}

# XGBoost configuration
XGBOOST_CONFIG = {
    'n_estimators': 50,
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'eval_metric': 'logloss'
}

# LightGBM configuration
LIGHTGBM_CONFIG = {
    'n_estimators': 50,
    'learning_rate': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'verbose': -1
}

# Experiment configuration
EXPERIMENTS = {
    'n_estimators_range': [5, 10, 25, 50, 100, 200],
    'learning_rate_range': [0.01, 0.05, 0.1, 0.5, 1.0],
    'max_depth_range': [1, 2, 3, 5, 7]
}

# Output configuration
SAVE_PLOTS = True
PLOTS_DIR = 'plots'
RESULTS_DIR = 'results'

# Visualization
FIGURE_SIZE = (12, 8)
DPI = 300
