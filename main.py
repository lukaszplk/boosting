"""
Main script for boosting ensemble classifiers with CLI interface.
"""
import argparse
import numpy as np
from data_loader import load_data, get_dataset_info
from models import (
    create_adaboost_model,
    create_gradient_boosting_model,
    create_xgboost_model,
    create_lightgbm_model,
    train_model,
    evaluate_model,
    compare_boosting_algorithms,
    get_feature_importance
)
from visualization import (
    plot_confusion_matrix,
    plot_algorithm_comparison,
    plot_feature_importance,
    plot_learning_curve,
    plot_multiple_feature_importances,
    create_plots_directory
)
import config


def print_separator(char='=', length=80):
    """Print a separator line."""
    print(char * length)


def print_results(results: dict, model_name: str):
    """Print formatted model evaluation results."""
    print_separator()
    print(f"{model_name.upper()} RESULTS")
    print_separator()
    
    print("\nTraining Set Performance:")
    print(f"  Accuracy:  {results['train']['accuracy']:.4f}")
    print(f"  Precision: {results['train']['precision']:.4f}")
    print(f"  Recall:    {results['train']['recall']:.4f}")
    print(f"  F1-Score:  {results['train']['f1']:.4f}")
    if results['train']['auc'] is not None:
        print(f"  ROC-AUC:   {results['train']['auc']:.4f}")
    
    print("\nTest Set Performance:")
    print(f"  Accuracy:  {results['test']['accuracy']:.4f}")
    print(f"  Precision: {results['test']['precision']:.4f}")
    print(f"  Recall:    {results['test']['recall']:.4f}")
    print(f"  F1-Score:  {results['test']['f1']:.4f}")
    if results['test']['auc'] is not None:
        print(f"  ROC-AUC:   {results['test']['auc']:.4f}")
    
    print("\nDetailed Classification Report:")
    print(results['classification_report'])


def main(args):
    """Main execution function."""
    print_separator('=', 80)
    print("BOOSTING ENSEMBLE LEARNING")
    print_separator('=', 80)
    
    # Display dataset information
    if args.show_info:
        dataset_info = get_dataset_info(args.dataset)
        print("\nDataset Information:")
        print(f"  Name: {dataset_info['name']}")
        print(f"  Samples: {dataset_info['n_samples']}")
        print(f"  Features: {dataset_info['n_features']}")
        print(f"  Classes: {dataset_info['n_classes']}")
        print()
    
    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    x_train, x_test, y_train, y_test = load_data(
        dataset=args.dataset,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Features: {x_train.shape[1]}")
    
    dataset_info = get_dataset_info(args.dataset)
    
    if args.mode == 'single':
        # Train single algorithm
        print(f"\n[Mode: Single Algorithm - {args.algorithm.upper()}]")
        
        if args.algorithm == 'adaboost':
            model = create_adaboost_model(config.ADABOOST_CONFIG)
        elif args.algorithm == 'gradient_boosting':
            model = create_gradient_boosting_model(config.GRADIENT_BOOSTING_CONFIG)
        elif args.algorithm == 'xgboost':
            model = create_xgboost_model(config.XGBOOST_CONFIG)
        elif args.algorithm == 'lightgbm':
            model = create_lightgbm_model(config.LIGHTGBM_CONFIG)
        else:
            print(f"Unknown algorithm: {args.algorithm}")
            return
        
        print(f"\nTraining {args.algorithm}...")
        train_model(model, x_train, y_train)
        
        print("\nEvaluating model...")
        results = evaluate_model(model, x_train, y_train, x_test, y_test)
        print_results(results, args.algorithm)
        
        if args.visualize:
            create_plots_directory(config.PLOTS_DIR)
            
            # Confusion matrix
            plot_confusion_matrix(
                results['confusion_matrix'],
                dataset_info['target_names'],
                title=f"{args.algorithm.replace('_', ' ').title()} - Confusion Matrix",
                save_path=f"{config.PLOTS_DIR}/{args.algorithm}_confusion_matrix.png" if config.SAVE_PLOTS else None
            )
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = get_feature_importance(model, dataset_info['feature_names'])
                plot_feature_importance(
                    importance,
                    title=f"{args.algorithm.replace('_', ' ').title()} - Feature Importance",
                    save_path=f"{config.PLOTS_DIR}/{args.algorithm}_feature_importance.png" if config.SAVE_PLOTS else None
                )
    
    elif args.mode == 'compare':
        # Compare all algorithms
        print("\n[Mode: Algorithm Comparison]")
        print("\nComparing AdaBoost, Gradient Boosting, XGBoost, and LightGBM...")
        
        configs = {
            'adaboost': config.ADABOOST_CONFIG,
            'gradient_boosting': config.GRADIENT_BOOSTING_CONFIG,
            'xgboost': config.XGBOOST_CONFIG,
            'lightgbm': config.LIGHTGBM_CONFIG
        }
        
        comparison_results = compare_boosting_algorithms(
            x_train, y_train, x_test, y_test, configs
        )
        
        # Print results for each algorithm
        for algo_name, result in comparison_results.items():
            print_results(result['metrics'], algo_name)
        
        # Find best algorithm
        best_algo = max(
            comparison_results.items(),
            key=lambda x: x[1]['metrics']['test']['accuracy']
        )
        
        print_separator()
        print(f"Best Algorithm: {best_algo[0].replace('_', ' ').title()}")
        print(f"Test Accuracy: {best_algo[1]['metrics']['test']['accuracy']:.4f}")
        print_separator()
        
        if args.visualize:
            create_plots_directory(config.PLOTS_DIR)
            
            # Algorithm comparison
            plot_algorithm_comparison(
                comparison_results,
                save_path=f"{config.PLOTS_DIR}/algorithm_comparison.png" if config.SAVE_PLOTS else None
            )
            
            # Feature importance comparison
            importances_dict = {}
            for algo_name, result in comparison_results.items():
                if hasattr(result['model'], 'feature_importances_'):
                    importances_dict[algo_name] = get_feature_importance(
                        result['model'],
                        dataset_info['feature_names']
                    )
            
            if importances_dict:
                plot_multiple_feature_importances(
                    importances_dict,
                    save_path=f"{config.PLOTS_DIR}/feature_importance_comparison.png" if config.SAVE_PLOTS else None
                )
    
    elif args.mode == 'experiment':
        # Experiment with n_estimators
        print("\n[Mode: Parameter Experimentation]")
        print(f"\nAnalyzing effect of number of estimators for {args.algorithm}...")
        
        n_estimators_range = config.EXPERIMENTS['n_estimators_range']
        train_scores = []
        test_scores = []
        
        for n_est in n_estimators_range:
            print(f"  Training with n_estimators={n_est}...")
            
            if args.algorithm == 'adaboost':
                model_config = config.ADABOOST_CONFIG.copy()
                model_config['n_estimators'] = n_est
                model = create_adaboost_model(model_config)
            elif args.algorithm == 'gradient_boosting':
                model_config = config.GRADIENT_BOOSTING_CONFIG.copy()
                model_config['n_estimators'] = n_est
                model = create_gradient_boosting_model(model_config)
            elif args.algorithm == 'xgboost':
                model_config = config.XGBOOST_CONFIG.copy()
                model_config['n_estimators'] = n_est
                model = create_xgboost_model(model_config)
            elif args.algorithm == 'lightgbm':
                model_config = config.LIGHTGBM_CONFIG.copy()
                model_config['n_estimators'] = n_est
                model = create_lightgbm_model(model_config)
            
            train_model(model, x_train, y_train)
            train_scores.append(model.score(x_train, y_train))
            test_scores.append(model.score(x_test, y_test))
        
        if args.visualize:
            create_plots_directory(config.PLOTS_DIR)
            plot_learning_curve(
                n_estimators_range,
                train_scores,
                test_scores,
                title=f"{args.algorithm.replace('_', ' ').title()} - Learning Curve",
                save_path=f"{config.PLOTS_DIR}/{args.algorithm}_learning_curve.png" if config.SAVE_PLOTS else None
            )
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Boosting Ensemble Learning Classifiers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode single --algorithm adaboost --visualize
  python main.py --mode compare --visualize --show-info
  python main.py --mode experiment --algorithm xgboost --visualize
  python main.py --mode single --algorithm gradient_boosting --dataset wine --visualize
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['single', 'compare', 'experiment'],
        default='single',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['adaboost', 'gradient_boosting', 'xgboost', 'lightgbm'],
        default='adaboost',
        help='Algorithm to use (for single/experiment modes)'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['breast_cancer', 'wine', 'synthetic'],
        default='breast_cancer',
        help='Dataset to use'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate and display visualizations'
    )
    
    parser.add_argument(
        '--show-info',
        action='store_true',
        help='Display dataset information'
    )
    
    args = parser.parse_args()
    main(args)