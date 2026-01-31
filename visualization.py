"""
Visualization functions for boosting algorithms.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Any, List, Optional
import os


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: Names of classes
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_algorithm_comparison(
    comparison_results: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison between different boosting algorithms.
    
    Args:
        comparison_results: Results dictionary from compare_boosting_algorithms
        save_path: Path to save the plot
    """
    algorithms = list(comparison_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Extract test scores
    scores = {metric: [] for metric in metrics}
    for algo in algorithms:
        for metric in metrics:
            scores[metric].append(comparison_results[algo]['metrics']['test'][metric])
    
    # Create subplot for each metric
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(algorithms))
        bars = ax.bar(x, scores[metric], alpha=0.8, color=sns.color_palette("husl", len(algorithms)))
        
        ax.set_xlabel('Algorithm', fontsize=11, fontweight='bold')
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric.capitalize()} Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace('_', ' ').title() for a in algorithms], rotation=15, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Boosting Algorithms Performance Comparison', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_feature_importance(
    importance_dict: Dict[str, float],
    title: str = "Feature Importance",
    top_n: int = 20,
    save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance as horizontal bar chart.
    
    Args:
        importance_dict: Dictionary mapping feature names to importance scores
        title: Plot title
        top_n: Number of top features to display
        save_path: Path to save the plot
    """
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, importances = zip(*sorted_features)
    
    plt.figure(figsize=(10, max(6, len(features) * 0.3)))
    y_pos = np.arange(len(features))
    
    plt.barh(y_pos, importances, alpha=0.8, color=sns.color_palette("viridis", len(features)))
    plt.yticks(y_pos, features)
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_learning_curve(
    n_estimators_range: List[int],
    train_scores: List[float],
    test_scores: List[float],
    title: str = "Learning Curve",
    save_path: Optional[str] = None
) -> None:
    """
    Plot learning curve showing performance vs number of estimators.
    
    Args:
        n_estimators_range: List of n_estimators values
        train_scores: Training scores for each n_estimators
        test_scores: Test scores for each n_estimators
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(n_estimators_range, train_scores, marker='o', label='Train Score', linewidth=2)
    plt.plot(n_estimators_range, test_scores, marker='s', label='Test Score', linewidth=2)
    plt.xlabel('Number of Estimators', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy Score', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_learning_rate_effect(
    learning_rates: List[float],
    test_scores: List[float],
    title: str = "Effect of Learning Rate",
    save_path: Optional[str] = None
) -> None:
    """
    Plot the effect of learning rate on model performance.
    
    Args:
        learning_rates: List of learning rate values
        test_scores: Test scores for each learning rate
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, test_scores, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Learning Rate', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def plot_multiple_feature_importances(
    importances_dict: Dict[str, Dict[str, float]],
    top_n: int = 15,
    save_path: Optional[str] = None
) -> None:
    """
    Compare feature importances across multiple algorithms.
    
    Args:
        importances_dict: Dictionary mapping algorithm names to importance dictionaries
        top_n: Number of top features to display
        save_path: Path to save the plot
    """
    # Get union of top features from all algorithms
    all_features = set()
    for imp_dict in importances_dict.values():
        sorted_features = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        all_features.update([f[0] for f in sorted_features])
    
    # Create comparison data
    algorithms = list(importances_dict.keys())
    features = list(all_features)[:top_n]
    
    # Build matrix
    importance_matrix = np.zeros((len(features), len(algorithms)))
    for i, feature in enumerate(features):
        for j, algo in enumerate(algorithms):
            importance_matrix[i, j] = importances_dict[algo].get(feature, 0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(features) * 0.4)))
    x = np.arange(len(features))
    width = 0.8 / len(algorithms)
    
    for i, algo in enumerate(algorithms):
        offset = (i - len(algorithms)/2 + 0.5) * width
        ax.barh(x + offset, importance_matrix[:, i], width, 
                label=algo.replace('_', ' ').title(), alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance Comparison Across Algorithms', fontsize=16, fontweight='bold')
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def create_plots_directory(directory: str = 'plots') -> None:
    """
    Create directory for saving plots if it doesn't exist.
    
    Args:
        directory: Directory name
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
