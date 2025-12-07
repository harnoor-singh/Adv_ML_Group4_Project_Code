"""
Baseline model implementations using different feature scalers.

This module implements single-scaler baseline models for binary classification
using L2-regularized logistic regression with different feature scaling strategies.

Student 1 implementation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, average_precision_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats
import time
import warnings
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings('ignore')


class BaselineModel:
    """
    Single-scaler baseline model using LogisticRegressionCV.
    
    Combines a feature scaler with L2-regularized logistic regression,
    using cross-validation to select the optimal regularization parameter.
    """
    
    def __init__(self, 
                 scaler_name: str = 'standard',
                 C_range: List[float] = None,
                 cv_folds: int = 5,
                 random_state: int = 42,
                 max_iter: int = 1000,
                 verbose: bool = False):
        """
        Initialize the baseline model.
        
        Args:
            scaler_name: Name of scaler ('standard', 'minmax', 'robust', 'maxabs')
            C_range: Range of regularization parameters to search
            cv_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            max_iter: Maximum iterations for optimization
            verbose: Whether to print training progress
        """
        self.scaler_name = scaler_name.lower()
        self.C_range = C_range if C_range else [0.001, 0.01, 0.1, 1, 10, 100]
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose
        
        # Initialize scaler
        self.scaler = self._get_scaler()
        
        # Initialize model (will be fitted during training)
        self.model = None
        
        # Training metrics
        self.training_time = None
        self.best_C = None
        self.cv_scores = None
        
    def _get_scaler(self):
        """Get the appropriate scaler based on scaler_name."""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler(),
            'maxabs': MaxAbsScaler(),
        }
        
        if self.scaler_name not in scalers:
            raise ValueError(f"Unknown scaler: {self.scaler_name}. "
                           f"Choose from {list(scalers.keys())}")
        
        return scalers[self.scaler_name]
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'BaselineModel':
        """
        Fit the model with cross-validated regularization parameter selection.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            self
        """
        if self.verbose:
            print(f"\nTraining {self.scaler_name.upper()} scaler model...")
            print(f"  C range: {self.C_range}")
            print(f"  CV folds: {self.cv_folds}")
        
        start_time = time.time()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train LogisticRegressionCV
        self.model = LogisticRegressionCV(
            Cs=self.C_range,
            cv=self.cv_folds,
            scoring='roc_auc',
            penalty='l2',
            solver='lbfgs',
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        self.training_time = time.time() - start_time
        self.best_C = self.model.C_[0]
        self.cv_scores = self.model.scores_[1].mean(axis=0)  # Mean CV scores for positive class
        
        if self.verbose:
            print(f"  Training time: {self.training_time:.3f}s")
            print(f"  Best C: {self.best_C:.4f}")
            print(f"  Best CV AUC: {self.cv_scores.max():.4f}")
        
        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)
    
    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted probabilities (n_samples, 2)
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict_proba(X_test_scaled)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'scaler': self.scaler_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': average_precision_score(y_test, y_pred_proba),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'best_C': self.best_C,
            'training_time': self.training_time,
        }
        
        return metrics


class BaselineComparison:
    """
    Compare multiple baseline models with different scalers.
    """
    
    def __init__(self,
                 scaler_names: List[str] = None,
                 C_range: List[float] = None,
                 cv_folds: int = 5,
                 random_state: int = 42,
                 max_iter: int = 1000):
        """
        Initialize baseline comparison.
        
        Args:
            scaler_names: List of scaler names to compare
            C_range: Range of regularization parameters
            cv_folds: Number of CV folds
            random_state: Random seed
            max_iter: Maximum iterations
        """
        self.scaler_names = scaler_names if scaler_names else ['standard', 'minmax', 'robust', 'maxabs']
        self.C_range = C_range if C_range else [0.001, 0.01, 0.1, 1, 10, 100]
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.max_iter = max_iter
        
        self.models = {}
        self.results = None
        
    def fit_all(self, X_train: np.ndarray, y_train: np.ndarray, 
                verbose: bool = True) -> Dict[str, BaselineModel]:
        """
        Fit all baseline models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            verbose: Whether to print progress
            
        Returns:
            Dictionary of fitted models
        """
        if verbose:
            print("\n" + "="*60)
            print("TRAINING BASELINE MODELS")
            print("="*60)
        
        for scaler_name in self.scaler_names:
            model = BaselineModel(
                scaler_name=scaler_name,
                C_range=self.C_range,
                cv_folds=self.cv_folds,
                random_state=self.random_state,
                max_iter=self.max_iter,
                verbose=verbose
            )
            
            model.fit(X_train, y_train)
            self.models[scaler_name] = model
        
        if verbose:
            print("\n✓ All baseline models trained!")
        
        return self.models
    
    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray,
                    verbose: bool = True) -> pd.DataFrame:
        """
        Evaluate all models on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Whether to print results
            
        Returns:
            DataFrame with evaluation results
        """
        if not self.models:
            raise ValueError("Models not fitted. Call fit_all() first.")
        
        results = []
        
        for scaler_name, model in self.models.items():
            metrics = model.evaluate(X_test, y_test)
            results.append(metrics)
        
        self.results = pd.DataFrame(results)
        
        if verbose:
            print("\n" + "="*60)
            print("BASELINE MODEL RESULTS")
            print("="*60)
            print(self.results.to_string(index=False))
            print("\nBest model by AUC-ROC:")
            best_idx = self.results['auc_roc'].idxmax()
            print(f"  {self.results.loc[best_idx, 'scaler'].upper()}: "
                  f"{self.results.loc[best_idx, 'auc_roc']:.4f}")
        
        return self.results
    
    def plot_performance_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Plot performance comparison across scalers.
        
        Args:
            save_path: Path to save the figure (optional)
        """
        if self.results is None:
            raise ValueError("No results to plot. Call evaluate_all() first.")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # AUC-ROC comparison
        ax1 = axes[0]
        scalers = self.results['scaler'].values
        aucs = self.results['auc_roc'].values
        
        colors = plt.cm.Set2(range(len(scalers)))
        bars1 = ax1.bar(scalers, aucs, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('AUC-ROC', fontsize=12)
        ax1.set_xlabel('Scaler', fontsize=12)
        ax1.set_title('Test AUC-ROC by Scaler', fontsize=13, fontweight='bold')
        ax1.set_ylim([max(0, aucs.min() - 0.05), min(1, aucs.max() + 0.05)])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        # Accuracy comparison
        ax2 = axes[1]
        accuracies = self.results['accuracy'].values
        
        bars2 = ax2.bar(scalers, accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_xlabel('Scaler', fontsize=12)
        ax2.set_title('Test Accuracy by Scaler', fontsize=13, fontweight='bold')
        ax2.set_ylim([max(0, accuracies.min() - 0.05), min(1, accuracies.max() + 0.05)])
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison saved to {save_path}")
        
        plt.show()
    
    def plot_training_time_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Plot training time comparison.
        
        Args:
            save_path: Path to save the figure (optional)
        """
        if self.results is None:
            raise ValueError("No results to plot. Call evaluate_all() first.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scalers = self.results['scaler'].values
        times = self.results['training_time'].values
        
        colors = plt.cm.Set3(range(len(scalers)))
        bars = ax.bar(scalers, times, color=colors, alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Training Time (seconds)', fontsize=12)
        ax.set_xlabel('Scaler', fontsize=12)
        ax.set_title('Training Time Comparison', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}s',
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training time comparison saved to {save_path}")
        
        plt.show()
    
    def statistical_comparison(self, X: np.ndarray, y: np.ndarray,
                               n_iterations: int = 10) -> pd.DataFrame:
        """
        Perform statistical comparison using repeated cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            n_iterations: Number of CV iterations
            
        Returns:
            DataFrame with statistical test results
        """
        print("\n" + "="*60)
        print("STATISTICAL COMPARISON (Repeated Cross-Validation)")
        print("="*60)
        print(f"Running {n_iterations} iterations of {self.cv_folds}-fold CV...")
        
        # Store scores for each scaler
        scaler_scores = {scaler: [] for scaler in self.scaler_names}
        
        for iteration in range(n_iterations):
            seed = self.random_state + iteration
            
            for scaler_name in self.scaler_names:
                # Create fresh model
                model = BaselineModel(
                    scaler_name=scaler_name,
                    C_range=self.C_range,
                    cv_folds=self.cv_folds,
                    random_state=seed,
                    max_iter=self.max_iter,
                    verbose=False
                )
                
                # Get scaler
                scaler = model._get_scaler()
                
                # Scale data
                X_scaled = scaler.fit_transform(X)
                
                # Cross-validation
                lr = LogisticRegressionCV(
                    Cs=self.C_range,
                    cv=self.cv_folds,
                    scoring='roc_auc',
                    penalty='l2',
                    solver='lbfgs',
                    max_iter=self.max_iter,
                    random_state=seed,
                    n_jobs=-1
                )
                
                scores = cross_val_score(lr, X_scaled, y, cv=self.cv_folds, 
                                        scoring='roc_auc', n_jobs=-1)
                scaler_scores[scaler_name].extend(scores)
        
        # Compute statistics
        stats_data = []
        for scaler_name in self.scaler_names:
            scores = np.array(scaler_scores[scaler_name])
            stats_data.append({
                'scaler': scaler_name,
                'mean_auc': scores.mean(),
                'std_auc': scores.std(),
                'min_auc': scores.min(),
                'max_auc': scores.max(),
                'median_auc': np.median(scores),
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        print("\nCross-Validation Statistics:")
        print(stats_df.to_string(index=False))
        
        # Pairwise t-tests
        print("\nPairwise T-Tests (p-values):")
        print("-" * 60)
        
        for i, scaler1 in enumerate(self.scaler_names):
            for scaler2 in self.scaler_names[i+1:]:
                scores1 = scaler_scores[scaler1]
                scores2 = scaler_scores[scaler2]
                
                t_stat, p_value = stats.ttest_rel(scores1, scores2)
                
                significance = ""
                if p_value < 0.001:
                    significance = "***"
                elif p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"
                
                print(f"  {scaler1:8s} vs {scaler2:8s}: p={p_value:.4f} {significance}")
        
        print("\n  * p < 0.05, ** p < 0.01, *** p < 0.001")
        
        return stats_df


def train_and_evaluate_baseline(X_train: np.ndarray, 
                                y_train: np.ndarray,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                dataset_name: str = "Dataset",
                                save_dir: Optional[str] = None,
                                generate_plots: bool = True,
                                run_statistical_tests: bool = False) -> Tuple[BaselineComparison, pd.DataFrame]:
    """
    Convenience function to train and evaluate all baseline models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        dataset_name: Name of the dataset
        save_dir: Directory to save results
        generate_plots: Whether to generate plots
        run_statistical_tests: Whether to run statistical comparison
        
    Returns:
        Tuple of (BaselineComparison object, results DataFrame)
    """
    comparison = BaselineComparison()
    
    # Train all models
    comparison.fit_all(X_train, y_train, verbose=True)
    
    # Evaluate on test set
    results = comparison.evaluate_all(X_test, y_test, verbose=True)
    
    # Save results
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        results_path = os.path.join(save_dir, f"{dataset_name}_baseline_results.csv")
        results.to_csv(results_path, index=False)
        print(f"\n✓ Results saved to {results_path}")
    
    # Generate plots
    if generate_plots:
        if save_dir:
            perf_path = os.path.join(save_dir, f"{dataset_name}_performance_comparison.png")
            time_path = os.path.join(save_dir, f"{dataset_name}_training_time.png")
        else:
            perf_path = None
            time_path = None
        
        comparison.plot_performance_comparison(save_path=perf_path)
        comparison.plot_training_time_comparison(save_path=time_path)
    
    # Statistical tests
    if run_statistical_tests:
        # Combine train and test for CV
        X_full = np.vstack([X_train, X_test])
        y_full = np.hstack([y_train, y_test])
        
        stats_df = comparison.statistical_comparison(X_full, y_full, n_iterations=10)
        
        if save_dir:
            stats_path = os.path.join(save_dir, f"{dataset_name}_statistical_comparison.csv")
            stats_df.to_csv(stats_path, index=False)
            print(f"\n✓ Statistical comparison saved to {stats_path}")
    
    return comparison, results


if __name__ == "__main__":
    """
    Train and evaluate baseline models on all datasets.
    
    Usage:
        python src/baseline_models.py [--data-dir DATA_DIR] [--results-dir RESULTS_DIR] 
                                      [--no-plots] [--run-stats]
    """
    import argparse
    import os
    import sys
    
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.data_loader import (
        load_wine_quality,
        preprocess_data,
        split_data
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train and evaluate baseline models on all datasets'
    )
    parser.add_argument('--data-dir', type=str, default='../data',
                       help='Directory containing dataset files')
    parser.add_argument('--results-dir', type=str, default='../results/student1',
                       help='Directory to save results')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    parser.add_argument('--run-stats', action='store_true',
                       help='Run statistical comparison tests')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("BASELINE MODELS - WINE QUALITY DATASET")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Test size: {args.test_size}")
    print(f"Random state: {args.random_state}")
    print(f"Generate plots: {not args.no_plots}")
    print(f"Run statistical tests: {args.run_stats}")
    
    # Store all results
    all_results = []
    
    # Wine Quality Dataset
    print("\n" + "="*80)
    print("TRAINING BASELINE MODELS")
    print("="*80)
    try:
        X_wine, y_wine = load_wine_quality(args.data_dir)
        X_wine, y_wine = preprocess_data(X_wine, y_wine)
        X_train, X_test, y_train, y_test = split_data(
            X_wine, y_wine, 
            test_size=args.test_size, 
            random_state=args.random_state
        )
        
        comparison, results = train_and_evaluate_baseline(
            X_train, y_train, X_test, y_test,
            dataset_name="wine_quality",
            save_dir=args.results_dir,
            generate_plots=not args.no_plots,
            run_statistical_tests=args.run_stats
        )
        
        results['dataset'] = 'Wine Quality'
        all_results.append(results)
        
        print("\n✓ Wine Quality baseline training complete!")
        
    except Exception as e:
        print(f"\n✗ Error with Wine Quality dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Display results
    if all_results:
        print("\n" + "="*80)
        print("BASELINE RESULTS SUMMARY")
        print("="*80)
        
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Reorder columns
        cols = ['dataset', 'scaler', 'auc_roc', 'accuracy', 'precision', 
                'recall', 'f1', 'best_C', 'training_time']
        combined_df = combined_df[cols]
        
        print(combined_df.to_string(index=False))
        
        # Save combined results
        combined_path = os.path.join(args.results_dir, "all_baseline_results.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"\n✓ Results saved to {combined_path}")
        
        # Best scaler
        print("\n" + "-"*80)
        print("BEST SCALER (by AUC-ROC)")
        print("-"*80)
        best_idx = combined_df['auc_roc'].idxmax()
        best_row = combined_df.loc[best_idx]
        print(f"  Scaler: {best_row['scaler'].upper()}")
        print(f"  AUC-ROC: {best_row['auc_roc']:.4f}")
        print(f"  Accuracy: {best_row['accuracy']:.4f}")
        print(f"  Best C: {best_row['best_C']:.4f}")
    
    print("\n" + "="*80)
    print("BASELINE TRAINING COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {args.results_dir}")
    print("\nFiles generated:")
    print("  - wine_quality_baseline_results.csv (results)")
    print("  - all_baseline_results.csv (summary)")
    if args.run_stats:
        print("  - wine_quality_statistical_comparison.csv (statistical tests)")
    if not args.no_plots:
        print("  - wine_quality_performance_comparison.png (AUC & accuracy plots)")
        print("  - wine_quality_training_time.png (training time comparison)")
    print("\n")
