"""
Complex Baseline Models for Comparison

This module provides Random Forest and Gradient Boosting classifiers as
strong baseline models to compare against the scaling ensemble.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, make_scorer
from typing import Dict, Tuple, Union, Optional
import time
import warnings

warnings.filterwarnings('ignore')


class ComplexBaseline:
    """
    Wrapper for complex baseline models with hyperparameter tuning.
    
    Parameters
    ----------
    model_type : str
        Type of model: 'random_forest' or 'gradient_boosting'
    param_grid : dict, optional
        Custom parameter grid for GridSearchCV. If None, uses defaults.
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='roc_auc'
        Scoring metric for hyperparameter selection
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all processors)
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print training progress
        
    Attributes
    ----------
    best_model_ : estimator
        Best model found by GridSearchCV
    best_params_ : dict
        Best hyperparameters
    cv_results_ : dict
        Cross-validation results
    training_time_ : float
        Time taken to train the model
    """
    
    def __init__(
        self,
        model_type: str = 'random_forest',
        param_grid: Optional[Dict] = None,
        cv: int = 5,
        scoring: str = 'roc_auc',
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: bool = False
    ):
        self.model_type = model_type.lower()
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
        # Validate model type
        if self.model_type not in ['random_forest', 'gradient_boosting']:
            raise ValueError(f"model_type must be 'random_forest' or 'gradient_boosting', got '{model_type}'")
        
    def _get_default_param_grid(self) -> Dict:
        """Get default parameter grid based on model type."""
        if self.model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        else:  # gradient_boosting
            return {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [2, 5],
                'subsample': [0.8, 1.0]
            }
    
    def _get_base_estimator(self):
        """Get base estimator based on model type."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        else:  # gradient_boosting
            return GradientBoostingClassifier(
                random_state=self.random_state
            )
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit the model with hyperparameter tuning via GridSearchCV.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target labels
            
        Returns
        -------
        self : object
            Fitted model
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Get parameter grid
        param_grid = self.param_grid if self.param_grid is not None else self._get_default_param_grid()
        
        # Get base estimator
        base_estimator = self._get_base_estimator()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training {self.model_type.replace('_', ' ').title()}")
            print(f"{'='*60}")
            print(f"Parameter grid: {param_grid}")
            print(f"CV folds: {self.cv}")
            print(f"Scoring: {self.scoring}\n")
        
        # Perform grid search
        start_time = time.time()
        
        grid_search = GridSearchCV(
            estimator=base_estimator,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1 if self.verbose else 0,
            refit=True
        )
        
        grid_search.fit(X, y)
        
        self.training_time_ = time.time() - start_time
        self.best_model_ = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        self.cv_results_ = grid_search.cv_results_
        self.best_score_ = grid_search.best_score_
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training completed in {self.training_time_:.3f}s")
            print(f"Best parameters: {self.best_params_}")
            print(f"Best CV score: {self.best_score_:.4f}")
            print(f"{'='*60}\n")
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted class labels
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.best_model_.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each class
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.best_model_.predict_proba(X)
    
    def score(self, X: Union[np.ndarray, pd.DataFrame], 
              y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Compute performance metrics on test data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features
        y : array-like of shape (n_samples,)
            True labels
            
        Returns
        -------
        scores : dict
            Dictionary containing accuracy and AUC-ROC scores
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_proba),
            'pr_auc': average_precision_score(y, y_proba)
        }
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance from the trained model.
        
        Returns
        -------
        importance : np.ndarray
            Feature importance scores
        """
        return self.best_model_.feature_importances_


def train_random_forest(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = False
) -> ComplexBaseline:
    """
    Convenience function to train a Random Forest classifier.
    
    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training features
    y_train : array-like of shape (n_samples,)
        Training labels
    param_grid : dict, optional
        Custom parameter grid
    cv : int, default=5
        Number of CV folds
    random_state : int, default=42
        Random seed
    verbose : bool, default=False
        Print progress
        
    Returns
    -------
    model : ComplexBaseline
        Trained Random Forest model
    """
    model = ComplexBaseline(
        model_type='random_forest',
        param_grid=param_grid,
        cv=cv,
        random_state=random_state,
        verbose=verbose
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = False
) -> ComplexBaseline:
    """
    Convenience function to train a Gradient Boosting classifier.
    
    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training features
    y_train : array-like of shape (n_samples,)
        Training labels
    param_grid : dict, optional
        Custom parameter grid
    cv : int, default=5
        Number of CV folds
    random_state : int, default=42
        Random seed
    verbose : bool, default=False
        Print progress
        
    Returns
    -------
    model : ComplexBaseline
        Trained Gradient Boosting model
    """
    model = ComplexBaseline(
        model_type='gradient_boosting',
        param_grid=param_grid,
        cv=cv,
        random_state=random_state,
        verbose=verbose
    )
    model.fit(X_train, y_train)
    return model


def compare_baselines(
    X_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict[str, Dict]:
    """
    Train and compare both Random Forest and Gradient Boosting models.
    
    Parameters
    ----------
    X_train : array-like of shape (n_samples, n_features)
        Training features
    y_train : array-like of shape (n_samples,)
        Training labels
    X_test : array-like of shape (n_samples, n_features)
        Test features
    y_test : array-like of shape (n_samples,)
        Test labels
    cv : int, default=5
        Number of CV folds
    random_state : int, default=42
        Random seed
    verbose : bool, default=True
        Print progress
        
    Returns
    -------
    results : dict
        Dictionary containing models and their scores
    """
    results = {}
    
    # Train Random Forest
    if verbose:
        print("\n" + "="*60)
        print("Training Random Forest Baseline")
        print("="*60)
    
    rf_model = train_random_forest(
        X_train, y_train,
        cv=cv,
        random_state=random_state,
        verbose=verbose
    )
    rf_scores = rf_model.score(X_test, y_test)
    
    results['random_forest'] = {
        'model': rf_model,
        'best_params': rf_model.best_params_,
        'cv_score': rf_model.best_score_,
        'test_scores': rf_scores,
        'training_time': rf_model.training_time_
    }
    
    # Train Gradient Boosting
    if verbose:
        print("\n" + "="*60)
        print("Training Gradient Boosting Baseline")
        print("="*60)
    
    gb_model = train_gradient_boosting(
        X_train, y_train,
        cv=cv,
        random_state=random_state,
        verbose=verbose
    )
    gb_scores = gb_model.score(X_test, y_test)
    
    results['gradient_boosting'] = {
        'model': gb_model,
        'best_params': gb_model.best_params_,
        'cv_score': gb_model.best_score_,
        'test_scores': gb_scores,
        'training_time': gb_model.training_time_
    }
    
    # Print comparison
    if verbose:
        print("\n" + "="*60)
        print("BASELINE COMPARISON RESULTS")
        print("="*60)
        
        for model_name, result in results.items():
            print(f"\n{model_name.replace('_', ' ').title()}:")
            print(f"  Best params: {result['best_params']}")
            print(f"  CV Score: {result['cv_score']:.4f}")
            print(f"  Test Accuracy: {result['test_scores']['accuracy']:.4f}")
            print(f"  Test AUC-ROC: {result['test_scores']['auc_roc']:.4f}")
            print(f"  Training Time: {result['training_time']:.3f}s")
        
        print("\n" + "="*60)
    
    return results


def benchmark_inference_time(
    models: Union[Dict[str, object], object],
    X_test: Union[np.ndarray, pd.DataFrame],
    n_samples: int = 10000,
    n_iterations: int = 10,
    verbose: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark inference time for one or more models.
    
    Parameters
    ----------
    models : dict or model
        Either a dictionary mapping model names to trained models,
        or a single model to benchmark
    X_test : array-like
        Test data
    n_samples : int, default=10000
        Number of samples to use for benchmarking
    n_iterations : int, default=10
        Number of iterations to average
    verbose : bool, default=False
        Whether to print progress
        
    Returns
    -------
    results : dict
        Dictionary mapping model names to timing statistics.
        If a single model was passed, returns dict with 'model' key.
    """
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    
    # Handle single model case
    if not isinstance(models, dict):
        models = {'model': models}
    
    # Sample data if needed
    if len(X_test) < n_samples:
        # Repeat samples to reach n_samples
        n_repeats = (n_samples // len(X_test)) + 1
        X_benchmark = np.tile(X_test, (n_repeats, 1))[:n_samples]
    else:
        X_benchmark = X_test[:n_samples]
    
    results = {}
    
    for name, model in models.items():
        if verbose:
            print(f"Benchmarking {name}...")
        
        # Warmup
        _ = model.predict(X_benchmark[:100])
        
        # Benchmark
        times = []
        for _ in range(n_iterations):
            start = time.time()
            _ = model.predict(X_benchmark)
            elapsed = time.time() - start
            times.append(elapsed)
        
        times = np.array(times)
        predictions_per_second = n_samples / times.mean()
        
        results[name] = {
            'mean': times.mean(),
            'std': times.std(),
            'min': times.min(),
            'max': times.max(),
            'predictions_per_second': predictions_per_second,
            'n_samples': n_samples
        }
        
        if verbose:
            print(f"  Mean: {times.mean()*1000:.2f}ms, Std: {times.std()*1000:.2f}ms")
    
    return results


if __name__ == "__main__":
    # Test the complex baselines with synthetic data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("Testing Complex Baselines with synthetic data...")
    print("=" * 60)
    
    # Generate synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Compare both baselines
    results = compare_baselines(
        X_train, y_train,
        X_test, y_test,
        cv=3,  # Use 3 folds for faster testing
        verbose=True
    )
    
    # Benchmark inference time
    print("\n" + "="*60)
    print("INFERENCE TIME BENCHMARKS")
    print("="*60)
    
    models_to_benchmark = {name.replace('_', ' ').title(): result['model'] 
                           for name, result in results.items()}
    timing_results = benchmark_inference_time(
        models_to_benchmark,
        X_test,
        n_samples=1000,
        n_iterations=5,
        verbose=True
    )
    
    for model_name, timing in timing_results.items():
        print(f"\n{model_name}:")
        print(f"  Mean time: {timing['mean']*1000:.2f}ms")
        print(f"  Std time: {timing['std']*1000:.2f}ms")
        print(f"  Predictions/second: {timing['predictions_per_second']:.0f}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
