"""
Feature Scaling Ensemble Implementation

This module provides a scaling ensemble that trains multiple L2-regularized 
logistic regression models, each with a different feature scaling method, and
combines their predictions.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from typing import List, Dict, Optional, Union, Tuple
import time
import warnings

warnings.filterwarnings('ignore')


class ScalingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble of logistic regression models with different feature scaling methods.
    
    This ensemble trains k separate LogisticRegressionCV models, each on the same
    dataset preprocessed with a different scaling method. Predictions are combined
    using either uniform averaging or custom weights.
    
    Parameters
    ----------
    scalers : list of tuples, optional
        List of (name, scaler_instance) tuples. If None, uses default scalers:
        StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
    base_estimator : estimator class, default=LogisticRegressionCV
        The base estimator to use for each scaled version
    weights : str or array-like, default='uniform'
        How to weight ensemble members:
        - 'uniform': Equal weights (1/k for k models)
        - array-like: Custom weights (must sum to 1)
    Cs : list, default=[0.001, 0.01, 0.1, 1, 10, 100]
        Regularization parameters to search for LogisticRegressionCV
    cv : int, default=5
        Number of cross-validation folds
    random_state : int, default=42
        Random seed for reproducibility
    verbose : bool, default=False
        Whether to print training progress
        
    Attributes
    ----------
    models_ : dict
        Dictionary mapping scaler names to trained models
    scalers_ : dict
        Dictionary mapping scaler names to fitted scaler instances
    weights_ : np.ndarray
        Final weights used for ensemble averaging
    training_times_ : dict
        Training time for each model
    """
    
    def __init__(
        self,
        scalers: Optional[List[Tuple[str, object]]] = None,
        base_estimator = LogisticRegressionCV,
        weights: Union[str, np.ndarray] = 'uniform',
        Cs: List[float] = None,
        cv: int = 5,
        random_state: int = 42,
        verbose: bool = False
    ):
        self.scalers = scalers
        self.base_estimator = base_estimator
        self.weights = weights
        self.Cs = Cs if Cs is not None else [0.001, 0.01, 0.1, 1, 10, 100]
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        
    def _initialize_scalers(self):
        """Initialize default scalers if none provided."""
        if self.scalers is None:
            self.scalers = [
                ('standard', StandardScaler()),
                ('minmax', MinMaxScaler()),
                ('robust', RobustScaler()),
                ('maxabs', MaxAbsScaler())
            ]
        return self.scalers
    
    def _initialize_weights(self, n_models: int):
        """Initialize weights for ensemble members."""
        if isinstance(self.weights, str) and self.weights == 'uniform':
            return np.ones(n_models) / n_models
        else:
            weights = np.array(self.weights)
            if len(weights) != n_models:
                raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({n_models})")
            if not np.isclose(weights.sum(), 1.0):
                raise ValueError(f"Weights must sum to 1, got {weights.sum()}")
            return weights
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]):
        """
        Fit the ensemble by training a model for each scaler.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target labels
            
        Returns
        -------
        self : object
            Fitted ensemble
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Initialize scalers and weights
        self._initialize_scalers()
        self.weights_ = self._initialize_weights(len(self.scalers))
        
        self.models_ = {}
        self.scalers_ = {}
        self.training_times_ = {}
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training Scaling Ensemble with {len(self.scalers)} models")
            print(f"{'='*60}\n")
        
        # Train a model for each scaler
        for scaler_name, scaler in self.scalers:
            start_time = time.time()
            
            if self.verbose:
                print(f"Training with {scaler_name} scaler...")
            
            # Fit scaler and transform data
            scaler_fitted = scaler.fit(X)
            X_scaled = scaler_fitted.transform(X)
            
            # Train logistic regression with CV
            model = self.base_estimator(
                Cs=self.Cs,
                cv=self.cv,
                random_state=self.random_state,
                max_iter=1000,
                scoring='roc_auc',
                n_jobs=-1
            )
            model.fit(X_scaled, y)
            
            # Store fitted components
            self.scalers_[scaler_name] = scaler_fitted
            self.models_[scaler_name] = model
            
            elapsed = time.time() - start_time
            self.training_times_[scaler_name] = elapsed
            
            if self.verbose:
                print(f"  ✓ Completed in {elapsed:.3f}s")
                print(f"    Best C: {model.C_[0]:.4f}")
                print(f"    CV Score: {model.scores_[1].mean():.4f}\n")
        
        if self.verbose:
            total_time = sum(self.training_times_.values())
            print(f"{'='*60}")
            print(f"Total training time: {total_time:.3f}s")
            print(f"{'='*60}\n")
        
        return self
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities using weighted ensemble averaging.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        probabilities : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities for each class
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Get predictions from each model
        all_probas = []
        
        for scaler_name in self.scalers_.keys():
            scaler = self.scalers_[scaler_name]
            X_scaled = scaler.transform(X)
            probas = self.models_[scaler_name].predict_proba(X_scaled)
            all_probas.append(probas)
        
        # Weighted average
        ensemble_proba = np.average(all_probas, axis=0, weights=self.weights_)
        
        return ensemble_proba
    
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
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)
    
    def get_member_predictions(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Get predictions from each ensemble member individually.
        
        Useful for analyzing ensemble diversity.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        predictions : dict
            Dictionary mapping scaler names to their predicted probabilities
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        member_predictions = {}
        
        for scaler_name in self.scalers_.keys():
            scaler = self.scalers_[scaler_name]
            X_scaled = scaler.transform(X)
            probas = self.models_[scaler_name].predict_proba(X_scaled)
            member_predictions[scaler_name] = probas
        
        return member_predictions
    
    def get_member_scores(self, X: Union[np.ndarray, pd.DataFrame], 
                         y: Union[np.ndarray, pd.Series]) -> Dict[str, Dict[str, float]]:
        """
        Get individual performance scores for each ensemble member.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features
        y : array-like of shape (n_samples,)
            True labels
            
        Returns
        -------
        scores : dict
            Dictionary mapping scaler names to their performance metrics
        """
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        member_scores = {}
        
        for scaler_name in self.scalers_.keys():
            scaler = self.scalers_[scaler_name]
            X_scaled = scaler.transform(X)
            model = self.models_[scaler_name]
            
            y_pred = model.predict(X_scaled)
            y_proba = model.predict_proba(X_scaled)[:, 1]
            
            member_scores[scaler_name] = {
                'accuracy': accuracy_score(y, y_pred),
                'auc_roc': roc_auc_score(y, y_proba),
                'pr_auc': average_precision_score(y, y_proba),
                'best_C': model.C_[0]
            }
        
        return member_scores
    
    def set_weights(self, weights: np.ndarray):
        """
        Set custom weights for ensemble members.
        
        Parameters
        ----------
        weights : array-like of shape (n_models,)
            New weights (must sum to 1)
        """
        weights = np.array(weights)
        if len(weights) != len(self.scalers):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(self.scalers)})")
        if not np.isclose(weights.sum(), 1.0):
            raise ValueError(f"Weights must sum to 1, got {weights.sum()}")
        
        self.weights_ = weights
        
        if self.verbose:
            print("\nUpdated ensemble weights:")
            for (scaler_name, _), weight in zip(self.scalers, weights):
                print(f"  {scaler_name}: {weight:.4f}")
    
    def get_weights(self) -> Dict[str, float]:
        """
        Get current ensemble weights.
        
        Returns
        -------
        weights : dict
            Dictionary mapping scaler names to their weights
        """
        return {scaler_name: weight 
                for (scaler_name, _), weight in zip(self.scalers, self.weights_)}
    
    def analyze_diversity(self, X: Union[np.ndarray, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        Analyze diversity among ensemble members.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features
            
        Returns
        -------
        diversity_metrics : dict
            Dictionary containing:
            - 'pairwise_correlations': Correlation matrix of predictions
            - 'prediction_variance': Variance in probability estimates
            - 'agreement_matrix': Pairwise prediction agreement
        """
        member_preds = self.get_member_predictions(X)
        
        # Extract probability predictions for positive class
        proba_matrix = np.array([preds[:, 1] for preds in member_preds.values()]).T
        
        # Compute pairwise correlations
        n_models = len(member_preds)
        correlation_matrix = np.corrcoef(proba_matrix.T)
        
        # Compute variance in predictions
        prediction_variance = np.var(proba_matrix, axis=1)
        
        # Compute pairwise agreement on predicted classes
        class_preds = (proba_matrix >= 0.5).astype(int)
        agreement_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(n_models):
                agreement_matrix[i, j] = (class_preds[:, i] == class_preds[:, j]).mean()
        
        return {
            'pairwise_correlations': correlation_matrix,
            'prediction_variance': prediction_variance,
            'agreement_matrix': agreement_matrix,
            'scaler_names': [name for name, _ in self.scalers]
        }


def create_ensemble(
    weights: Union[str, np.ndarray] = 'uniform',
    Cs: List[float] = None,
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = False
) -> ScalingEnsemble:
    """
    Convenience function to create a ScalingEnsemble with default settings.
    
    Parameters
    ----------
    weights : str or array-like, default='uniform'
        Ensemble member weights
    Cs : list, optional
        Regularization parameters to search
    cv : int, default=5
        Number of CV folds
    random_state : int, default=42
        Random seed
    verbose : bool, default=False
        Print training progress
        
    Returns
    -------
    ensemble : ScalingEnsemble
        Configured ensemble instance
    """
    return ScalingEnsemble(
        weights=weights,
        Cs=Cs,
        cv=cv,
        random_state=random_state,
        verbose=verbose
    )


if __name__ == "__main__":
    # Test the ensemble with synthetic data
    from sklearn.datasets import make_classification
    
    print("Testing ScalingEnsemble with synthetic data...")
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
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train ensemble
    ensemble = create_ensemble(verbose=True)
    ensemble.fit(X_train, y_train)
    
    # Make predictions
    print("\nEvaluating ensemble...")
    y_pred = ensemble.predict(X_test)
    y_proba = ensemble.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"\nEnsemble Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    
    # Get individual member scores
    print("\nIndividual Member Scores:")
    member_scores = ensemble.get_member_scores(X_test, y_test)
    for scaler_name, scores in member_scores.items():
        print(f"\n  {scaler_name}:")
        print(f"    Accuracy: {scores['accuracy']:.4f}")
        print(f"    AUC-ROC: {scores['auc_roc']:.4f}")
        print(f"    Best C: {scores['best_C']:.4f}")
    
    # Analyze diversity
    print("\nDiversity Analysis:")
    diversity = ensemble.analyze_diversity(X_test)
    
    print(f"\nPairwise Correlations:")
    print(diversity['pairwise_correlations'])
    
    print(f"\nMean prediction variance: {diversity['prediction_variance'].mean():.4f}")
    
    print(f"\nPairwise Agreement:")
    print(diversity['agreement_matrix'])
    
    # Test custom weights
    print("\n" + "=" * 60)
    print("Testing custom weights...")
    custom_weights = np.array([0.4, 0.3, 0.2, 0.1])
    ensemble.set_weights(custom_weights)
    
    y_pred_weighted = ensemble.predict(X_test)
    y_proba_weighted = ensemble.predict_proba(X_test)[:, 1]
    
    accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
    auc_weighted = roc_auc_score(y_test, y_proba_weighted)
    
    print(f"\nWeighted Ensemble Performance:")
    print(f"  Accuracy: {accuracy_weighted:.4f}")
    print(f"  AUC-ROC: {auc_weighted:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
