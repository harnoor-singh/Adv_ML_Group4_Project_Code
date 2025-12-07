"""
Data loading and preprocessing utilities for the Feature Scaling Ensemble project.

This module provides functions to load, preprocess, and split datasets for 
binary classification tasks.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from typing import Tuple, Dict, Any
import warnings

warnings.filterwarnings('ignore')


def load_wine_quality(data_dir: str = '../data') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Wine Quality dataset and convert to binary classification.
    
    Binary task: quality >= 7 (good wine) vs quality < 7 (average/poor wine)
    
    Args:
        data_dir: Directory containing the wine quality CSV files
        
    Returns:
        X: Feature matrix (pandas DataFrame)
        y: Binary target labels (pandas Series)
    """
    # Try to load red and white wine datasets
    red_path = os.path.join(data_dir, 'winequality-red.csv')
    white_path = os.path.join(data_dir, 'winequality-white.csv')
    
    dfs = []
    
    if os.path.exists(red_path):
        red_wine = pd.read_csv(red_path, sep=';')
        red_wine['wine_type'] = 0  # Red wine
        dfs.append(red_wine)
    
    if os.path.exists(white_path):
        white_wine = pd.read_csv(white_path, sep=';')
        white_wine['wine_type'] = 1  # White wine
        dfs.append(white_wine)
    
    if not dfs:
        raise FileNotFoundError(
            f"Wine quality data not found in {data_dir}. "
            "Please download from UCI ML Repository."
        )
    
    # Combine datasets
    df = pd.concat(dfs, ignore_index=True)
    
    # Convert to binary classification: quality >= 7 is "good" (1), else "not good" (0)
    y = (df['quality'] >= 7).astype(int)
    
    # Features (all columns except quality)
    X = df.drop('quality', axis=1)
    
    print(f"Wine Quality Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X, y


def load_breast_cancer_data() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Breast Cancer Wisconsin dataset from sklearn.
    
    Binary task: malignant (1) vs benign (0)
    
    Returns:
        X: Feature matrix (pandas DataFrame)
        y: Binary target labels (pandas Series)
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    print(f"Breast Cancer Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X, y


def load_heart_disease(data_dir: str = '../data') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load Heart Disease dataset and convert to binary classification.
    
    Binary task: disease present (1) vs no disease (0)
    
    Args:
        data_dir: Directory containing the heart disease CSV file
        
    Returns:
        X: Feature matrix (pandas DataFrame)
        y: Binary target labels (pandas Series)
    """
    heart_path = os.path.join(data_dir, 'heart.csv')
    
    if not os.path.exists(heart_path):
        raise FileNotFoundError(
            f"Heart disease data not found at {heart_path}. "
            "Please download from UCI/Kaggle."
        )
    
    df = pd.read_csv(heart_path)
    
    # Assuming the target column is named 'target' or 'num' or similar
    # Adjust based on actual dataset structure
    possible_target_cols = ['target', 'num', 'disease', 'HeartDisease']
    target_col = None
    
    for col in possible_target_cols:
        if col in df.columns:
            target_col = col
            break
    
    if target_col is None:
        # Assume last column is target
        target_col = df.columns[-1]
        print(f"Warning: Assuming '{target_col}' is the target column")
    
    # Convert to binary: any disease (>0) vs no disease (0)
    y = (df[target_col] > 0).astype(int)
    X = df.drop(target_col, axis=1)
    
    # Handle categorical variables if present
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"Encoding categorical columns: {list(categorical_cols)}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    print(f"Heart Disease Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {y.value_counts().to_dict()}")
    
    return X, y


def preprocess_data(X: pd.DataFrame, y: pd.Series, 
                   handle_missing: str = 'mean') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the dataset by handling missing values.
    
    Args:
        X: Feature matrix
        y: Target labels
        handle_missing: Strategy for missing values ('mean', 'median', 'drop')
        
    Returns:
        X: Preprocessed feature matrix
        y: Corresponding target labels
    """
    # Check for missing values
    missing_counts = X.isnull().sum()
    if missing_counts.sum() > 0:
        print(f"Found {missing_counts.sum()} missing values")
        
        if handle_missing == 'drop':
            # Drop rows with missing values
            mask = X.isnull().any(axis=1)
            X = X[~mask]
            y = y[~mask]
            print(f"Dropped {mask.sum()} rows with missing values")
        elif handle_missing == 'mean':
            X = X.fillna(X.mean())
            print("Filled missing values with column means")
        elif handle_missing == 'median':
            X = X.fillna(X.median())
            print("Filled missing values with column medians")
    
    # Reset indices
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2, 
              random_state: int = 42,
              stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets.
    
    Args:
        X: Feature matrix
        y: Target labels
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        stratify: Whether to maintain class distribution in splits
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"\nData split: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Train class distribution: {y_train.value_counts().to_dict()}")
    print(f"Test class distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def get_dataset_statistics(X: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute comprehensive statistics about the dataset.
    
    Args:
        X: Feature matrix
        
    Returns:
        Dictionary containing various dataset statistics
    """
    stats = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'feature_names': list(X.columns),
        'mean_values': X.mean().to_dict(),
        'std_values': X.std().to_dict(),
        'min_values': X.min().to_dict(),
        'max_values': X.max().to_dict(),
        'median_values': X.median().to_dict(),
        'skewness': X.skew().to_dict(),
        'kurtosis': X.kurtosis().to_dict(),
        'missing_values': X.isnull().sum().to_dict(),
        'scale_ranges': (X.max() - X.min()).to_dict(),
    }
    
    # Compute scale variance (variance of feature scales)
    feature_scales = X.max() - X.min()
    stats['scale_variance'] = float(feature_scales.var())
    stats['scale_range_ratio'] = float(feature_scales.max() / (feature_scales.min() + 1e-10))
    
    return stats


def load_all_datasets(data_dir: str = '../data') -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Load all available datasets.
    
    Args:
        data_dir: Directory containing dataset files
        
    Returns:
        Dictionary mapping dataset names to (X, y) tuples
    """
    datasets = {}
    
    print("=" * 60)
    print("Loading datasets...")
    print("=" * 60)
    
    # Load Wine Quality
    try:
        X_wine, y_wine = load_wine_quality(data_dir)
        X_wine, y_wine = preprocess_data(X_wine, y_wine)
        datasets['wine_quality'] = (X_wine, y_wine)
    except Exception as e:
        print(f"Warning: Could not load Wine Quality dataset: {e}")
    
    # Load Breast Cancer
    try:
        X_cancer, y_cancer = load_breast_cancer_data()
        X_cancer, y_cancer = preprocess_data(X_cancer, y_cancer)
        datasets['breast_cancer'] = (X_cancer, y_cancer)
    except Exception as e:
        print(f"Warning: Could not load Breast Cancer dataset: {e}")
    
    # Load Heart Disease
    try:
        X_heart, y_heart = load_heart_disease(data_dir)
        X_heart, y_heart = preprocess_data(X_heart, y_heart)
        datasets['heart_disease'] = (X_heart, y_heart)
    except Exception as e:
        print(f"Warning: Could not load Heart Disease dataset: {e}")
    
    print("=" * 60)
    print(f"Successfully loaded {len(datasets)} dataset(s)")
    print("=" * 60)
    
    return datasets


def prepare_dataset(dataset_name: str, 
                   data_dir: str = '../data',
                   test_size: float = 0.2,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Complete pipeline to load, preprocess, and split a dataset.
    
    Args:
        dataset_name: Name of dataset ('wine_quality', 'breast_cancer', 'heart_disease')
        data_dir: Directory containing dataset files
        test_size: Proportion for test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test, dataset_stats
    """
    print(f"\n{'='*60}")
    print(f"Preparing {dataset_name} dataset")
    print(f"{'='*60}\n")
    
    # Load dataset
    if dataset_name == 'wine_quality':
        X, y = load_wine_quality(data_dir)
    elif dataset_name == 'breast_cancer':
        X, y = load_breast_cancer_data()
    elif dataset_name == 'heart_disease':
        X, y = load_heart_disease(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Preprocess
    X, y = preprocess_data(X, y)
    
    # Get statistics before splitting
    stats = get_dataset_statistics(X)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    return X_train, X_test, y_train, y_test, stats


if __name__ == "__main__":
    # Test the data loader
    import sys
    sys.path.append('..')
    
    # Load all datasets
    datasets = load_all_datasets('../data')
    
    # Print statistics for each dataset
    print("\n" + "=" * 60)
    print("Dataset Statistics Summary")
    print("=" * 60)
    
    for name, (X, y) in datasets.items():
        print(f"\n{name.upper()}:")
        stats = get_dataset_statistics(X)
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Features: {stats['n_features']}")
        print(f"  Scale variance: {stats['scale_variance']:.2f}")
        print(f"  Scale range ratio: {stats['scale_range_ratio']:.2f}")
        
        # Class balance
        class_counts = y.value_counts()
        balance_ratio = class_counts.min() / class_counts.max()
        print(f"  Class balance ratio: {balance_ratio:.2f}")
