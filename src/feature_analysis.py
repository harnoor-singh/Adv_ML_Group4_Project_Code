"""
Feature analysis and dataset characterization utilities.

This module provides functions to analyze feature distributions, detect outliers,
compute scale variance, and generate visualizations for understanding dataset
characteristics in the context of feature scaling.

Student 1 implementation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')


class FeatureAnalyzer:
    """
    Comprehensive feature analysis for dataset characterization.
    
    Analyzes distributions, outliers, scale variance, and other characteristics
    that inform feature scaling strategy selection.
    """
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "Dataset"):
        """
        Initialize the feature analyzer.
        
        Args:
            X: Feature matrix (pandas DataFrame)
            y: Target labels (pandas Series)
            dataset_name: Name of the dataset for labeling outputs
        """
        self.X = X
        self.y = y
        self.dataset_name = dataset_name
        self.n_samples, self.n_features = X.shape
        self.feature_names = X.columns.tolist()
        
        # Storage for computed statistics
        self.stats_computed = False
        self.statistics = {}
        
    def compute_distribution_statistics(self) -> pd.DataFrame:
        """
        Compute distribution statistics for each feature.
        
        Computes:
        - Mean, median, std
        - Skewness (measure of asymmetry)
        - Kurtosis (measure of tail heaviness)
        - Min, max, range
        
        Returns:
            DataFrame with statistics for each feature
        """
        stats_dict = {
            'feature': [],
            'mean': [],
            'median': [],
            'std': [],
            'min': [],
            'max': [],
            'range': [],
            'skewness': [],
            'kurtosis': [],
        }
        
        for col in self.feature_names:
            data = self.X[col].values
            
            stats_dict['feature'].append(col)
            stats_dict['mean'].append(np.mean(data))
            stats_dict['median'].append(np.median(data))
            stats_dict['std'].append(np.std(data))
            stats_dict['min'].append(np.min(data))
            stats_dict['max'].append(np.max(data))
            stats_dict['range'].append(np.max(data) - np.min(data))
            stats_dict['skewness'].append(stats.skew(data))
            stats_dict['kurtosis'].append(stats.kurtosis(data))
        
        df_stats = pd.DataFrame(stats_dict)
        self.statistics['distribution'] = df_stats
        
        return df_stats
    
    def compute_outlier_statistics(self, iqr_multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect and count outliers using the IQR (Interquartile Range) method.
        
        Outliers are defined as:
        - Below Q1 - iqr_multiplier * IQR
        - Above Q3 + iqr_multiplier * IQR
        
        Args:
            iqr_multiplier: Multiplier for IQR threshold (default 1.5)
            
        Returns:
            DataFrame with outlier statistics for each feature
        """
        outlier_stats = {
            'feature': [],
            'Q1': [],
            'Q3': [],
            'IQR': [],
            'lower_bound': [],
            'upper_bound': [],
            'n_outliers_low': [],
            'n_outliers_high': [],
            'n_outliers_total': [],
            'outlier_percentage': [],
        }
        
        for col in self.feature_names:
            data = self.X[col].values
            
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
            outliers_low = np.sum(data < lower_bound)
            outliers_high = np.sum(data > upper_bound)
            outliers_total = outliers_low + outliers_high
            
            outlier_stats['feature'].append(col)
            outlier_stats['Q1'].append(q1)
            outlier_stats['Q3'].append(q3)
            outlier_stats['IQR'].append(iqr)
            outlier_stats['lower_bound'].append(lower_bound)
            outlier_stats['upper_bound'].append(upper_bound)
            outlier_stats['n_outliers_low'].append(outliers_low)
            outlier_stats['n_outliers_high'].append(outliers_high)
            outlier_stats['n_outliers_total'].append(outliers_total)
            outlier_stats['outlier_percentage'].append(100 * outliers_total / len(data))
        
        df_outliers = pd.DataFrame(outlier_stats)
        self.statistics['outliers'] = df_outliers
        
        return df_outliers
    
    def compute_scale_variance(self) -> Dict[str, float]:
        """
        Compute scale variance across features.
        
        Measures how much the scales of different features vary, which is
        important for determining if StandardScaler would be beneficial.
        
        Returns:
            Dictionary with scale variance metrics
        """
        # Compute range for each feature
        ranges = []
        stds = []
        
        for col in self.feature_names:
            data = self.X[col].values
            ranges.append(np.max(data) - np.min(data))
            stds.append(np.std(data))
        
        ranges = np.array(ranges)
        stds = np.array(stds)
        
        # Remove zero stds to avoid division by zero
        stds_nonzero = stds[stds > 0]
        ranges_nonzero = ranges[ranges > 0]
        
        scale_stats = {
            'range_min': np.min(ranges_nonzero) if len(ranges_nonzero) > 0 else 0,
            'range_max': np.max(ranges_nonzero) if len(ranges_nonzero) > 0 else 0,
            'range_ratio': np.max(ranges_nonzero) / np.min(ranges_nonzero) if len(ranges_nonzero) > 0 else 1,
            'range_std': np.std(ranges),
            'std_min': np.min(stds_nonzero) if len(stds_nonzero) > 0 else 0,
            'std_max': np.max(stds_nonzero) if len(stds_nonzero) > 0 else 0,
            'std_ratio': np.max(stds_nonzero) / np.min(stds_nonzero) if len(stds_nonzero) > 0 else 1,
            'std_variance': np.var(stds),
        }
        
        self.statistics['scale_variance'] = scale_stats
        
        return scale_stats
    
    def compute_all_statistics(self, iqr_multiplier: float = 1.5) -> Dict[str, Any]:
        """
        Compute all feature statistics in one call.
        
        Args:
            iqr_multiplier: Multiplier for IQR threshold in outlier detection
            
        Returns:
            Dictionary containing all computed statistics
        """
        print(f"\n{'='*60}")
        print(f"Feature Analysis: {self.dataset_name}")
        print(f"{'='*60}")
        print(f"Samples: {self.n_samples}, Features: {self.n_features}")
        
        # Compute all statistics
        dist_stats = self.compute_distribution_statistics()
        outlier_stats = self.compute_outlier_statistics(iqr_multiplier)
        scale_stats = self.compute_scale_variance()
        
        # Compute correlation matrix
        corr_matrix = self.X.corr()
        self.statistics['correlation'] = corr_matrix
        
        # Summary metrics
        summary = {
            'dataset_name': self.dataset_name,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'total_outliers': outlier_stats['n_outliers_total'].sum(),
            'avg_outlier_percentage': outlier_stats['outlier_percentage'].mean(),
            'max_outlier_percentage': outlier_stats['outlier_percentage'].max(),
            'scale_variance': scale_stats,
            'avg_skewness': dist_stats['skewness'].abs().mean(),
            'max_skewness': dist_stats['skewness'].abs().max(),
            'avg_kurtosis': dist_stats['kurtosis'].abs().mean(),
            'max_kurtosis': dist_stats['kurtosis'].abs().max(),
            'class_balance': self.y.value_counts().to_dict(),
        }
        
        self.statistics['summary'] = summary
        self.stats_computed = True
        
        # Print summary
        print(f"\nSummary Statistics:")
        print(f"  Total outliers: {summary['total_outliers']}")
        print(f"  Avg outlier %: {summary['avg_outlier_percentage']:.2f}%")
        print(f"  Max outlier %: {summary['max_outlier_percentage']:.2f}%")
        print(f"  Range ratio (max/min): {scale_stats['range_ratio']:.2f}")
        print(f"  Std ratio (max/min): {scale_stats['std_ratio']:.2f}")
        print(f"  Avg |skewness|: {summary['avg_skewness']:.2f}")
        print(f"  Max |kurtosis|: {summary['max_kurtosis']:.2f}")
        print(f"  Class balance: {summary['class_balance']}")
        
        return self.statistics
    
    def plot_distributions(self, save_path: Optional[str] = None, 
                          max_cols: int = 4) -> None:
        """
        Plot distribution histograms for all features.
        
        Args:
            save_path: Path to save the figure (optional)
            max_cols: Maximum number of columns in the subplot grid
        """
        n_plots = len(self.feature_names)
        n_cols = min(max_cols, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = np.array(axes).flatten()
        
        for idx, col in enumerate(self.feature_names):
            ax = axes[idx]
            data = self.X[col].values
            
            ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
            ax.set_title(f'{col}', fontsize=10)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add skewness annotation
            if 'distribution' in self.statistics:
                skew = self.statistics['distribution'].loc[
                    self.statistics['distribution']['feature'] == col, 'skewness'
                ].values[0]
                ax.text(0.02, 0.98, f'Skew: {skew:.2f}', 
                       transform=ax.transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Feature Distributions - {self.dataset_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        
        plt.show()
    
    def plot_boxplots(self, save_path: Optional[str] = None,
                     max_cols: int = 4) -> None:
        """
        Plot box plots for outlier visualization.
        
        Args:
            save_path: Path to save the figure (optional)
            max_cols: Maximum number of columns in the subplot grid
        """
        n_plots = len(self.feature_names)
        n_cols = min(max_cols, n_plots)
        n_rows = int(np.ceil(n_plots / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = np.array(axes).flatten()
        
        for idx, col in enumerate(self.feature_names):
            ax = axes[idx]
            data = self.X[col].values
            
            bp = ax.boxplot(data, vert=True, patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][0].set_alpha(0.7)
            
            ax.set_title(f'{col}', fontsize=10)
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add outlier count annotation
            if 'outliers' in self.statistics:
                n_outliers = self.statistics['outliers'].loc[
                    self.statistics['outliers']['feature'] == col, 'n_outliers_total'
                ].values[0]
                outlier_pct = self.statistics['outliers'].loc[
                    self.statistics['outliers']['feature'] == col, 'outlier_percentage'
                ].values[0]
                ax.text(0.02, 0.98, f'Outliers: {n_outliers}\n({outlier_pct:.1f}%)', 
                       transform=ax.transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
                       fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Feature Box Plots (Outlier Detection) - {self.dataset_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Box plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_heatmap(self, save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (12, 10),
                                annot: bool = False) -> None:
        """
        Plot correlation heatmap for features.
        
        Args:
            save_path: Path to save the figure (optional)
            figsize: Figure size (width, height)
            annot: Whether to annotate cells with correlation values
        """
        if 'correlation' not in self.statistics:
            corr_matrix = self.X.corr()
            self.statistics['correlation'] = corr_matrix
        else:
            corr_matrix = self.statistics['correlation']
        
        plt.figure(figsize=figsize)
        
        # Use a diverging colormap
        sns.heatmap(corr_matrix, annot=annot, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   fmt='.2f' if annot else None)
        
        plt.title(f'Feature Correlation Heatmap - {self.dataset_name}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_scale_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Visualize scale differences across features.
        
        Shows range and standard deviation for each feature to highlight
        scale variance that would benefit from standardization.
        
        Args:
            save_path: Path to save the figure (optional)
        """
        if 'distribution' not in self.statistics:
            self.compute_distribution_statistics()
        
        dist_stats = self.statistics['distribution']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot ranges
        features = dist_stats['feature'].values
        ranges = dist_stats['range'].values
        
        ax1.bar(range(len(features)), ranges, color='steelblue', alpha=0.7)
        ax1.set_xticks(range(len(features)))
        ax1.set_xticklabels(features, rotation=45, ha='right')
        ax1.set_ylabel('Range (max - min)')
        ax1.set_title('Feature Ranges')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot standard deviations
        stds = dist_stats['std'].values
        
        ax2.bar(range(len(features)), stds, color='coral', alpha=0.7)
        ax2.set_xticks(range(len(features)))
        ax2.set_xticklabels(features, rotation=45, ha='right')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Feature Standard Deviations')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Feature Scale Comparison - {self.dataset_name}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scale comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_all(self, save_dir: Optional[str] = None) -> None:
        """
        Generate all visualization plots.
        
        Args:
            save_dir: Directory to save all plots (optional)
        """
        print(f"\nGenerating visualizations for {self.dataset_name}...")
        
        # Ensure statistics are computed
        if not self.stats_computed:
            self.compute_all_statistics()
        
        # Generate plots
        dist_path = f"{save_dir}/{self.dataset_name}_distributions.png" if save_dir else None
        box_path = f"{save_dir}/{self.dataset_name}_boxplots.png" if save_dir else None
        corr_path = f"{save_dir}/{self.dataset_name}_correlation.png" if save_dir else None
        scale_path = f"{save_dir}/{self.dataset_name}_scale_comparison.png" if save_dir else None
        
        self.plot_distributions(save_path=dist_path)
        self.plot_boxplots(save_path=box_path)
        self.plot_correlation_heatmap(save_path=corr_path)
        self.plot_scale_comparison(save_path=scale_path)
        
        print("All visualizations generated!")
    
    def save_statistics(self, save_path: str) -> None:
        """
        Save computed statistics to CSV files.
        
        Args:
            save_path: Base path for saving statistics (without extension)
        """
        if not self.stats_computed:
            self.compute_all_statistics()
        
        # Save distribution statistics
        if 'distribution' in self.statistics:
            dist_path = f"{save_path}_distribution.csv"
            self.statistics['distribution'].to_csv(dist_path, index=False)
            print(f"Distribution statistics saved to {dist_path}")
        
        # Save outlier statistics
        if 'outliers' in self.statistics:
            outlier_path = f"{save_path}_outliers.csv"
            self.statistics['outliers'].to_csv(outlier_path, index=False)
            print(f"Outlier statistics saved to {outlier_path}")
        
        # Save correlation matrix
        if 'correlation' in self.statistics:
            corr_path = f"{save_path}_correlation.csv"
            self.statistics['correlation'].to_csv(corr_path)
            print(f"Correlation matrix saved to {corr_path}")
        
        # Save summary as JSON-like format
        if 'summary' in self.statistics:
            summary_path = f"{save_path}_summary.csv"
            summary = self.statistics['summary']
            
            # Convert to DataFrame for saving
            summary_data = {
                'metric': [],
                'value': []
            }
            
            for key, value in summary.items():
                if not isinstance(value, dict):
                    summary_data['metric'].append(key)
                    summary_data['value'].append(str(value))
            
            # Add scale variance metrics
            if 'scale_variance' in summary:
                for key, value in summary['scale_variance'].items():
                    summary_data['metric'].append(f"scale_{key}")
                    summary_data['value'].append(value)
            
            pd.DataFrame(summary_data).to_csv(summary_path, index=False)
            print(f"Summary statistics saved to {summary_path}")


def analyze_dataset(X: pd.DataFrame, y: pd.Series, 
                   dataset_name: str = "Dataset",
                   save_dir: Optional[str] = None,
                   generate_plots: bool = True) -> FeatureAnalyzer:
    """
    Convenience function to perform complete feature analysis on a dataset.
    
    Args:
        X: Feature matrix
        y: Target labels
        dataset_name: Name of the dataset
        save_dir: Directory to save results (optional)
        generate_plots: Whether to generate visualization plots
        
    Returns:
        FeatureAnalyzer object with computed statistics
    """
    analyzer = FeatureAnalyzer(X, y, dataset_name)
    analyzer.compute_all_statistics()
    
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Save statistics
        save_path = os.path.join(save_dir, dataset_name)
        analyzer.save_statistics(save_path)
        
        # Generate plots
        if generate_plots:
            analyzer.plot_all(save_dir=save_dir)
    elif generate_plots:
        analyzer.plot_all()
    
    return analyzer


def compare_datasets(analyzers: List[FeatureAnalyzer], 
                    save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Compare summary statistics across multiple datasets.
    
    Args:
        analyzers: List of FeatureAnalyzer objects
        save_path: Path to save comparison table (optional)
        
    Returns:
        DataFrame with comparison statistics
    """
    comparison_data = {
        'dataset': [],
        'n_samples': [],
        'n_features': [],
        'total_outliers': [],
        'avg_outlier_pct': [],
        'range_ratio': [],
        'std_ratio': [],
        'avg_skewness': [],
        'max_kurtosis': [],
    }
    
    for analyzer in analyzers:
        if not analyzer.stats_computed:
            analyzer.compute_all_statistics()
        
        summary = analyzer.statistics['summary']
        
        comparison_data['dataset'].append(analyzer.dataset_name)
        comparison_data['n_samples'].append(summary['n_samples'])
        comparison_data['n_features'].append(summary['n_features'])
        comparison_data['total_outliers'].append(summary['total_outliers'])
        comparison_data['avg_outlier_pct'].append(summary['avg_outlier_percentage'])
        comparison_data['range_ratio'].append(summary['scale_variance']['range_ratio'])
        comparison_data['std_ratio'].append(summary['scale_variance']['std_ratio'])
        comparison_data['avg_skewness'].append(summary['avg_skewness'])
        comparison_data['max_kurtosis'].append(summary['max_kurtosis'])
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n" + "="*80)
    print("Dataset Comparison Summary")
    print("="*80)
    print(df_comparison.to_string(index=False))
    
    if save_path:
        df_comparison.to_csv(save_path, index=False)
        print(f"\nComparison table saved to {save_path}")
    
    return df_comparison


if __name__ == "__main__":
    """
    Run feature analysis on all datasets and save results.
    
    Usage:
        python src/feature_analysis.py [--data-dir DATA_DIR] [--results-dir RESULTS_DIR] [--no-plots]
    """
    import argparse
    import os
    import sys
    
    # Add parent directory to path to import data_loader
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from src.data_loader import (
        load_wine_quality,
        preprocess_data
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run feature analysis on all datasets'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../data',
        help='Directory containing dataset files (default: ../data)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='../results/student1',
        help='Directory to save results (default: ../results/student1)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots (only compute statistics)'
    )
    parser.add_argument(
        '--iqr-multiplier',
        type=float,
        default=1.5,
        help='IQR multiplier for outlier detection (default: 1.5)'
    )
    
    args = parser.parse_args()
    
    # Create results directories
    os.makedirs(args.results_dir, exist_ok=True)
    figures_dir = os.path.join(args.results_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("FEATURE ANALYSIS - WINE QUALITY DATASET")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Results directory: {args.results_dir}")
    print(f"Generate plots: {not args.no_plots}")
    print(f"IQR multiplier: {args.iqr_multiplier}")
    
    # Store analyzers for comparison
    analyzers = []
    
    # Wine Quality Dataset
    print("\n" + "-"*80)
    print("Loading Wine Quality Dataset...")
    print("-"*80)
    try:
        X_wine, y_wine = load_wine_quality(args.data_dir)
        X_wine, y_wine = preprocess_data(X_wine, y_wine)
        
        analyzer_wine = FeatureAnalyzer(X_wine, y_wine, "Wine_Quality")
        analyzer_wine.compute_all_statistics(iqr_multiplier=args.iqr_multiplier)
        
        # Save statistics
        save_path = os.path.join(args.results_dir, "wine_quality")
        analyzer_wine.save_statistics(save_path)
        
        # Generate plots
        if not args.no_plots:
            analyzer_wine.plot_all(save_dir=figures_dir)
        
        analyzers.append(analyzer_wine)
        print("✓ Wine Quality analysis complete!")
        
    except Exception as e:
        print(f"✗ Error analyzing Wine Quality dataset: {e}")
    
    # Generate summary report
    if len(analyzers) > 0:
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)
        
        # Create a summary report
        report_path = os.path.join(args.results_dir, "analysis_report.txt")
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("FEATURE ANALYSIS REPORT - WINE QUALITY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Generated: {pd.Timestamp.now()}\n")
            f.write(f"IQR Multiplier: {args.iqr_multiplier}\n")
            f.write(f"Dataset: Wine Quality\n\n")
            
            # Dataset details
            for analyzer in analyzers:
                f.write("="*80 + "\n")
                f.write(f"DATASET: {analyzer.dataset_name}\n")
                f.write("="*80 + "\n\n")
                
                summary = analyzer.statistics['summary']
                f.write(f"Samples: {summary['n_samples']}\n")
                f.write(f"Features: {summary['n_features']}\n")
                f.write(f"Total Outliers: {summary['total_outliers']}\n")
                f.write(f"Avg Outlier %: {summary['avg_outlier_percentage']:.2f}%\n")
                f.write(f"Max Outlier %: {summary['max_outlier_percentage']:.2f}%\n")
                f.write(f"Range Ratio: {summary['scale_variance']['range_ratio']:.2f}\n")
                f.write(f"Std Ratio: {summary['scale_variance']['std_ratio']:.2f}\n")
                f.write(f"Avg |Skewness|: {summary['avg_skewness']:.2f}\n")
                f.write(f"Max |Kurtosis|: {summary['max_kurtosis']:.2f}\n")
                f.write(f"Class Balance: {summary['class_balance']}\n")
                f.write("\n")
                
                # Top 5 features by outlier percentage
                outlier_stats = analyzer.statistics['outliers']
                top_outliers = outlier_stats.nlargest(5, 'outlier_percentage')
                f.write("Top 5 Features by Outlier Percentage:\n")
                for idx, row in top_outliers.iterrows():
                    f.write(f"  {row['feature']}: {row['outlier_percentage']:.2f}% "
                           f"({row['n_outliers_total']} outliers)\n")
                f.write("\n")
                
                # Top 5 features by absolute skewness
                dist_stats = analyzer.statistics['distribution']
                dist_stats['abs_skewness'] = dist_stats['skewness'].abs()
                top_skewed = dist_stats.nlargest(5, 'abs_skewness')
                f.write("Top 5 Features by Absolute Skewness:\n")
                for idx, row in top_skewed.iterrows():
                    f.write(f"  {row['feature']}: {row['skewness']:.2f}\n")
                f.write("\n\n")
        
        print(f"\n✓ Analysis report saved to {report_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {args.results_dir}")
    print(f"Figures saved to: {figures_dir}")
    print("\nFiles generated:")
    print("  - wine_quality_distribution.csv (distribution statistics)")
    print("  - wine_quality_outliers.csv (outlier statistics)")
    print("  - wine_quality_correlation.csv (correlation matrix)")
    print("  - wine_quality_summary.csv (summary statistics)")
    print("  - analysis_report.txt (detailed text report)")
    if not args.no_plots:
        print("  - figures/*.png (visualization plots)")
    print("\n")
