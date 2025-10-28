"""
Explainability Module for ChronoCast
Provides model interpretability using SHAP and feature importance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List, Union
import json
from datetime import datetime
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not installed. Install with: pip install shap")


class ModelExplainer:
    """
    Provides explainability for ChronoCast models
    """
    
    def __init__(self, model, X_train, feature_names: Optional[List[str]] = None):
        """
        Initialize explainer
        
        Args:
            model: Trained model (ChronoModel or sklearn model)
            X_train: Training data for SHAP background
            feature_names: List of feature names
        """
        # Handle ChronoModel wrapper
        if hasattr(model, 'model'):
            self.model = model.model
            self.model_wrapper = model
        else:
            self.model = model
            self.model_wrapper = None
        
        self.X_train = X_train if isinstance(X_train, np.ndarray) else X_train.values
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
        # Initialize SHAP explainer if available
        if SHAP_AVAILABLE:
            self._initialize_shap_explainer()
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer based on model type"""
        try:
            # Tree-based models (XGBoost, RandomForest, GradientBoosting)
            if hasattr(self.model, 'tree_') or hasattr(self.model, 'estimators_'):
                self.explainer = shap.TreeExplainer(self.model)
                self.explainer_type = 'tree'
            # Linear models
            elif hasattr(self.model, 'coef_'):
                # Use background data for linear models
                self.explainer = shap.LinearExplainer(self.model, self.X_train)
                self.explainer_type = 'linear'
            # General models (fallback to Kernel explainer with sampling)
            else:
                # Sample data to speed up kernel explainer
                background = shap.sample(self.X_train, min(100, len(self.X_train)))
                self.explainer = shap.KernelExplainer(self.model.predict, background)
                self.explainer_type = 'kernel'
        except Exception as e:
            warnings.warn(f"Could not initialize SHAP explainer: {str(e)}")
            self.explainer = None
    
    def calculate_shap_values(self, X_test, max_samples: Optional[int] = None):
        """
        Calculate SHAP values for test data
        
        Args:
            X_test: Test data
            max_samples: Maximum samples to explain (for performance)
        
        Returns:
            SHAP values array
        """
        if not SHAP_AVAILABLE or self.explainer is None:
            warnings.warn("SHAP not available or explainer not initialized")
            return None
        
        X_test_array = X_test if isinstance(X_test, np.ndarray) else X_test.values
        
        # Limit samples for performance
        if max_samples and len(X_test_array) > max_samples:
            indices = np.random.choice(len(X_test_array), max_samples, replace=False)
            X_test_array = X_test_array[indices]
        
        try:
            if self.explainer_type == 'tree':
                self.shap_values = self.explainer.shap_values(X_test_array)
            elif self.explainer_type == 'linear':
                self.shap_values = self.explainer.shap_values(X_test_array)
            else:  # kernel
                self.shap_values = self.explainer.shap_values(X_test_array)
            
            return self.shap_values
        except Exception as e:
            warnings.warn(f"Error calculating SHAP values: {str(e)}")
            return None
    
    def plot_feature_importance(self, top_n: int = 20, 
                               figsize: tuple = (10, 8),
                               save_path: Optional[str] = None):
        """
        Plot feature importance (for tree-based models)
        
        Args:
            top_n: Number of top features to show
            figsize: Figure size
            save_path: Path to save plot
        """
        # Try to get feature importance from model
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            if self.feature_names:
                features = self.feature_names
            else:
                features = [f'Feature {i}' for i in range(len(importances))]
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': importances
            }).sort_values('importance', ascending=False).head(top_n)
            
            # Plot
            plt.figure(figsize=figsize)
            plt.barh(range(len(importance_df)), importance_df['importance'].values)
            plt.yticks(range(len(importance_df)), importance_df['feature'].values)
            plt.xlabel('Importance', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title(f'Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            plt.show()
            
            return importance_df
        
        elif hasattr(self.model, 'coef_'):
            # For linear models, use coefficients
            coefs = np.abs(self.model.coef_)
            
            if self.feature_names:
                features = self.feature_names
            else:
                features = [f'Feature {i}' for i in range(len(coefs))]
            
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': coefs
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=figsize)
            plt.barh(range(len(importance_df)), importance_df['importance'].values)
            plt.yticks(range(len(importance_df)), importance_df['feature'].values)
            plt.xlabel('|Coefficient|', fontsize=12)
            plt.ylabel('Feature', fontsize=12)
            plt.title(f'Top {top_n} Feature Coefficients (Absolute)', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            plt.show()
            
            return importance_df
        
        else:
            print("Model does not support feature importance")
            return None
    
    def plot_shap_summary(self, X_test, max_samples: int = 100,
                         figsize: tuple = (10, 8),
                         save_path: Optional[str] = None):
        """
        Plot SHAP summary plot
        
        Args:
            X_test: Test data
            max_samples: Maximum samples to use
            figsize: Figure size
            save_path: Path to save plot
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap")
            return
        
        # Calculate SHAP values if not already done
        if self.shap_values is None:
            self.calculate_shap_values(X_test, max_samples)
        
        if self.shap_values is None:
            print("Could not calculate SHAP values")
            return
        
        # Prepare data
        X_test_array = X_test if isinstance(X_test, np.ndarray) else X_test.values
        if len(X_test_array) > max_samples:
            indices = np.random.choice(len(X_test_array), max_samples, replace=False)
            X_test_array = X_test_array[indices]
        
        plt.figure(figsize=figsize)
        shap.summary_plot(self.shap_values, X_test_array, 
                         feature_names=self.feature_names,
                         show=False)
        plt.title('SHAP Summary Plot', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_shap_waterfall(self, X_test, instance_idx: int = 0,
                           figsize: tuple = (10, 8),
                           save_path: Optional[str] = None):
        """
        Plot SHAP waterfall for a single prediction
        
        Args:
            X_test: Test data
            instance_idx: Index of instance to explain
            figsize: Figure size
            save_path: Path to save plot
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap")
            return
        
        # Calculate SHAP values if not already done
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        if self.shap_values is None:
            print("Could not calculate SHAP values")
            return
        
        X_test_array = X_test if isinstance(X_test, np.ndarray) else X_test.values
        
        plt.figure(figsize=figsize)
        
        # Create explanation object for waterfall plot
        if self.explainer_type == 'tree':
            expected_value = self.explainer.expected_value
        else:
            expected_value = self.explainer.expected_value
        
        explanation = shap.Explanation(
            values=self.shap_values[instance_idx],
            base_values=expected_value if np.isscalar(expected_value) else expected_value[0],
            data=X_test_array[instance_idx],
            feature_names=self.feature_names
        )
        
        shap.waterfall_plot(explanation, show=False)
        plt.title(f'SHAP Waterfall Plot - Instance {instance_idx}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_shap_force(self, X_test, instance_idx: int = 0,
                       matplotlib: bool = True,
                       save_path: Optional[str] = None):
        """
        Plot SHAP force plot for a single prediction
        
        Args:
            X_test: Test data
            instance_idx: Index of instance to explain
            matplotlib: Use matplotlib (True) or interactive HTML (False)
            save_path: Path to save plot
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap")
            return
        
        # Calculate SHAP values if not already done
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        if self.shap_values is None:
            print("Could not calculate SHAP values")
            return
        
        X_test_array = X_test if isinstance(X_test, np.ndarray) else X_test.values
        
        if self.explainer_type == 'tree':
            expected_value = self.explainer.expected_value
        else:
            expected_value = self.explainer.expected_value
        
        if matplotlib:
            shap.force_plot(
                expected_value if np.isscalar(expected_value) else expected_value[0],
                self.shap_values[instance_idx],
                X_test_array[instance_idx],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f'SHAP Force Plot - Instance {instance_idx}', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Plot saved to {save_path}")
            
            plt.show()
        else:
            # Return interactive HTML
            return shap.force_plot(
                expected_value if np.isscalar(expected_value) else expected_value[0],
                self.shap_values[instance_idx],
                X_test_array[instance_idx],
                feature_names=self.feature_names
            )
    
    def get_feature_contributions(self, X_test, instance_idx: int = 0) -> pd.DataFrame:
        """
        Get feature contributions for a single prediction
        
        Args:
            X_test: Test data
            instance_idx: Index of instance
        
        Returns:
            DataFrame with features and their contributions
        """
        if self.shap_values is None:
            self.calculate_shap_values(X_test)
        
        if self.shap_values is None:
            return None
        
        X_test_array = X_test if isinstance(X_test, np.ndarray) else X_test.values
        
        contributions = pd.DataFrame({
            'feature': self.feature_names if self.feature_names else [f'Feature {i}' for i in range(X_test_array.shape[1])],
            'value': X_test_array[instance_idx],
            'shap_value': self.shap_values[instance_idx]
        })
        
        contributions['abs_shap'] = np.abs(contributions['shap_value'])
        contributions = contributions.sort_values('abs_shap', ascending=False)
        
        return contributions
    
    def save_explainability_log(self, X_test, output_path: str = 'explainability_log.json'):
        """
        Save explainability information to JSON
        
        Args:
            X_test: Test data
            output_path: Path to save log
        """
        log = {
            'timestamp': datetime.now().isoformat(),
            'model_type': str(type(self.model).__name__),
            'explainer_type': self.explainer_type if hasattr(self, 'explainer_type') else 'unknown',
            'n_features': X_test.shape[1],
            'n_samples': X_test.shape[0],
            'shap_available': SHAP_AVAILABLE
        }
        
        # Add feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            if self.feature_names:
                log['feature_importance'] = dict(zip(self.feature_names, importance.tolist()))
        
        # Add SHAP summary statistics if available
        if self.shap_values is not None:
            log['shap_summary'] = {
                'mean_abs_shap': float(np.mean(np.abs(self.shap_values))),
                'max_abs_shap': float(np.max(np.abs(self.shap_values))),
                'min_abs_shap': float(np.min(np.abs(self.shap_values)))
            }
        
        with open(output_path, 'w') as f:
            json.dump(log, f, indent=4)
        
        print(f"Explainability log saved to {output_path}")


if __name__ == "__main__":
    print("Explainability Module - Demo\n")
    print("="*60)
    
    # Create sample data
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    
    X, y = make_regression(n_samples=500, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    feature_names = [f'feature_{i}' for i in range(20)]
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("✓ Model trained")
    
    # Create explainer
    print("\nInitializing explainer...")
    explainer = ModelExplainer(model, X_train, feature_names=feature_names)
    print("✓ Explainer initialized")
    
    # Feature importance
    print("\n1. Feature Importance")
    print("-"*60)
    importance_df = explainer.plot_feature_importance(top_n=10)
    
    if SHAP_AVAILABLE:
        # SHAP summary
        print("\n2. SHAP Summary Plot")
        print("-"*60)
        explainer.plot_shap_summary(X_test, max_samples=100)
        
        # SHAP waterfall
        print("\n3. SHAP Waterfall Plot (Instance 0)")
        print("-"*60)
        explainer.plot_shap_waterfall(X_test, instance_idx=0)
        
        # Feature contributions
        print("\n4. Feature Contributions")
        print("-"*60)
        contributions = explainer.get_feature_contributions(X_test, instance_idx=0)
        print(contributions.head(10))
        
        # Save log
        print("\n5. Saving Explainability Log")
        print("-"*60)
        explainer.save_explainability_log(X_test)
    
    print("\n" + "="*60)
    print("Demo complete!")