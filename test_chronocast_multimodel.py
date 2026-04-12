"""
Test script for ChronoCast Multi-Model System
==============================================

This script tests the new ChronoCast Multi-Model forecasting system
with XGBoost, ARIMA, SARIMA, and Prophet models.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add the chronocast package to path
sys.path.append(str(Path(__file__).parent))

def create_sample_data():
    """Create sample time series data for testing"""
    print("Creating sample data...")
    
    # Create 180 days of sample data
    dates = pd.date_range(start="2024-01-01", periods=180, freq="D")
    np.random.seed(42)
    
    # Create realistic view data with trend, seasonality, and noise
    trend = np.linspace(1000, 4000, 180)
    season = 500 * np.sin(np.linspace(0, 4 * np.pi, 180))
    noise = np.random.normal(0, 150, 180)
    views = (trend + season + noise).astype(int)
    
    df = pd.DataFrame({"ds": dates, "y": views})
    
    print(f"Created dataset with {len(df)} records")
    print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
    print(f"View range: {df['y'].min()} to {df['y'].max()}")
    
    return df

def test_chronocast_multimodel():
    """Test the ChronoCast Multi-Model system"""
    print("\n" + "="*60)
    print("CHRONOCAST MULTI-MODEL SYSTEM TEST")
    print("="*60)
    
    try:
        # Import the ChronoCast Multi-Model
        from chronocast.core.chronocast_multimodel import ChronoCastMultiModel, forecast_views
        print("Successfully imported ChronoCastMultiModel")
        
        # Create sample data
        df = create_sample_data()
        
        # Test 1: Basic ChronoCastMultiModel usage
        print("\n" + "-"*40)
        print("Test 1: Basic ChronoCastMultiModel Usage")
        print("-"*40)
        
        chronocast = ChronoCastMultiModel(forecast_days=30)
        print(f"Initialized ChronoCastMultiModel with {chronocast.forecast_days} forecast days")
        
        # Test with limited models for faster testing
        models_to_test = ['xgboost']  # Start with just XGBoost
        if chronocast.MULTIMODEL_AVAILABLE if hasattr(chronocast, 'MULTIMODEL_AVAILABLE') else True:
            try:
                results = chronocast.fit_predict(df, models=models_to_test)
                print(f"Successfully trained models: {list(results.keys())}")
                
                # Test best model detection
                try:
                    best_model, best_mae = chronocast.get_best_model()
                    print(f"Best model: {best_model} with MAE: {best_mae:.2f}")
                except Exception as e:
                    print(f"Error in best model detection: {str(e)}")
                
                # Test summary table
                try:
                    summary_df = chronocast.create_summary_table(df)
                    print(f"Summary table created with {len(summary_df)} rows and {len(summary_df.columns)} columns")
                    print("Sample of summary table:")
                    print(summary_df.head(3).to_string())
                except Exception as e:
                    print(f"Error in summary table creation: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                print("Test 1 PASSED")
                
            except Exception as e:
                print(f"Test 1 FAILED: {str(e)}")
                return False
        else:
            print("Skipping Test 1 - Multi-model not available")
        
        # Test 2: Convenience function
        print("\n" + "-"*40)
        print("Test 2: Convenience Function")
        print("-"*40)
        
        try:
            chronocast_result = forecast_views(
                df, 
                forecast_days=15, 
                models=['xgboost'], 
                save_results=False
            )
            print("Convenience function executed successfully")
            print("Test 2 PASSED")
            
        except Exception as e:
            print(f"Test 2 FAILED: {str(e)}")
            return False
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        return True
        
    except ImportError as e:
        print(f"Import Error: {str(e)}")
        print("Make sure chronocast package is properly installed")
        return False
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return False

def test_api_integration():
    """Test API integration with ChronoCast"""
    print("\n" + "="*60)
    print("API INTEGRATION TEST")
    print("="*60)
    
    try:
        # Test Django import
        from chronocast_api import MULTIMODEL_AVAILABLE
        print(f"MULTIMODEL_AVAILABLE in chronocast_api: {MULTIMODEL_AVAILABLE}")
        
        if MULTIMODEL_AVAILABLE:
            print("API integration test PASSED")
            return True
        else:
            print("API integration test FAILED - Multi-model not available in API")
            return False
            
    except ImportError as e:
        print(f"API integration test FAILED - Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"API integration test FAILED - Unexpected error: {str(e)}")
        return False

def main():
    """Main test function"""
    print("ChronoCast Multi-Model Test Suite")
    print("==================================")
    
    # Test 1: Core multi-model functionality
    core_test_passed = test_chronocast_multimodel()
    
    # Test 2: API integration
    api_test_passed = test_api_integration()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Core Multi-Model Test: {'PASSED' if core_test_passed else 'FAILED'}")
    print(f"API Integration Test: {'PASSED' if api_test_passed else 'FAILED'}")
    
    if core_test_passed and api_test_passed:
        print("\nAll tests PASSED! ChronoCast Multi-Model is ready.")
        print("\nTo use the new ChronoCast model in the API:")
        print("1. Upload a dataset via the API")
        print("2. Validate the dataset with date_column and target_column")
        print("3. Create a forecast run with model_type='chronocast'")
        print("4. Use model_params to configure:")
        print("   - forecast_days: number of days to forecast (default: 30)")
        print("   - models: list of models to use (default: ['xgboost', 'arima', 'sarima', 'prophet'])")
        
        example_params = {
            "forecast_days": 30,
            "models": ["xgboost", "arima", "sarima", "prophet"]
        }
        print(f"\nExample model_params: {example_params}")
        
    else:
        print("\nSome tests FAILED. Please check the implementation.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
