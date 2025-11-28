"""
Quick fix script for ChronoCast API testing
Ensures dataset is properly validated before forecasting
"""

import requests
import time
import json
import pandas as pd
import numpy as np
from pathlib import Path

BASE_URL = "http://localhost:8000/api"

def print_status(message, status="info"):
    """Print colored status messages"""
    colors = {
        "success": "\033[92m✓",
        "error": "\033[91m✗",
        "info": "\033[94mℹ",
        "warning": "\033[93m⚠"
    }
    reset = "\033[0m"
    print(f"{colors.get(status, colors['info'])} {message}{reset}")

def test_corrected_workflow():
    """Test API with proper validation workflow"""
    
    print("\n" + "="*70)
    print("ChronoCast API - Corrected Test Workflow")
    print("="*70)
    
    try:
        # 1. Check server health
        print("\n1. Checking server health...")
        response = requests.get(f"{BASE_URL.replace('/api', '')}/health/")
        if response.status_code == 200:
            print_status("Server is running", "success")
        else:
            print_status("Server not responding", "error")
            return
        
        # 2. Create sample dataset
        print("\n2. Creating sample dataset...")
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'views': np.random.randint(100, 500, 100) + np.arange(100) * 2
        })
        
        csv_path = Path('test_data.csv')
        data.to_csv(csv_path, index=False)
        print_status(f"Sample data created: {len(data)} rows", "success")
        
        # 3. Upload dataset
        print("\n3. Uploading dataset...")
        with open(csv_path, 'rb') as f:
            files = {'file': f}
            data_payload = {'name': 'Test Dataset', 'description': 'API test data'}
            response = requests.post(f"{BASE_URL}/datasets/", files=files, data=data_payload)
        
        if response.status_code != 201:
            print_status(f"Upload failed: {response.text}", "error")
            csv_path.unlink()  # Clean up before returning
            return
        
        dataset_id = response.json()['id']
        print_status(f"Dataset uploaded: {dataset_id}", "success")
        
        # 4. Validate dataset (IMPORTANT - with correct headers)
        print("\n4. Validating dataset...")
        validate_payload = {
            'date_column': 'date',
            'target_column': 'views'
        }
        response = requests.post(
            f"{BASE_URL}/datasets/{dataset_id}/validate/",
            json=validate_payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 200:
            print_status(f"Validation failed: {response.text}", "error")
            print_status("Cannot proceed without validation", "error")
            return
        
        validation_result = response.json()
        print_status(f"Dataset validated successfully", "success")
        print(f"   Date range: {validation_result['date_range']['start']} to {validation_result['date_range']['end']}")
        print(f"   Samples: {validation_result['date_range']['n_periods']}")
        
        # 5. Verify dataset has columns set
        print("\n5. Verifying dataset configuration...")
        response = requests.get(f"{BASE_URL}/datasets/{dataset_id}/")
        dataset_info = response.json()
        
        if dataset_info.get('date_column') and dataset_info.get('target_column'):
            print_status(f"Date column: {dataset_info['date_column']}", "success")
            print_status(f"Target column: {dataset_info['target_column']}", "success")
        else:
            print_status("Dataset columns not configured!", "error")
            return
        
        # 6. Create forecast run
        print("\n6. Creating forecast run...")
        forecast_payload = {
            'dataset': dataset_id,
            'model_type': 'xgb',
            'model_params': {
                'n_estimators': 50,
                'max_depth': 5,
                'learning_rate': 0.1
            },
            'use_time_features': True,
            'use_lag_features': True,
            'lag_periods': [1, 7],
            'use_rolling_features': True,
            'rolling_windows': [7]
        }
        response = requests.post(
            f"{BASE_URL}/forecast-runs/",
            json=forecast_payload,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code != 201:
            print_status(f"Forecast creation failed: {response.text}", "error")
            return
        
        run_id = response.json()['id']
        print_status(f"Forecast run created: {run_id}", "success")
        
        # 7. Poll for completion
        print("\n7. Waiting for forecast to complete...")
        max_attempts = 30
        
        for attempt in range(max_attempts):
            response = requests.get(f"{BASE_URL}/forecast-runs/{run_id}/")
            run_data = response.json()
            status_val = run_data['status']
            progress = run_data['progress']
            
            if status_val == 'completed':
                print_status(f"Forecast completed! (took {attempt+1} checks)", "success")
                break
            elif status_val == 'failed':
                error_msg = run_data.get('error_message', 'Unknown error')
                print_status(f"Forecast failed: {error_msg}", "error")
                return
            
            print(f"   Status: {status_val}, Progress: {progress}%", end='\r')
            time.sleep(2)
        
        # 8. Get results
        print("\n\n8. Retrieving results...")
        
        # Metrics
        response = requests.get(f"{BASE_URL}/forecast-runs/{run_id}/metrics/")
        if response.status_code == 200:
            metrics = response.json()
            print_status("Metrics retrieved", "success")
            print(f"   RMSE: {metrics.get('RMSE', 'N/A'):.2f}")
            print(f"   MAE: {metrics.get('MAE', 'N/A'):.2f}")
            print(f"   R²: {metrics.get('R²', 'N/A'):.4f}")
        
        # Predictions
        response = requests.get(f"{BASE_URL}/forecast-runs/{run_id}/predictions/?page_size=5")
        if response.status_code == 200:
            pred_data = response.json()
            print_status(f"Predictions retrieved: {pred_data['count']} total", "success")
            print("   First 3 predictions:")
            for pred in pred_data['results'][:3]:
                print(f"   - Date: {pred['date'][:10]}, Actual: {pred['actual']:.0f}, Predicted: {pred['predicted']:.0f}")
        
        # Cleanup
        try:
            if csv_path.exists():
                csv_path.unlink()
                print_status("Cleanup completed", "success")
        except PermissionError:
            print_status("Note: test_data.csv will be reused on next run", "warning")
        
        print("\n" + "="*70)
        print_status("Test completed successfully!", "success")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print_status("Cannot connect to server. Is it running?", "error")
        print("   Run: python manage.py runserver")
    except Exception as e:
        print_status(f"Error: {str(e)}", "error")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_corrected_workflow()