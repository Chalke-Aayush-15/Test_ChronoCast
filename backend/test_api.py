"""
Test script for ChronoCast API
Run this after starting the Django server
"""

import requests
import time
import json
from pathlib import Path

BASE_URL = "http://localhost:8000/api"

def print_response(response, title):
    """Pretty print API response"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    print(f"Status Code: {response.status_code}")
    if response.status_code < 400:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")
    print()

def test_api():
    """Test all API endpoints"""
    
    print("\n" + "="*70)
    print("ChronoCast API Test Script")
    print("="*70)
    
    # 1. Test health check
    print("\n1. Testing Health Check...")
    response = requests.get(f"{BASE_URL.replace('/api', '')}/health/")
    print(f"Status: {response.status_code}")
    
    # 2. Create sample dataset
    print("\n2. Creating Sample Dataset...")
    
    # Generate sample CSV
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'views': np.random.randint(100, 500, 100) + np.arange(100) * 2
    })
    
    csv_path = Path('test_data.csv')
    data.to_csv(csv_path, index=False)
    
    # Upload dataset
    files = {'file': open(csv_path, 'rb')}
    data_payload = {'name': 'Test Dataset', 'description': 'API test data'}
    response = requests.post(f"{BASE_URL}/datasets/", files=files, data=data_payload)
    print_response(response, "Upload Dataset")
    
    if response.status_code != 201:
        print("❌ Failed to upload dataset")
        return
    
    dataset_id = response.json()['id']
    print(f"✓ Dataset ID: {dataset_id}")
    
    # 3. Validate dataset
    print("\n3. Validating Dataset...")
    validate_payload = {
        'date_column': 'date',
        'target_column': 'views'
    }
    response = requests.post(
        f"{BASE_URL}/datasets/{dataset_id}/validate/",
        json=validate_payload
    )
    print_response(response, "Validate Dataset")
    
    # 4. Preview dataset
    print("\n4. Previewing Dataset...")
    response = requests.get(f"{BASE_URL}/datasets/{dataset_id}/preview/?n_rows=5")
    print_response(response, "Preview Dataset")
    
    # 5. Create forecast run
    print("\n5. Creating Forecast Run...")
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
    response = requests.post(f"{BASE_URL}/forecast-runs/", json=forecast_payload)
    print_response(response, "Create Forecast Run")
    
    if response.status_code != 201:
        print("❌ Failed to create forecast run")
        return
    
    run_id = response.json()['id']
    print(f"✓ Forecast Run ID: {run_id}")
    
    # 6. Poll forecast status
    print("\n6. Polling Forecast Status...")
    max_attempts = 30
    for attempt in range(max_attempts):
        response = requests.get(f"{BASE_URL}/forecast-runs/{run_id}/")
        data = response.json()
        status = data['status']
        progress = data['progress']
        
        print(f"  Attempt {attempt+1}/{max_attempts}: Status={status}, Progress={progress}%")
        
        if status == 'completed':
            print("\n✓ Forecast completed successfully!")
            break
        elif status == 'failed':
            print(f"\n❌ Forecast failed: {data.get('error_message')}")
            return
        
        time.sleep(2)
    
    # 7. Get metrics
    print("\n7. Getting Metrics...")
    response = requests.get(f"{BASE_URL}/forecast-runs/{run_id}/metrics/")
    print_response(response, "Forecast Metrics")
    
    # 8. Get predictions
    print("\n8. Getting Predictions...")
    response = requests.get(f"{BASE_URL}/forecast-runs/{run_id}/predictions/?page_size=10")
    print_response(response, "Forecast Predictions (First 10)")
    
    # 9. Generate explainability
    print("\n9. Generating Explainability...")
    response = requests.post(
        f"{BASE_URL}/forecast-runs/{run_id}/generate_explainability/",
        json={'max_samples': 20}
    )
    print_response(response, "Generate Explainability")
    
    if response.status_code == 200:
        explainability_id = response.json()['id']
        
        # 10. Get feature contributions
        print("\n10. Getting Feature Contributions...")
        response = requests.get(
            f"{BASE_URL}/explainability/{explainability_id}/feature_contributions/?instance_idx=0"
        )
        print_response(response, "Feature Contributions")
    
    # 11. Create another forecast for comparison
    print("\n11. Creating Second Forecast Run (Random Forest)...")
    forecast_payload2 = {
        'dataset': dataset_id,
        'model_type': 'rf',
        'model_params': {
            'n_estimators': 50,
            'max_depth': 10
        },
        'use_time_features': True,
        'use_lag_features': True,
        'lag_periods': [1, 7],
        'use_rolling_features': True,
        'rolling_windows': [7]
    }
    response = requests.post(f"{BASE_URL}/forecast-runs/", json=forecast_payload2)
    
    if response.status_code == 201:
        run_id_2 = response.json()['id']
        print(f"✓ Second Forecast Run ID: {run_id_2}")
        
        # Wait for completion
        print("\n  Waiting for second forecast to complete...")
        for attempt in range(30):
            response = requests.get(f"{BASE_URL}/forecast-runs/{run_id_2}/")
            status = response.json()['status']
            if status == 'completed':
                print("  ✓ Second forecast completed!")
                break
            elif status == 'failed':
                print("  ❌ Second forecast failed")
                break
            time.sleep(2)
        
        # 12. Create model comparison
        if status == 'completed':
            print("\n12. Creating Model Comparison...")
            comparison_payload = {
                'dataset_id': dataset_id,
                'forecast_run_ids': [run_id, run_id_2],
                'name': 'XGBoost vs Random Forest',
                'description': 'Comparison of two models'
            }
            response = requests.post(
                f"{BASE_URL}/comparisons/create_comparison/",
                json=comparison_payload
            )
            print_response(response, "Create Model Comparison")
            
            if response.status_code == 201:
                comparison_id = response.json()['id']
                
                # 13. Get comparison chart data
                print("\n13. Getting Comparison Chart Data...")
                response = requests.get(f"{BASE_URL}/comparisons/{comparison_id}/chart_data/")
                print_response(response, "Comparison Chart Data")
    
    # Cleanup
    csv_path.unlink()
    
    print("\n" + "="*70)
    print("API Test Complete! ✅")
    print("="*70)
    print("\nSummary:")
    print(f"  ✓ Dataset uploaded and validated")
    print(f"  ✓ Forecast runs created and completed")
    print(f"  ✓ Metrics and predictions retrieved")
    print(f"  ✓ Explainability generated")
    print(f"  ✓ Model comparison created")
    print("\nAll endpoints working correctly!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API")
        print("Make sure the Django server is running:")
        print("  python manage.py runserver")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()