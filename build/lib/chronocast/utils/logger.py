"""
Logging Module for ChronoCast
Provides comprehensive logging and transparency tracking
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd


class ChronoLogger:
    """
    Comprehensive logging system for ChronoCast
    Tracks training, predictions, and model lifecycle
    """
    
    def __init__(self, log_dir: str = 'chronocast_logs', log_level: str = 'INFO'):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to store logs
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Set up Python logging
        self.logger = logging.getLogger('ChronoCast')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # File handler
        log_file = self.log_dir / f'chronocast_{datetime.now().strftime("%Y%m%d")}.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers if not already added
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
        
        # Training history
        self.training_runs = []
        self.current_run = None
    
    def start_training(self, model_name: str, params: Dict[str, Any]) -> str:
        """
        Log start of training
        
        Args:
            model_name: Name of the model
            params: Model parameters
        
        Returns:
            Run ID
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_run = {
            'run_id': run_id,
            'model_name': model_name,
            'params': params,
            'start_time': datetime.now().isoformat(),
            'status': 'training'
        }
        
        self.logger.info(f"Started training run: {run_id}")
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Parameters: {json.dumps(params, indent=2)}")
        
        return run_id
    
    def log_training_data(self, n_samples: int, n_features: int, 
                         feature_names: Optional[List[str]] = None):
        """
        Log training data information
        
        Args:
            n_samples: Number of training samples
            n_features: Number of features
            feature_names: Optional list of feature names
        """
        if self.current_run:
            self.current_run['data_info'] = {
                'n_samples': n_samples,
                'n_features': n_features,
                'feature_names': feature_names
            }
            
            self.logger.info(f"Training data: {n_samples} samples, {n_features} features")
    
    def log_training_progress(self, epoch: int, metrics: Dict[str, float]):
        """
        Log training progress (for iterative models)
        
        Args:
            epoch: Current epoch/iteration
            metrics: Training metrics
        """
        self.logger.debug(f"Epoch {epoch}: {metrics}")
    
    def end_training(self, training_time: float, final_metrics: Dict[str, float]):
        """
        Log end of training
        
        Args:
            training_time: Total training time in seconds
            final_metrics: Final evaluation metrics
        """
        if self.current_run:
            self.current_run['end_time'] = datetime.now().isoformat()
            self.current_run['training_time'] = training_time
            self.current_run['metrics'] = final_metrics
            self.current_run['status'] = 'completed'
            
            self.training_runs.append(self.current_run.copy())
            
            self.logger.info(f"Training completed in {training_time:.2f}s")
            self.logger.info(f"Metrics: {json.dumps(final_metrics, indent=2)}")
            
            # Save to JSON
            self._save_run_log(self.current_run)
            
            self.current_run = None
    
    def log_prediction(self, n_predictions: int, prediction_time: float):
        """
        Log prediction information
        
        Args:
            n_predictions: Number of predictions made
            prediction_time: Time taken for predictions
        """
        self.logger.info(f"Made {n_predictions} predictions in {prediction_time:.4f}s")
    
    def log_error(self, error_message: str, exception: Optional[Exception] = None):
        """
        Log an error
        
        Args:
            error_message: Error description
            exception: Optional exception object
        """
        if exception:
            self.logger.error(f"{error_message}: {str(exception)}", exc_info=True)
        else:
            self.logger.error(error_message)
        
        if self.current_run:
            self.current_run['status'] = 'failed'
            self.current_run['error'] = error_message
    
    def log_model_save(self, model_path: str, model_size: Optional[int] = None):
        """
        Log model save operation
        
        Args:
            model_path: Path where model was saved
            model_size: Optional model size in bytes
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'model_size': model_size
        }
        
        self.logger.info(f"Model saved to: {model_path}")
        if model_size:
            self.logger.info(f"Model size: {model_size / (1024*1024):.2f} MB")
    
    def log_model_load(self, model_path: str):
        """
        Log model load operation
        
        Args:
            model_path: Path from which model was loaded
        """
        self.logger.info(f"Model loaded from: {model_path}")
    
    def _save_run_log(self, run_data: Dict[str, Any]):
        """
        Save run log to JSON file
        
        Args:
            run_data: Run information to save
        """
        log_file = self.log_dir / f"{run_data['run_id']}.json"
        
        with open(log_file, 'w') as f:
            json.dump(run_data, f, indent=4)
        
        self.logger.debug(f"Run log saved to: {log_file}")
    
    def get_training_history(self) -> pd.DataFrame:
        """
        Get training history as DataFrame
        
        Returns:
            DataFrame with training runs
        """
        if not self.training_runs:
            return pd.DataFrame()
        
        # Flatten metrics for DataFrame
        flattened = []
        for run in self.training_runs:
            flat_run = {
                'run_id': run['run_id'],
                'model_name': run['model_name'],
                'start_time': run['start_time'],
                'training_time': run.get('training_time', 0),
                'status': run['status']
            }
            
            # Add metrics
            if 'metrics' in run:
                for key, value in run['metrics'].items():
                    flat_run[f'metric_{key}'] = value
            
            flattened.append(flat_run)
        
        return pd.DataFrame(flattened)
    
    def export_logs(self, output_path: str = 'training_history.json'):
        """
        Export all training logs
        
        Args:
            output_path: Path to save logs
        """
        with open(output_path, 'w') as f:
            json.dump(self.training_runs, f, indent=4)
        
        self.logger.info(f"Training history exported to: {output_path}")
    
    def generate_report(self, output_path: str = 'training_report.txt'):
        """
        Generate human-readable training report
        
        Args:
            output_path: Path to save report
        """
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ChronoCast Training Report\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total training runs: {len(self.training_runs)}\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for idx, run in enumerate(self.training_runs, 1):
                f.write(f"\n{'-'*70}\n")
                f.write(f"Run #{idx}: {run['run_id']}\n")
                f.write(f"{'-'*70}\n")
                f.write(f"Model: {run['model_name']}\n")
                f.write(f"Status: {run['status']}\n")
                f.write(f"Start: {run['start_time']}\n")
                
                if 'training_time' in run:
                    f.write(f"Training time: {run['training_time']:.2f}s\n")
                
                if 'data_info' in run:
                    data = run['data_info']
                    f.write(f"Samples: {data['n_samples']}\n")
                    f.write(f"Features: {data['n_features']}\n")
                
                if 'metrics' in run:
                    f.write("\nMetrics:\n")
                    for key, value in run['metrics'].items():
                        f.write(f"  {key}: {value:.4f}\n")
                
                if 'params' in run:
                    f.write("\nParameters:\n")
                    for key, value in run['params'].items():
                        f.write(f"  {key}: {value}\n")
        
        self.logger.info(f"Report generated: {output_path}")


class ExperimentTracker:
    """
    Track multiple experiments for comparison
    """
    
    def __init__(self, experiment_name: str, log_dir: str = 'experiments'):
        """
        Initialize experiment tracker
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to store experiment logs
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments = []
        self.logger = ChronoLogger(log_dir=str(self.log_dir))
    
    def log_experiment(self, 
                      model_name: str,
                      params: Dict[str, Any],
                      metrics: Dict[str, float],
                      notes: Optional[str] = None):
        """
        Log an experiment
        
        Args:
            model_name: Name of the model
            params: Model parameters
            metrics: Evaluation metrics
            notes: Optional notes
        """
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'params': params,
            'metrics': metrics,
            'notes': notes
        }
        
        self.experiments.append(experiment)
        self._save_experiments()
        
        self.logger.logger.info(f"Logged experiment: {model_name}")
    
    def _save_experiments(self):
        """Save all experiments to JSON"""
        exp_file = self.log_dir / 'experiments.json'
        
        with open(exp_file, 'w') as f:
            json.dump(self.experiments, f, indent=4)
    
    def get_best_experiment(self, metric: str = 'RMSE', 
                           minimize: bool = True) -> Dict[str, Any]:
        """
        Get best experiment based on a metric
        
        Args:
            metric: Metric to compare
            minimize: Whether to minimize the metric
        
        Returns:
            Best experiment dictionary
        """
        if not self.experiments:
            return {}
        
        valid_exps = [e for e in self.experiments if metric in e['metrics']]
        
        if not valid_exps:
            return {}
        
        if minimize:
            best = min(valid_exps, key=lambda x: x['metrics'][metric])
        else:
            best = max(valid_exps, key=lambda x: x['metrics'][metric])
        
        return best
    
    def compare_experiments(self) -> pd.DataFrame:
        """
        Compare all experiments
        
        Returns:
            DataFrame with experiment comparison
        """
        if not self.experiments:
            return pd.DataFrame()
        
        rows = []
        for exp in self.experiments:
            row = {
                'model_name': exp['model_name'],
                'timestamp': exp['timestamp']
            }
            row.update(exp['metrics'])
            rows.append(row)
        
        return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Logging Module - Demo\n")
    print("="*60)
    
    # Initialize logger
    logger = ChronoLogger(log_dir='demo_logs')
    
    # Simulate training
    print("\n1. Starting Training...")
    run_id = logger.start_training('XGBoost', {'n_estimators': 100, 'max_depth': 5})
    logger.log_training_data(1000, 50, ['feature_' + str(i) for i in range(50)])
    
    import time
    time.sleep(0.5)
    
    # End training
    print("\n2. Ending Training...")
    logger.end_training(0.5, {'RMSE': 25.5, 'MAE': 18.3, 'R²': 0.85})
    
    # Log prediction
    print("\n3. Logging Prediction...")
    logger.log_prediction(200, 0.05)
    
    # Get history
    print("\n4. Training History:")
    history = logger.get_training_history()
    print(history)
    
    # Export logs
    print("\n5. Exporting Logs...")
    logger.export_logs('demo_training_history.json')
    logger.generate_report('demo_training_report.txt')
    
    # Experiment tracker
    print("\n6. Experiment Tracking...")
    tracker = ExperimentTracker('model_comparison', log_dir='demo_experiments')
    
    tracker.log_experiment('XGBoost', {'n_estimators': 100}, {'RMSE': 25.5, 'R²': 0.85})
    tracker.log_experiment('RandomForest', {'n_estimators': 100}, {'RMSE': 28.2, 'R²': 0.82})
    
    comparison = tracker.compare_experiments()
    print("\nExperiment Comparison:")
    print(comparison)
    
    best = tracker.get_best_experiment('RMSE', minimize=True)
    print(f"\nBest Model: {best['model_name']} (RMSE: {best['metrics']['RMSE']})")
    
    print("\n" + "="*60)
    print("Demo complete!")