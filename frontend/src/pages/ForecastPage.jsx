import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Loader, AlertCircle, CheckCircle, Settings } from 'lucide-react';
import { datasetAPI, forecastAPI, pollForecastStatus, handleAPIError } from '../services/api';

export default function ForecastPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const datasetId = searchParams.get('dataset');
  
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(datasetId || '');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [forecasting, setForecasting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  
  // Model configuration
  const [modelType, setModelType] = useState('xgb');
  const [modelParams, setModelParams] = useState({
    n_estimators: 100,
    max_depth: 5,
    learning_rate: 0.1,
  });
  const [lagPeriods, setLagPeriods] = useState([1, 7, 14]);
  const [rollingWindows, setRollingWindows] = useState([7, 14]);
  
  useEffect(() => {
    loadDatasets();
  }, []);
  
  const loadDatasets = async () => {
    setLoading(true);
    try {
      const response = await datasetAPI.list();
      setDatasets(response.data);
    } catch (err) {
      const { message } = handleAPIError(err);
      setError(message);
    } finally {
      setLoading(false);
    }
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedDataset) {
      setError('Please select a dataset');
      return;
    }
    
    setForecasting(true);
    setError(null);
    setProgress(0);
    setStatus('pending');
    
    try {
      // Create forecast run
      const response = await forecastAPI.create({
        dataset: selectedDataset,
        model_type: modelType,
        model_params: modelParams,
        use_time_features: true,
        use_lag_features: true,
        lag_periods: lagPeriods,
        use_rolling_features: true,
        rolling_windows: rollingWindows,
      });
      
      const runId = response.data.id;
      
      // Poll for completion
      await pollForecastStatus(
        runId,
        ({ status, progress }) => {
          setStatus(status);
          setProgress(progress);
        },
        2000
      );
      
      // Navigate to results
      navigate(`/forecast/${runId}`);
      
    } catch (err) {
      const { message } = handleAPIError(err);
      setError(message);
      setForecasting(false);
    }
  };
  
  const modelOptions = [
    { value: 'linear', label: 'Linear Regression', description: 'Simple linear model' },
    { value: 'ridge', label: 'Ridge Regression', description: 'Linear with L2 regularization' },
    { value: 'rf', label: 'Random Forest', description: 'Ensemble of decision trees' },
    { value: 'gbm', label: 'Gradient Boosting', description: 'Sequential boosting algorithm' },
    { value: 'xgb', label: 'XGBoost', description: 'Optimized gradient boosting' },
  ];
  
  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Generate Forecast</h1>
        <p className="text-gray-600">
          Configure and train your forecasting model
        </p>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Dataset Selection */}
        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            1. Select Dataset
          </h2>
          
          {loading ? (
            <div className="text-center py-4">
              <Loader className="animate-spin h-6 w-6 mx-auto text-primary-600" />
            </div>
          ) : (
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
              required
            >
              <option value="">Select a dataset</option>
              {datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name} ({dataset.n_rows} rows)
                </option>
              ))}
            </select>
          )}
          
          {datasets.length === 0 && !loading && (
            <p className="text-sm text-gray-500 mt-2">
              No datasets found. <a href="/upload" className="text-primary-600 hover:underline">Upload one</a> to get started.
            </p>
          )}
        </div>
        
        {/* Model Selection */}
        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            2. Choose Model
          </h2>
          
          <div className="grid md:grid-cols-2 gap-4">
            {modelOptions.map((option) => (
              <label
                key={option.value}
                className={`flex items-start p-4 border-2 rounded-lg cursor-pointer transition-colors ${
                  modelType === option.value
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <input
                  type="radio"
                  name="model"
                  value={option.value}
                  checked={modelType === option.value}
                  onChange={(e) => setModelType(e.target.value)}
                  className="mt-1 text-primary-600 focus:ring-primary-500"
                />
                <div className="ml-3">
                  <div className="font-medium text-gray-900">{option.label}</div>
                  <div className="text-sm text-gray-500">{option.description}</div>
                </div>
              </label>
            ))}
          </div>
        </div>
        
        {/* Model Parameters */}
        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <div className="flex items-center mb-4">
            <Settings className="h-5 w-5 text-gray-600 mr-2" />
            <h2 className="text-lg font-semibold text-gray-900">
              3. Model Parameters
            </h2>
          </div>
          
          {(modelType === 'rf' || modelType === 'gbm' || modelType === 'xgb') && (
            <div className="grid md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Number of Estimators
                </label>
                <input
                  type="number"
                  value={modelParams.n_estimators}
                  onChange={(e) => setModelParams({ ...modelParams, n_estimators: parseInt(e.target.value) })}
                  min="10"
                  max="500"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Max Depth
                </label>
                <input
                  type="number"
                  value={modelParams.max_depth}
                  onChange={(e) => setModelParams({ ...modelParams, max_depth: parseInt(e.target.value) })}
                  min="1"
                  max="20"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Learning Rate
                </label>
                <input
                  type="number"
                  step="0.01"
                  value={modelParams.learning_rate}
                  onChange={(e) => setModelParams({ ...modelParams, learning_rate: parseFloat(e.target.value) })}
                  min="0.01"
                  max="1"
                  className="w-full px-3 py-2 border border-gray-300 rounded-md"
                />
              </div>
            </div>
          )}
        </div>
        
        {/* Feature Engineering */}
        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            4. Feature Engineering
          </h2>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Lag Periods
              </label>
              <input
                type="text"
                value={lagPeriods.join(', ')}
                onChange={(e) => setLagPeriods(e.target.value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v)))}
                placeholder="1, 7, 14, 30"
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              />
              <p className="text-sm text-gray-500 mt-1">
                Comma-separated list of lag periods (e.g., 1, 7, 14)
              </p>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Rolling Windows
              </label>
              <input
                type="text"
                value={rollingWindows.join(', ')}
                onChange={(e) => setRollingWindows(e.target.value.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v)))}
                placeholder="7, 14, 30"
                className="w-full px-3 py-2 border border-gray-300 rounded-md"
              />
              <p className="text-sm text-gray-500 mt-1">
                Comma-separated list of rolling window sizes
              </p>
            </div>
          </div>
        </div>
        
        {/* Submit */}
        <div className="flex justify-end space-x-4">
          <button
            type="button"
            onClick={() => navigate('/')}
            className="px-6 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={forecasting || !selectedDataset}
            className="px-6 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {forecasting ? (
              <span className="flex items-center">
                <Loader className="animate-spin h-4 w-4 mr-2" />
                Training Model...
              </span>
            ) : (
              'Start Forecast'
            )}
          </button>
        </div>
      </form>
      
      {/* Progress */}
      {forecasting && (
        <div className="mt-6 bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Training Progress
          </h3>
          
          <div className="mb-2">
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Status: {status}</span>
              <span>{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
          
          <div className="flex items-start mt-4 text-sm text-gray-600">
            <Loader className="animate-spin h-4 w-4 mr-2 mt-0.5" />
            <span>
              {progress < 20 && 'Loading dataset...'}
              {progress >= 20 && progress < 40 && 'Creating features...'}
              {progress >= 40 && progress < 70 && 'Training model...'}
              {progress >= 70 && progress < 100 && 'Evaluating results...'}
              {progress === 100 && 'Complete!'}
            </span>
          </div>
        </div>
      )}
      
      {/* Error */}
      {error && (
        <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-start">
            <AlertCircle className="h-5 w-5 text-red-600 mt-0.5 mr-3" />
            <div>
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}