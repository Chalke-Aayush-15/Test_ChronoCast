import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Loader, AlertCircle, TrendingUp, BarChart3, Brain, Download } from 'lucide-react';
import Plot from 'react-plotly.js';
import { forecastAPI, explainabilityAPI, handleAPIError } from '../services/api';

export default function ResultsPage() {
  const { runId } = useParams();
  const navigate = useNavigate();
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [forecastRun, setForecastRun] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [explainability, setExplainability] = useState(null);
  const [generatingExplain, setGeneratingExplain] = useState(false);
  
  useEffect(() => {
    loadResults();
  }, [runId]);
  
  const loadResults = async () => {
    setLoading(true);
    try {
      // Get forecast run
      const runResponse = await forecastAPI.get(runId);
      setForecastRun(runResponse.data);
      
      // Get predictions
      const predResponse = await forecastAPI.getPredictions(runId, 1, 1000);
      setPredictions(predResponse.data.results);
      
    } catch (err) {
      const { message } = handleAPIError(err);
      setError(message);
    } finally {
      setLoading(false);
    }
  };
  
  const generateExplainability = async () => {
    setGeneratingExplain(true);
    try {
      const response = await forecastAPI.generateExplainability(runId, 50);
      setExplainability(response.data);
    } catch (err) {
      const { message } = handleAPIError(err);
      setError(message);
    } finally {
      setGeneratingExplain(false);
    }
  };
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader className="animate-spin h-8 w-8 text-primary-600" />
      </div>
    );
  }
  
  if (error || !forecastRun) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex items-start">
          <AlertCircle className="h-6 w-6 text-red-600 mt-0.5 mr-3" />
          <div>
            <h3 className="text-lg font-medium text-red-800">Error Loading Results</h3>
            <p className="text-sm text-red-700 mt-1">{error || 'Forecast run not found'}</p>
            <button
              onClick={() => navigate('/forecast')}
              className="mt-4 text-sm text-red-800 hover:text-red-900 underline"
            >
              Back to Forecast
            </button>
          </div>
        </div>
      </div>
    );
  }
  
  const metrics = forecastRun.metrics || {};
  
  // Prepare chart data
  const dates = predictions.map(p => p.date);
  const actual = predictions.map(p => p.actual);
  const predicted = predictions.map(p => p.predicted);
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Forecast Results</h1>
          <p className="text-gray-600">
            {forecastRun.model_type.toUpperCase()} - {forecastRun.dataset_name}
          </p>
        </div>
        <button
          onClick={() => navigate('/forecast')}
          className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
        >
          New Forecast
        </button>
      </div>
      
      {/* Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'RMSE', value: metrics.RMSE?.toFixed(2), icon: TrendingUp },
          { label: 'MAE', value: metrics.MAE?.toFixed(2), icon: BarChart3 },
          { label: 'R²', value: metrics['R²']?.toFixed(4), icon: BarChart3 },
          { label: 'MAPE', value: metrics.MAPE?.toFixed(2) + '%', icon: TrendingUp },
        ].map((metric) => {
          const Icon = metric.icon;
          return (
            <div key={metric.label} className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-600">{metric.label}</span>
                <Icon className="h-4 w-4 text-gray-400" />
              </div>
              <div className="text-2xl font-bold text-gray-900">
                {metric.value || 'N/A'}
              </div>
            </div>
          );
        })}
      </div>
      
      {/* Forecast Chart */}
      <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Actual vs Predicted
        </h2>
        <Plot
          data={[
            {
              x: dates,
              y: actual,
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Actual',
              line: { color: '#3b82f6' },
              marker: { size: 6 },
            },
            {
              x: dates,
              y: predicted,
              type: 'scatter',
              mode: 'lines+markers',
              name: 'Predicted',
              line: { color: '#10b981', dash: 'dash' },
              marker: { size: 6, symbol: 'x' },
            },
          ]}
          layout={{
            autosize: true,
            height: 400,
            xaxis: { title: 'Date' },
            yaxis: { title: 'Value' },
            hovermode: 'x unified',
            showlegend: true,
            legend: { x: 0, y: 1 },
          }}
          useResizeHandler
          style={{ width: '100%' }}
          config={{ responsive: true }}
        />
      </div>
      
      {/* Error Distribution */}
      <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Error Distribution
        </h2>
        <Plot
          data={[
            {
              x: predictions.map(p => p.error),
              type: 'histogram',
              marker: { color: '#8b5cf6' },
              nbinsx: 30,
            },
          ]}
          layout={{
            autosize: true,
            height: 300,
            xaxis: { title: 'Prediction Error' },
            yaxis: { title: 'Frequency' },
            showlegend: false,
          }}
          useResizeHandler
          style={{ width: '100%' }}
          config={{ responsive: true }}
        />
      </div>
      
      {/* Feature Importance */}
      {forecastRun.feature_importance && (
        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Top 10 Feature Importance
          </h2>
          <Plot
            data={[
              {
                y: forecastRun.feature_importance.slice(0, 10).map(f => f.feature),
                x: forecastRun.feature_importance.slice(0, 10).map(f => f.importance),
                type: 'bar',
                orientation: 'h',
                marker: { color: '#0ea5e9' },
              },
            ]}
            layout={{
              autosize: true,
              height: 400,
              xaxis: { title: 'Importance' },
              yaxis: { title: '' },
              margin: { l: 150 },
            }}
            useResizeHandler
            style={{ width: '100%' }}
            config={{ responsive: true }}
          />
        </div>
      )}
      
      {/* Explainability */}
      <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <Brain className="h-5 w-5 text-gray-600 mr-2" />
            <h2 className="text-lg font-semibold text-gray-900">
              Model Explainability
            </h2>
          </div>
          {!explainability && (
            <button
              onClick={generateExplainability}
              disabled={generatingExplain}
              className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50"
            >
              {generatingExplain ? (
                <span className="flex items-center">
                  <Loader className="animate-spin h-4 w-4 mr-2" />
                  Generating...
                </span>
              ) : (
                'Generate SHAP Analysis'
              )}
            </button>
          )}
        </div>
        
        {explainability ? (
          <div>
            <p className="text-sm text-gray-600 mb-4">
              SHAP analysis shows how each feature contributes to predictions.
            </p>
            <div className="bg-green-50 border border-green-200 rounded-lg p-4">
              <p className="text-sm text-green-800">
                ✓ Explainability analysis generated successfully!
              </p>
              <p className="text-sm text-green-700 mt-1">
                Analyzed {explainability.n_samples_explained} samples
              </p>
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-600">
            Generate SHAP values to understand how each feature contributes to predictions.
          </p>
        )}
      </div>
      
      {/* Training Info */}
      <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Training Information
        </h2>
        <dl className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <dt className="text-gray-600">Model Type</dt>
            <dd className="font-medium text-gray-900">{forecastRun.model_type.toUpperCase()}</dd>
          </div>
          <div>
            <dt className="text-gray-600">Training Time</dt>
            <dd className="font-medium text-gray-900">{forecastRun.training_time?.toFixed(2)}s</dd>
          </div>
          <div>
            <dt className="text-gray-600">Train Samples</dt>
            <dd className="font-medium text-gray-900">{forecastRun.n_train_samples}</dd>
          </div>
          <div>
            <dt className="text-gray-600">Test Samples</dt>
            <dd className="font-medium text-gray-900">{forecastRun.n_test_samples}</dd>
          </div>
          <div>
            <dt className="text-gray-600">Features Used</dt>
            <dd className="font-medium text-gray-900">{forecastRun.n_features}</dd>
          </div>
          <div>
            <dt className="text-gray-600">Status</dt>
            <dd className="font-medium text-green-600">{forecastRun.status}</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}