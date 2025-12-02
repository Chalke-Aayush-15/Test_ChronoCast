import { useState, useEffect } from 'react';
import { Loader, AlertCircle, GitCompare, CheckCircle } from 'lucide-react';
import Plot from 'react-plotly.js';
import { datasetAPI, forecastAPI, comparisonAPI, handleAPIError } from '../services/api';

export default function ComparePage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [forecastRuns, setForecastRuns] = useState([]);
  const [selectedRuns, setSelectedRuns] = useState([]);
  const [comparing, setComparing] = useState(false);
  const [comparisonResult, setComparisonResult] = useState(null);
  
  useEffect(() => {
    loadDatasets();
  }, []);
  
  useEffect(() => {
    if (selectedDataset) {
      loadForecastRuns();
    }
  }, [selectedDataset]);
  
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
  
  const loadForecastRuns = async () => {
    try {
      const response = await forecastAPI.list({ dataset: selectedDataset });
      const completedRuns = response.data.filter(run => run.status === 'completed');
      setForecastRuns(completedRuns);
      setSelectedRuns([]);
    } catch (err) {
      const { message } = handleAPIError(err);
      setError(message);
    }
  };
  
  const toggleRunSelection = (runId) => {
    setSelectedRuns(prev => 
      prev.includes(runId)
        ? prev.filter(id => id !== runId)
        : [...prev, runId]
    );
  };
  
  const handleCompare = async () => {
    if (selectedRuns.length < 2) {
      setError('Please select at least 2 forecast runs to compare');
      return;
    }
    
    setComparing(true);
    setError(null);
    
    try {
      const response = await comparisonAPI.create({
        dataset_id: selectedDataset,
        forecast_run_ids: selectedRuns,
        name: `Comparison ${new Date().toLocaleString()}`,
      });
      
      const comparisonId = response.data.id;
      
      // Get chart data
      const chartResponse = await comparisonAPI.getChartData(comparisonId);
      setComparisonResult(chartResponse.data);
      
    } catch (err) {
      const { message } = handleAPIError(err);
      setError(message);
    } finally {
      setComparing(false);
    }
  };
  
  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader className="animate-spin h-8 w-8 text-primary-600" />
      </div>
    );
  }
  
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Compare Models</h1>
          <p className="text-gray-600">
            Select multiple forecast runs to compare their performance
          </p>
        </div>
      </div>
      
      {/* Dataset Selection */}
      <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          1. Select Dataset
        </h2>
        <select
          value={selectedDataset}
          onChange={(e) => setSelectedDataset(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
        >
          <option value="">Choose a dataset</option>
          {datasets.map((dataset) => (
            <option key={dataset.id} value={dataset.id}>
              {dataset.name}
            </option>
          ))}
        </select>
      </div>
      
      {/* Forecast Run Selection */}
      {selectedDataset && (
        <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            2. Select Forecast Runs (min 2)
          </h2>
          
          {forecastRuns.length === 0 ? (
            <p className="text-gray-600">
              No completed forecast runs found for this dataset.
            </p>
          ) : (
            <div className="space-y-2">
              {forecastRuns.map((run) => (
                <label
                  key={run.id}
                  className={`flex items-center p-4 border-2 rounded-lg cursor-pointer transition-colors ${
                    selectedRuns.includes(run.id)
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selectedRuns.includes(run.id)}
                    onChange={() => toggleRunSelection(run.id)}
                    className="text-primary-600 focus:ring-primary-500 mr-3"
                  />
                  <div className="flex-1 flex items-center justify-between">
                    <div>
                      <div className="font-medium text-gray-900">
                        {run.model_type.toUpperCase()}
                      </div>
                      <div className="text-sm text-gray-600">
                        Created {new Date(run.created_at).toLocaleString()}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-600">RMSE</div>
                      <div className="font-semibold text-gray-900">
                        {run.metrics?.RMSE?.toFixed(2) || 'N/A'}
                      </div>
                    </div>
                  </div>
                </label>
              ))}
            </div>
          )}
          
          <button
            onClick={handleCompare}
            disabled={selectedRuns.length < 2 || comparing}
            className="mt-4 w-full px-6 py-3 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {comparing ? (
              <span className="flex items-center justify-center">
                <Loader className="animate-spin h-5 w-5 mr-2" />
                Comparing...
              </span>
            ) : (
              <span className="flex items-center justify-center">
                <GitCompare className="h-5 w-5 mr-2" />
                Compare {selectedRuns.length} Models
              </span>
            )}
          </button>
        </div>
      )}
      
      {/* Comparison Results */}
      {comparisonResult && (
        <div className="space-y-6">
          <div className="bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-center">
              <CheckCircle className="h-5 w-5 text-green-600 mr-2" />
              <span className="text-green-800 font-medium">
                Comparison Complete!
              </span>
            </div>
          </div>
          
          {/* Metrics Comparison */}
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Metrics Comparison
            </h2>
            <div className="grid md:grid-cols-2 gap-6">
              {Object.entries(comparisonResult.metrics).map(([metric, values]) => (
                <Plot
                  key={metric}
                  data={[
                    {
                      x: comparisonResult.models,
                      y: values,
                      type: 'bar',
                      marker: {
                        color: comparisonResult.models.map((_, idx) => 
                          `hsl(${idx * 360 / comparisonResult.models.length}, 70%, 50%)`
                        ),
                      },
                    },
                  ]}
                  layout={{
                    title: metric,
                    autosize: true,
                    height: 300,
                    xaxis: { title: 'Model' },
                    yaxis: { title: metric },
                    showlegend: false,
                  }}
                  useResizeHandler
                  style={{ width: '100%' }}
                  config={{ responsive: true }}
                />
              ))}
            </div>
          </div>
          
          {/* Training Time Comparison */}
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Training Time Comparison
            </h2>
            <Plot
              data={[
                {
                  x: comparisonResult.models,
                  y: Object.values(comparisonResult.training_times),
                  type: 'bar',
                  marker: { color: '#8b5cf6' },
                },
              ]}
              layout={{
                autosize: true,
                height: 300,
                xaxis: { title: 'Model' },
                yaxis: { title: 'Training Time (seconds)' },
                showlegend: false,
              }}
              useResizeHandler
              style={{ width: '100%' }}
              config={{ responsive: true }}
            />
          </div>
          
          {/* Summary Table */}
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200 overflow-x-auto">
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Summary Table
            </h2>
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Model
                  </th>
                  {Object.keys(comparisonResult.metrics).map(metric => (
                    <th key={metric} className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                      {metric}
                    </th>
                  ))}
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Training Time
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {comparisonResult.models.map((model, idx) => (
                  <tr key={model} className={idx === 0 ? 'bg-green-50' : ''}>
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">
                      {model}
                      {idx === 0 && (
                        <span className="ml-2 px-2 py-1 text-xs bg-green-200 text-green-800 rounded">
                          Best
                        </span>
                      )}
                    </td>
                    {Object.entries(comparisonResult.metrics).map(([metric, values]) => (
                      <td key={metric} className="px-4 py-3 text-sm text-gray-900">
                        {values[idx]?.toFixed(4) || 'N/A'}
                      </td>
                    ))}
                    <td className="px-4 py-3 text-sm text-gray-900">
                      {comparisonResult.training_times[model]?.toFixed(2)}s
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      
      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
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