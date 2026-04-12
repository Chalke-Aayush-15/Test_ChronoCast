import { useParams, useNavigate } from 'react-router-dom';
import { useEffect, useState, useMemo } from 'react';
import { forecastAPI } from '../services/api';
import { AlertCircle, ArrowLeft, X, ExternalLink, BarChart2, LineChart, ScatterChart as ScatterChartIcon, Activity } from 'lucide-react';
import {
  LineChart as RechartsLineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  ComposedChart,
  Area,
  Cell,
  PieChart,
  Pie
} from 'recharts';
import { format, parseISO } from 'date-fns';

export default function ForecastResultsPage() {
  const { forecastId } = useParams();
  const navigate = useNavigate();
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [forecastData, setForecastData] = useState([]);
  const [loadingData, setLoadingData] = useState(false);
  const [activeTab, setActiveTab] = useState('overview');
  const [timeRange, setTimeRange] = useState('all');

  const fetchForecastData = async () => {
    try {
      setLoadingData(true);
      console.log('Fetching predictions for forecast ID:', forecastId);
      const response = await forecastAPI.getPredictions(forecastId, 1, 1000).catch(err => {
        console.error('Error fetching predictions:', err);
        if (err.response) {
          console.error('Response data:', err.response.data);
          console.error('Response status:', err.response.status);
          console.error('Response headers:', err.response.headers);
        } else if (err.request) {
          console.error('No response received:', err.request);
        } else {
          console.error('Error:', err.message);
        }
        throw err;
      });

      console.log('Predictions response:', response);
      
      if (!response.data || !Array.isArray(response.data.results)) {
        console.error('Unexpected response format:', response);
        throw new Error('Invalid response format from server');
      }

      const processedData = (response.data.results || []).map(item => ({
        ...item,
        date: item.date,
        dateFormatted: format(new Date(item.date), 'MMM dd, yyyy'),
        actual: parseFloat(item.actual) || null,
        predicted: parseFloat(item.predicted) || null,
        error: item.actual !== null && item.predicted !== null ? 
          Math.abs(parseFloat(item.actual) - parseFloat(item.predicted)) : null,
        errorPercentage: item.actual !== null && item.predicted !== null && parseFloat(item.actual) !== 0 
          ? (Math.abs(parseFloat(item.actual) - parseFloat(item.predicted)) / Math.abs(parseFloat(item.actual))) * 100 
          : null
      }));
      
      console.log('Processed data:', processedData);
      setForecastData(processedData);
      setShowModal(true);
    } catch (err) {
      setError('Failed to load forecast data');
      console.error(err);
    } finally {
      setLoadingData(false);
    }
  };

  useEffect(() => {
    const fetchForecast = async () => {
      try {
        const response = await forecastAPI.get(forecastId);
        setForecast(response.data);
        await fetchForecastData();
      } catch (err) {
        setError('Failed to load forecast results');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchForecast();
  }, [forecastId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border-l-4 border-red-400 p-4">
        <div className="flex">
          <div className="flex-shrink-0">
            <AlertCircle className="h-5 w-5 text-red-400" />
          </div>
          <div className="ml-3">
            <p className="text-sm text-red-700">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <button
        onClick={() => navigate(-1)}
        className="inline-flex items-center text-sm text-primary-600 hover:text-primary-800 dark:text-primary-400 dark:hover:text-primary-300 mb-6"
      >
        <ArrowLeft className="h-4 w-4 mr-2" />
        Back to Forecasts
      </button>
      
      <div className="bg-white dark:bg-gray-800 shadow rounded-lg p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">Forecast Results</h2>
          <div className="flex space-x-2">
            <button
              onClick={() => setActiveTab('overview')}
              className={`px-4 py-2 rounded-md ${activeTab === 'overview' ? 'bg-primary-600 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200'}`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab('details')}
              className={`px-4 py-2 rounded-md ${activeTab === 'details' ? 'bg-primary-600 text-white' : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200'}`}
            >
              Details
            </button>
          </div>
        </div>
        
        {forecast && (
          <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-l-4 border-blue-500">
                <div className="flex items-center">
                  <div className="p-2 rounded-full bg-blue-100 dark:bg-blue-800/50 mr-3">
                    <Activity className="h-5 w-5 text-blue-600 dark:text-blue-300" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-blue-700 dark:text-blue-200">Forecast ID</p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">{forecast.id}</p>
                  </div>
                </div>
              </div>
              
              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
                <div className="flex items-center">
                  <div className="p-2 rounded-full bg-green-100 dark:bg-green-800/50 mr-3">
                    <LineChart className="h-5 w-5 text-green-600 dark:text-green-300" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-green-700 dark:text-green-200">Status</p>
                    <p className="text-lg font-semibold text-green-600 dark:text-green-400">Completed</p>
                  </div>
                </div>
              </div>
            </div>

            {/* ChronoCast Multi-Model Details */}
            {forecast.model_type === 'chronocast' && forecast.metrics?.model_comparison && (
              <div className="mb-8">
                <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-4">ChronoCast Multi-Model Analysis</h4>
                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-lg border border-blue-200 dark:border-blue-800">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                      <h5 className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-2">Models Trained</h5>
                      <div className="flex flex-wrap gap-2">
                        {forecast.metrics.model_comparison.models_trained?.map((model, index) => (
                          <span key={index} className="px-3 py-1 bg-blue-100 dark:bg-blue-800/50 text-blue-700 dark:text-blue-300 rounded-full text-sm font-medium">
                            {model.toUpperCase()}
                          </span>
                        ))}
                      </div>
                    </div>
                    
                    <div>
                      <h5 className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-2">Best Performing Model</h5>
                      <div className="flex items-center">
                        <span className="px-3 py-1 bg-green-100 dark:bg-green-800/50 text-green-700 dark:text-green-300 rounded-full text-sm font-semibold">
                          {forecast.metrics.model_comparison.best_model?.toUpperCase() || 'ENSEMBLE'}
                        </span>
                      </div>
                    </div>
                    
                    <div>
                      <h5 className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-2">All Model Metrics</h5>
                      <div className="space-y-1">
                        {Object.entries(forecast.metrics.model_comparison.all_metrics || {}).map(([model, metrics]) => (
                          <div key={model} className="flex justify-between text-sm">
                            <span className="text-gray-700 dark:text-gray-300 font-medium">{model.toUpperCase()}:</span>
                            <span className="text-gray-900 dark:text-gray-100">
                              MAE: {typeof metrics?.test_mae === 'number' ? metrics.test_mae.toFixed(2) : 'N/A'}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Performance Metrics */}
            <div>
              <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Performance Metrics</h4>
              {forecast.metrics && (
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  {Object.entries(forecast.metrics).map(([key, value]) => {
                    // Skip complex objects that can't be rendered directly
                    if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                      return null;
                    }
                    
                    let bgColor = 'bg-gray-100 dark:bg-gray-700';
                    let textColor = 'text-gray-900 dark:text-white';
                    let icon = null;
                    let displayValue = value;
                    
                    // Handle different metric types
                    if (key.includes('mae') || key.includes('mse') || key.includes('rmse')) {
                      const score = parseFloat(value);
                      displayValue = typeof value === 'number' ? value.toFixed(4) : value;
                      if (score < 0.1) {
                        bgColor = 'bg-green-100 dark:bg-green-900/30';
                        textColor = 'text-green-700 dark:text-green-400';
                      } else if (score < 0.3) {
                        bgColor = 'bg-yellow-100 dark:bg-yellow-900/30';
                        textColor = 'text-yellow-700 dark:text-yellow-400';
                      } else {
                        bgColor = 'bg-red-100 dark:bg-red-900/30';
                        textColor = 'text-red-700 dark:text-red-400';
                      }
                      icon = <BarChart2 className="h-5 w-5 mr-2" />;
                    } else if (key.includes('r2') || key.includes('score')) {
                      const score = parseFloat(value);
                      displayValue = typeof value === 'number' ? value.toFixed(4) : value;
                      if (score >= 0.9) {
                        bgColor = 'bg-green-100 dark:bg-green-900/30';
                        textColor = 'text-green-700 dark:text-green-400';
                      } else if (score >= 0.7) {
                        bgColor = 'bg-yellow-100 dark:bg-yellow-900/30';
                        textColor = 'text-yellow-700 dark:text-yellow-400';
                      } else {
                        bgColor = 'bg-red-100 dark:bg-red-900/30';
                        textColor = 'text-red-700 dark:text-red-400';
                      }
                      icon = <LineChart className="h-5 w-5 mr-2" />;
                    } else if (key.includes('model_comparison')) {
                      // Skip model_comparison as it's a complex object
                      return null;
                    } else {
                      icon = <Activity className="h-5 w-5 mr-2" />;
                      displayValue = typeof value === 'number' ? value.toFixed(4) : String(value);
                    }
                    
                    return (
                      <div 
                        key={key} 
                        className={`${bgColor} p-4 rounded-lg transition-all duration-200 hover:shadow-md`}
                      >
                        <div className="flex items-center">
                          {icon}
                          <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                            {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                          </h4>
                        </div>
                        <p className={`text-xl font-semibold mt-1 ${textColor}`}>
                          {displayValue}
                        </p>
                      </div>
                    );
                  }).filter(Boolean)}
                </div>
              )}
            </div>

            <div className="mt-6">
              <button
                onClick={fetchForecastData}
                className="flex items-center px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors"
                disabled={loadingData}
              >
                {loadingData ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Loading...
                  </>
                ) : (
                  <>
                    <ExternalLink className="h-4 w-4 mr-2" />
                    View Full Forecast Data
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Forecast Data Modal */}
      {showModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[80vh] flex flex-col">
            <div className="flex justify-between items-center p-4 border-b border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Forecast Data</h3>
              <button
                onClick={() => setShowModal(false)}
                className="text-gray-400 hover:text-gray-500 dark:hover:text-gray-300"
              >
                <X className="h-6 w-6" />
              </button>
            </div>
            <div className="flex-1 overflow-auto p-4">
              {loadingData ? (
                <div className="flex items-center justify-center h-32">
                  <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-primary-600"></div>
                </div>
              ) : forecastData.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead className="bg-gray-50 dark:bg-gray-700">
                      <tr>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Date
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Actual
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Predicted
                        </th>
                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                          Error
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                      {forecastData.map((row, index) => (
                        <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-200">
                            {row.date}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-200">
                            {row.actual?.toFixed(2) || 'N/A'}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-200">
                            {row.predicted?.toFixed(2) || 'N/A'}
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-200">
                            {row.error ? row.error.toFixed(2) : 'N/A'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                  No forecast data available
                </div>
              )}
            </div>
            <div className="p-4 border-t border-gray-200 dark:border-gray-700 flex justify-end">
              <button
                onClick={() => setShowModal(false)}
                className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-md hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
