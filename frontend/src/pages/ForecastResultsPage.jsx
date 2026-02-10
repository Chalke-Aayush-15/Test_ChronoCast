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
          // The request was made and the server responded with a status code
          // that falls out of the range of 2xx
          console.error('Response data:', err.response.data);
          console.error('Response status:', err.response.status);
          console.error('Response headers:', err.response.headers);
        } else if (err.request) {
          // The request was made but no response was received
          console.error('No response received:', err.request);
        } else {
          // Something happened in setting up the request that triggered an Error
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
      setShowModal(true); // Show the modal after data is loaded
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
        // Fetch the forecast data after getting the forecast
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
              
              {/* {forecast.metrics && (
                <div className="bg-amber-50 dark:bg-amber-900/20 p-4 rounded-lg border-l-4 border-amber-500">
                  <div className="flex items-center">
                    <div className="p-2 rounded-full bg-amber-100 dark:bg-amber-800/50 mr-3">
                      <ScatterChart className="h-5 w-5 text-amber-600 dark:text-amber-300" />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-amber-700 dark:text-amber-200">R² Score</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-white">
                        {forecast.metrics.r2 ? forecast.metrics.r2.toFixed(4) : 'N/A'}
                      </p>
                    </div>
                  </div>
                </div>
              )} */}
            </div>

            {/* Time Range Selector */}
            <div className="flex justify-end">
              <div className="inline-flex rounded-md shadow-sm" role="group">
                {['week', 'month', 'quarter', 'all'].map((range) => (
                  <button
                    key={range}
                    type="button"
                    onClick={() => setTimeRange(range)}
                    className={`px-4 py-2 text-sm font-medium ${
                      timeRange === range
                        ? 'bg-primary-600 text-white'
                        : 'bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-600'
                    } border border-gray-200 dark:border-gray-600 ${
                      range === 'week' ? 'rounded-l-lg' : ''
                    } ${range === 'all' ? 'rounded-r-lg' : ''} ${
                      range !== 'week' && range !== 'all' ? 'border-l-0' : ''
                    }`}
                  >
                    {range.charAt(0).toUpperCase() + range.slice(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Main Chart */}
            <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
              <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Forecast vs Actual</h3>
              <div className="h-80">
                {forecastData.length > 0 ? (
                  <div style={{ width: '100%', height: '100%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart
                        data={forecastData}
                        margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis 
                          dataKey="dateFormatted" 
                          tick={{ fontSize: 12 }}
                          tickFormatter={(value) => value}
                        />
                        <YAxis />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: 'white',
                            border: '1px solid #e5e7eb',
                            borderRadius: '0.5rem',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)'
                          }}
                          formatter={(value, name) => [value, name === 'actual' ? 'Actual' : 'Predicted']}
                          labelFormatter={(label) => `Date: ${label}`}
                        />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="actual" 
                          name="Actual" 
                          stroke="#3b82f6" 
                          dot={false} 
                          strokeWidth={2} 
                          activeDot={{ r: 6 }} 
                        />
                        <Line 
                          type="monotone" 
                          dataKey="predicted" 
                          name="Predicted" 
                          stroke="#10b981" 
                          strokeDasharray="5 5" 
                          strokeWidth={2} 
                          dot={false} 
                        />
                        <Area 
                          type="monotone" 
                          dataKey="predicted" 
                          fill="#a7f3d0" 
                          fillOpacity={0.2} 
                          stroke="none" 
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-500">
                    No data available for the selected time range
                  </div>
                )}
              </div>
            </div>

            {/* Additional Visualizations */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Error Distribution */}
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Error Distribution</h3>
                <div className="h-64">
                  {forecastData.filter(d => d.error !== null).length > 0 ? (
                    <div style={{ width: '100%', height: '100%' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={forecastData.filter(d => d.error !== null).slice(0, 20)}
                        margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis dataKey="dateFormatted" tick={{ fontSize: 10 }} />
                        <YAxis />
                        <Tooltip 
                          formatter={(value) => [value.toFixed(2), 'Error']}
                          labelFormatter={(label) => `Date: ${label}`}
                        />
                        <Bar dataKey="error" fill="#f59e0b" name="Error" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-500">
                      No error data available
                    </div>
                  )}
                </div>
              </div>

              {/* Scatter Plot */}
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Actual vs Predicted</h3>
                <div className="h-64">
                  {forecastData.filter(d => d.actual !== null && d.predicted !== null).length > 0 ? (
                    <div style={{ width: '100%', height: '100%' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <ScatterChart
                          margin={{ top: 20, right: 20, bottom: 20, left: 20 }}
                        >
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis 
                          type="number" 
                          dataKey="actual" 
                          name="Actual" 
                          domain={['auto', 'auto']} 
                          label={{ value: 'Actual', position: 'insideBottomRight', offset: -5 }}
                        />
                        <YAxis 
                          type="number" 
                          dataKey="predicted" 
                          name="Predicted" 
                          domain={['auto', 'auto']}
                          label={{ value: 'Predicted', angle: -90, position: 'insideLeft' }}
                        />
                        <Tooltip 
                          cursor={{ strokeDasharray: '3 3' }} 
                          formatter={(value, name) => [value, name === 'actual' ? 'Actual' : 'Predicted']}
                        />
                        <Scatter 
                          name="Actual vs Predicted" 
                          data={forecastData.filter(d => d.actual !== null && d.predicted !== null)} 
                          fill="#8884d8"
                        >
                          {forecastData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.errorPercentage > 10 ? '#ef4444' : '#8884d8'} />
                          ))}
                        </Scatter>
                        <Legend />
                        </ScatterChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <div className="flex items-center justify-center h-full text-gray-500">
                      No data available for scatter plot
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Combined Performance Metrics and Forecast Results */}
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white">Forecast Analysis</h3>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                  Comprehensive performance metrics and forecast results
                </p>
              </div>
              
              <div className="p-6">
                {/* Forecast Summary */}
                {/* <div className="mb-8">
                  <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Forecast Overview</h4>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border-l-4 border-blue-500">
                      <p className="text-sm font-medium text-blue-700 dark:text-blue-300">Forecast Period</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-white">
                        {forecastData.length > 0 ? (
                          `${format(new Date(forecastData[0].date), 'MMM d, yyyy')} - ${format(new Date(forecastData[forecastData.length - 1].date), 'MMM d, yyyy')}`
                        ) : 'N/A'}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {forecastData.length} data points
                      </p>
                    </div>
                    
                    <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg border-l-4 border-green-500">
                      <p className="text-sm font-medium text-green-700 dark:text-green-300">Average Prediction</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-white">
                        {forecastData.length > 0 ? (
                          (forecastData.reduce((sum, item) => sum + (item.predicted || 0), 0) / forecastData.length).toFixed(2)
                        ) : 'N/A'}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        Mean predicted value
                      </p>
                    </div>
                    
                    <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg border-l-4 border-purple-500">
                      <p className="text-sm font-medium text-purple-700 dark:text-purple-300">Confidence</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-white">
                        {forecast.metrics?.r2 ? (
                          forecast.metrics.r2 >= 0.8 ? 'High' : 
                          forecast.metrics.r2 >= 0.5 ? 'Medium' : 'Low'
                        ) : 'N/A'}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        Based on R² score
                      </p>
                    </div>
                  </div>
                </div> */}
                
                {/* Performance Metrics */}
                <div>
                  <h4 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Performance Metrics</h4>
                  {forecast.metrics && (
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                      {Object.entries(forecast.metrics).map(([key, value]) => {
                        let bgColor = 'bg-gray-100 dark:bg-gray-700';
                        let textColor = 'text-gray-900 dark:text-white';
                        let icon = null;
                        
                        if (key.includes('mae') || key.includes('mse') || key.includes('rmse')) {
                          const score = parseFloat(value);
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
                        } else {
                          icon = <Activity className="h-5 w-5 mr-2" />;
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
                              {typeof value === 'number' ? value.toFixed(4) : value}
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
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