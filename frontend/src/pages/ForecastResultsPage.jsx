import { useParams, useNavigate } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { forecastAPI } from '../services/api';
import { AlertCircle, ArrowLeft } from 'lucide-react';

export default function ForecastResultsPage() {
  const { forecastId } = useParams();
  const navigate = useNavigate();
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchForecast = async () => {
      try {
        const response = await forecastAPI.get(forecastId);
        setForecast(response.data);
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
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Forecast Results</h2>
        
        {forecast && (
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                <h3 className="font-medium text-gray-700 dark:text-gray-200">Forecast ID</h3>
                <p className="text-gray-900 dark:text-white">{forecast.id}</p>
              </div>
              <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                <h3 className="font-medium text-gray-700 dark:text-gray-200">Status</h3>
                <p className="text-green-600 dark:text-green-400 font-medium">Completed</p>
              </div>
            </div>
            
            {forecast.metrics && (
              <div className="mt-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-3">Performance Metrics</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {Object.entries(forecast.metrics).map(([key, value]) => (
                    <div key={key} className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400">
                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </h4>
                      <p className="text-lg font-semibold text-gray-900 dark:text-white">
                        {typeof value === 'number' ? value.toFixed(4) : value}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
