import { useState, useRef, useEffect } from 'react';
import { Loader2, Youtube, BarChart2, ThumbsUp, Eye, Calendar, TrendingUp, CalendarDays } from 'lucide-react';
import { motion } from 'framer-motion';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler } from 'chart.js';
import { Line } from 'react-chartjs-2';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Helper function to generate dummy historical data
const generateHistoricalData = (currentViews, days = 30) => {
  const data = [];
  const today = new Date();
  
  // Generate data points for the last 'days' days
  for (let i = days; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(today.getDate() - i);
    
    // Create a base growth pattern (more views on weekends, less on weekdays)
    const dayOfWeek = date.getDay();
    const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
    const randomFactor = 0.8 + Math.random() * 0.4; // 0.8 to 1.2
    const dayFactor = isWeekend ? 1.3 : 0.9;
    
    // Calculate views for this day (more views as we get closer to today)
    const progress = i / days;
    const dayViews = Math.round(
      (currentViews * 0.2 * progress + currentViews * 0.05) * 
      randomFactor * 
      dayFactor
    );
    
    data.push({
      date: date.toISOString().split('T')[0],
      views: Math.max(1, dayViews)
    });
  }
  
  // Calculate cumulative views
  let cumulative = 0;
  return data.map(day => {
    cumulative += day.views;
    return {
      ...day,
      cumulativeViews: Math.min(cumulative, currentViews)
    };
  });
};

// Simple forecasting function (linear regression)
const generateForecast = (historicalData, daysToForecast = 30) => {
  if (historicalData.length < 2) return [];
  
  const n = historicalData.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
  
  // Calculate necessary sums for linear regression
  historicalData.forEach((data, index) => {
    sumX += index;
    sumY += data.cumulativeViews;
    sumXY += index * data.cumulativeViews;
    sumX2 += index * index;
  });
  
  // Calculate slope (m) and intercept (b) for y = mx + b
  const m = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const b = (sumY - m * sumX) / n;
  
  // Generate forecast
  const forecast = [];
  const lastDate = new Date(historicalData[historicalData.length - 1].date);
  
  for (let i = 1; i <= daysToForecast; i++) {
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(lastDate.getDate() + i);
    
    const x = n + i - 1;
    const y = m * x + b;
    
    // Add some randomness to make it more realistic
    const randomFactor = 0.9 + Math.random() * 0.2;
    const forecastValue = Math.round(y * randomFactor);
    
    forecast.push({
      date: forecastDate.toISOString().split('T')[0],
      cumulativeViews: Math.max(0, forecastValue),
      isForecast: true
    });
  }
  
  return forecast;
};

const YouTubeAnalytics = () => {
  const [videoUrl, setVideoUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isForecasting, setIsForecasting] = useState(false);
  const [videoData, setVideoData] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [forecastData, setForecastData] = useState([]);
  const [error, setError] = useState('');
  const [forecastDays, setForecastDays] = useState(30);
  const chartRef = useRef(null);
  
  const API_KEY = 'AIzaSyBp6Wj5YPdGe-GY9gVmO1BRzCQqDsKVp9A';
  
  const extractVideoId = (url) => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[2].length === 11) ? match[2] : null;
  };

  const fetchVideoData = async () => {
    const videoId = extractVideoId(videoUrl);
    if (!videoId) {
      setError('Please enter a valid YouTube URL');
      return;
    }
    
    setIsLoading(true);
    setError('');
    
    try {
      // First get video statistics
      const statsResponse = await fetch(
        `https://www.googleapis.com/youtube/v3/videos?part=statistics,snippet&id=${videoId}&key=${API_KEY}`
      );
      
      if (!statsResponse.ok) {
        throw new Error('Failed to fetch video data');
      }
      
      const data = await statsResponse.json();
      
      if (data.items && data.items.length > 0) {
        const video = data.items[0];
        setVideoData({
          title: video.snippet.title,
          thumbnail: video.snippet.thumbnails.high.url,
          channelTitle: video.snippet.channelTitle,
          publishedAt: new Date(video.snippet.publishedAt).toLocaleDateString(),
          statistics: {
            viewCount: parseInt(video.statistics.viewCount).toLocaleString(),
            likeCount: parseInt(video.statistics.likeCount).toLocaleString(),
            commentCount: parseInt(video.statistics.commentCount).toLocaleString(),
          }
        });
      } else {
        setError('No video found with this URL');
      }
    } catch (err) {
      console.error('Error fetching YouTube data:', err);
      setError('Failed to fetch video data. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    fetchVideoData();
  };
  
  const handleGenerateForecast = () => {
    if (!videoData) return;
    
    setIsForecasting(true);
    
    // Simulate API call delay
    setTimeout(() => {
      const forecast = generateForecast(historicalData, forecastDays);
      setForecastData(forecast);
      setIsForecasting(false);
    }, 1000);
  };
  
  // Generate chart data
  const getChartData = () => {
    if (historicalData.length === 0) return { labels: [], datasets: [] };
    
    const labels = [
      ...historicalData.map(d => d.date.substring(5)), // Show MM-DD format
      ...forecastData.map(d => d.date.substring(5))
    ];
    
    return {
      labels,
      datasets: [
        {
          label: 'Historical Views',
          data: [
            ...historicalData.map(d => d.cumulativeViews),
            ...Array(forecastData.length).fill(null)
          ],
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.5)',
          borderWidth: 2,
          tension: 0.4,
          fill: false,
          pointRadius: 3,
          pointHoverRadius: 5
        },
        {
          label: 'Forecasted Views',
          data: [
            ...Array(historicalData.length - 1).fill(null),
            historicalData[historicalData.length - 1]?.cumulativeViews,
            ...forecastData.map(d => d.cumulativeViews)
          ],
          borderColor: 'rgb(16, 185, 129)',
          backgroundColor: 'rgba(16, 185, 129, 0.5)',
          borderWidth: 2,
          borderDash: [5, 5],
          tension: 0.4,
          fill: false,
          pointRadius: 0,
          pointHoverRadius: 5
        }
      ]
    };
  };
  
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
              label += ': ';
            }
            if (context.parsed.y !== null) {
              label += new Intl.NumberFormat('en-US').format(context.parsed.y) + ' views';
            }
            return label;
          }
        }
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          callback: function(value) {
            if (value >= 1000000) {
              return (value / 1000000).toFixed(1) + 'M';
            } else if (value >= 1000) {
              return (value / 1000).toFixed(0) + 'K';
            }
            return value;
          }
        }
      }
    }
  };
  
  // Update historical data when video data changes
  useEffect(() => {
    if (videoData && videoData.statistics) {
      const views = parseInt(videoData.statistics.viewCount.replace(/,/g, ''));
      const historical = generateHistoricalData(views);
      setHistoricalData(historical);
      setForecastData([]);
    }
  }, [videoData]);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden border border-gray-200 dark:border-gray-700 mt-10">
      <div className="p-6 md:p-8">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-6">
          <div className="flex items-center mb-4 md:mb-0">
            <Youtube className="w-8 h-8 text-red-600 mr-3" />
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">YouTube Video Analytics</h2>
          </div>
          {videoData && (
            <div className="flex items-center space-x-2">
              <div className="flex items-center">
                <CalendarDays className="w-4 h-4 text-gray-500 mr-2" />
                <select 
                  value={forecastDays}
                  onChange={(e) => setForecastDays(Number(e.target.value))}
                  className="bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                >
                  <option value={7}>7 days forecast</option>
                  <option value={14}>14 days forecast</option>
                  <option value={30}>30 days forecast</option>
                  <option value={60}>60 days forecast</option>
                </select>
              </div>
              <button
                onClick={handleGenerateForecast}
                disabled={isForecasting || !historicalData.length}
                className={`px-4 py-1.5 rounded-md text-sm font-medium ${
                  isForecasting || !historicalData.length
                    ? 'bg-gray-200 dark:bg-gray-700 text-gray-500 cursor-not-allowed'
                    : 'bg-primary-600 hover:bg-primary-700 text-white'
                }`}
              >
                {isForecasting ? 'Generating...' : 'Generate Forecast'}
              </button>
            </div>
          )}
        </div>
        
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <input
                type="text"
                value={videoUrl}
                onChange={(e) => setVideoUrl(e.target.value)}
                placeholder="Paste YouTube video URL"
                className="w-full px-4 py-3 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                required
              />
            </div>
            <button
              type="submit"
              disabled={isLoading}
              className="px-6 py-3 bg-red-600 hover:bg-red-700 text-white font-medium rounded-lg transition-colors duration-200 flex items-center justify-center"
            >
              {isLoading ? (
                <>
                  <Loader2 className="animate-spin mr-2 h-5 w-5" />
                  Analyzing...
                </>
              ) : (
                'Analyze Video'
              )}
            </button>
          </div>
          {error && <p className="mt-2 text-sm text-red-500">{error}</p>}
        </form>
        
        {videoData && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-gray-50 dark:bg-gray-700 rounded-xl p-6 mt-6"
          >
            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">{videoData.title}</h3>
                <p className="text-gray-600 dark:text-gray-300 mb-4">
                  Channel: <span className="text-gray-800 dark:text-white font-medium">{videoData.channelTitle}</span>
                </p>
                <p className="text-gray-600 dark:text-gray-300 mb-6">
                  Published: <span className="text-gray-800 dark:text-white font-medium">{videoData.publishedAt}</span>
                </p>
                
                <div className="space-y-3">
                  <div className="flex items-center">
                    <Eye className="w-5 h-5 text-gray-500 mr-2" />
                    <span className="text-gray-700 dark:text-gray-300">Views: </span>
                    <span className="ml-2 font-medium text-gray-900 dark:text-white">{videoData.statistics.viewCount}</span>
                  </div>
                  <div className="flex items-center">
                    <ThumbsUp className="w-5 h-5 text-gray-500 mr-2" />
                    <span className="text-gray-700 dark:text-gray-300">Likes: </span>
                    <span className="ml-2 font-medium text-gray-900 dark:text-white">{videoData.statistics.likeCount}</span>
                  </div>
                  <div className="flex items-center">
                    <svg className="w-5 h-5 text-gray-500 mr-2" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
                    </svg>
                    <span className="text-gray-700 dark:text-gray-300">Comments: </span>
                    <span className="ml-2 font-medium text-gray-900 dark:text-white">{videoData.statistics.commentCount}</span>
                  </div>
                </div>
              </div>
              
              <div className="bg-white dark:bg-gray-800 p-4 rounded-lg border border-gray-200 dark:border-gray-600">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="font-medium text-gray-900 dark:text-white">
                    {forecastData.length > 0 ? 'Performance Forecast' : 'Performance Analysis'}
                  </h4>
                  <TrendingUp className="w-5 h-5 text-blue-500" />
                </div>
                
                <div className="h-80 w-full">
                  {historicalData.length > 0 ? (
                    <div className="h-full">
                      <Line data={getChartData()} options={chartOptions} ref={chartRef} />
                      {forecastData.length > 0 && (
                        <div className="mt-4 text-sm text-gray-500 dark:text-gray-400">
                          <p>Forecast for next {forecastDays} days:</p>
                          <p className="font-medium text-green-600 dark:text-green-400">
                            Projected views: {new Intl.NumberFormat('en-US').format(
                              forecastData[forecastData.length - 1].cumulativeViews
                            )} (+
                            {Math.round(
                              ((forecastData[forecastData.length - 1].cumulativeViews - historicalData[historicalData.length - 1].cumulativeViews) / 
                              historicalData[historicalData.length - 1].cumulativeViews) * 100
                            )}%)
                          </p>
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="h-full flex items-center justify-center bg-gray-50 dark:bg-gray-700 rounded-lg">
                      <p className="text-gray-500 dark:text-gray-400 text-center p-4">
                        {isLoading ? 'Loading video data...' : 'Video data will be displayed here'}
                      </p>
                    </div>
                  )}
                </div>
                
                {historicalData.length > 0 && forecastData.length === 0 && (
                  <div className="mt-4 text-center">
                    <p className="text-sm text-gray-500 dark:text-gray-400 mb-3">
                      Generate a forecast to see projected view growth
                    </p>
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default YouTubeAnalytics;
