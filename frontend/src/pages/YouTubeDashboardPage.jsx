import { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Youtube, BarChart2, ThumbsUp, Eye, MessageSquare, Clock, Calendar, TrendingUp, ArrowLeft, Share2, Heart, MessageCircle } from 'lucide-react';
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

const YouTubeDashboardPage = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const [isForecasting, setIsForecasting] = useState(false);
  const [forecastDays, setForecastDays] = useState(30);
  const [videoData, setVideoData] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [forecastData, setForecastData] = useState([]);
  const [activeTab, setActiveTab] = useState('overview');
  const [relatedVideos, setRelatedVideos] = useState([]);
  const chartRef = useRef(null);
  
  // Mock data for top performing content
  const [topPerformingContent, setTopPerformingContent] = useState([
    {
      id: 1,
      title: 'How to prepare for final exams',
      type: 'Reel',
      platform: 'Instagram',
      likes: 1245,
      comments: 234,
      shares: 45,
      views: 12500,
      engagementRate: 12.2
    },
    {
      id: 2,
      title: 'Study tips and tricks',
      type: 'Carousel',
      platform: 'Instagram',
      likes: 980,
      comments: 145,
      shares: 32,
      views: 8900,
      engagementRate: 11.3
    },
    {
      id: 3,
      title: 'Time management for students',
      type: 'Video',
      platform: 'YouTube',
      likes: 2450,
      comments: 320,
      shares: 78,
      views: 18700,
      engagementRate: 15.2
    },
    {
      id: 4,
      title: 'Best study spots on campus',
      type: 'Static',
      platform: 'Facebook',
      likes: 780,
      comments: 98,
      shares: 23,
      views: 6500,
      engagementRate: 13.8
    },
    {
      id: 5,
      title: 'Exam preparation guide',
      type: 'Document',
      platform: 'LinkedIn',
      likes: 1560,
      comments: 210,
      shares: 67,
      views: 10200,
      engagementRate: 17.9
    }
  ]);

  // Extract video data from location state and fetch related videos
  useEffect(() => {
    if (location.state?.videoData) {
      const videoData = location.state.videoData;
      setVideoData(videoData);
      
      // Generate historical data based on video stats
      const viewCount = videoData.statistics.viewCount;
      const views = typeof viewCount === 'string' 
        ? parseInt(viewCount.replace(/[^0-9]/g, ''), 10) 
        : Math.floor(Number(viewCount));
      const historical = generateHistoricalData(views);
      setHistoricalData(historical);
      
      // Simulate fetching related videos from the same channel
      // In a real app, this would be an API call to get related videos
      const relatedVideos = [
        {
          id: 'related1',
          title: videoData.snippet.title.includes('Final Exams') 
            ? '5 Study Tips for Final Exams' 
            : 'How to Ace Your Exams',
          viewCount: Math.round(views * 0.8),
          thumbnail: videoData.snippet.thumbnails?.high?.url || ''
        },
        {
          id: 'related2',
          title: videoData.snippet.title.includes('Final Exams') 
            ? 'Last Minute Revision Strategy' 
            : 'Time Management for Students',
          viewCount: Math.round(views * 0.6),
          thumbnail: videoData.snippet.thumbnails?.high?.url || ''
        }
      ];
      
      setRelatedVideos(relatedVideos);
      setIsLoading(false);
    } else {
      // If no video data, redirect back to home
      navigate('/');
    }
  }, [location, navigate]);

  // Generate historical data (simulated)
  const generateHistoricalData = (currentViews, days = 30) => {
    const data = [];
    const today = new Date();
    
    for (let i = days; i >= 0; i--) {
      const date = new Date(today);
      date.setDate(today.getDate() - i);
      
      const dayOfWeek = date.getDay();
      const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
      const randomFactor = 0.8 + Math.random() * 0.4;
      const dayFactor = isWeekend ? 1.3 : 0.9;
      
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
    
    let cumulative = 0;
    return data.map(day => {
      cumulative += day.views;
      return {
        ...day,
        cumulativeViews: Math.min(cumulative, currentViews)
      };
    });
  };

  // Generate forecast
  const generateForecast = (historicalData, daysToForecast = 30) => {
    if (historicalData.length < 2) return [];
    
    const n = historicalData.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    
    historicalData.forEach((data, index) => {
      sumX += index;
      sumY += data.cumulativeViews;
      sumXY += index * data.cumulativeViews;
      sumX2 += index * index;
    });
    
    const m = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const b = (sumY - m * sumX) / n;
    
    const forecast = [];
    const lastDate = new Date(historicalData[historicalData.length - 1].date);
    
    for (let i = 1; i <= daysToForecast; i++) {
      const forecastDate = new Date(lastDate);
      forecastDate.setDate(lastDate.getDate() + i);
      
      const x = n + i - 1;
      const y = m * x + b;
      
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

  const handleGenerateForecast = () => {
    setIsForecasting(true);
    
    setTimeout(() => {
      const forecast = generateForecast(historicalData, forecastDays);
      setForecastData(forecast);
      setIsForecasting(false);
    }, 1000);
  };

  // Prepare chart data
  const getChartData = () => {
    if (historicalData.length === 0) return { labels: [], datasets: [] };
    
    const labels = [
      ...historicalData.map(d => d.date.substring(5)),
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
            if (label) label += ': ';
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

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-50 dark:bg-gray-900">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Header */}
      <header className="bg-white dark:bg-gray-800 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <button
                onClick={() => navigate('/')}
                className="mr-4 p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                <ArrowLeft className="h-5 w-5 text-gray-600 dark:text-gray-300" />
              </button>
              <h1 className="text-xl font-bold text-gray-900 dark:text-white">YouTube Analytics</h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="hidden md:flex items-center space-x-1">
                <span className="px-3 py-1 text-sm font-medium rounded-full bg-primary-100 text-primary-800 dark:bg-primary-900/30 dark:text-primary-400">
                  {videoData?.snippet?.channelTitle}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Video Info */}
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden mb-6">
          <div className="p-6">
            <div className="flex flex-col md:flex-row gap-6">
              <div className="w-full md:w-1/3 lg:w-1/4">
                <div className="relative aspect-video bg-gray-100 dark:bg-gray-700 rounded-lg overflow-hidden">
                  {videoData?.snippet?.thumbnails?.high?.url ? (
                    <img
                      src={videoData.snippet.thumbnails.high.url}
                      alt={videoData.snippet.title}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center bg-gray-200 dark:bg-gray-700">
                      <Youtube className="h-12 w-12 text-red-600" />
                    </div>
                  )}
                </div>
              </div>
              
              <div className="flex-1">
                <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-2">
                  {videoData?.snippet?.title || 'Video Title'}
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  Published on {videoData?.snippet?.publishedAt ? new Date(videoData.snippet.publishedAt).toLocaleDateString() : 'N/A'}
                </p>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                  <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                    <div className="flex items-center">
                      <Eye className="h-5 w-5 text-gray-500 mr-2" />
                      <span className="text-sm text-gray-600 dark:text-gray-300">Views</span>
                    </div>
                    <p className="text-xl font-semibold text-gray-900 dark:text-white mt-1">
                      {videoData?.statistics?.viewCount ? new Intl.NumberFormat('en-US').format(videoData.statistics.viewCount) : 'N/A'}
                    </p>
                  </div>
                  
                  <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                    <div className="flex items-center">
                      <ThumbsUp className="h-5 w-5 text-gray-500 mr-2" />
                      <span className="text-sm text-gray-600 dark:text-gray-300">Likes</span>
                    </div>
                    <p className="text-xl font-semibold text-gray-900 dark:text-white mt-1">
                      {videoData?.statistics?.likeCount !== undefined ? new Intl.NumberFormat('en-US').format(videoData.statistics.likeCount) : 'N/A'}
                    </p>
                  </div>
                  
                  <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                    <div className="flex items-center">
                      <MessageSquare className="h-5 w-5 text-gray-500 mr-2" />
                      <span className="text-sm text-gray-600 dark:text-gray-300">Comments</span>
                    </div>
                    <p className="text-xl font-semibold text-gray-900 dark:text-white mt-1">
                      {videoData?.statistics?.commentCount !== undefined ? new Intl.NumberFormat('en-US').format(videoData.statistics.commentCount) : 'N/A'}
                    </p>
                  </div>
                  
                  <div className="bg-gray-50 dark:bg-gray-700/50 p-4 rounded-lg">
                    <div className="flex items-center">
                      <Clock className="h-5 w-5 text-gray-500 mr-2" />
                      <span className="text-sm text-gray-600 dark:text-gray-300">Duration</span>
                    </div>
                    <p className="text-xl font-semibold text-gray-900 dark:text-white mt-1">
                      {videoData?.contentDetails?.durationFormatted || 'N/A'}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        {/* Tabs */}
        <div className="border-b border-gray-200 dark:border-gray-700 mb-6">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => setActiveTab('overview')}
              className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'overview'
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-200'
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab('content')}
              className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'content'
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-200'
              }`}
            >
              Content
            </button>
            <button
              onClick={() => setActiveTab('audience')}
              className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'audience'
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-200'
              }`}
            >
              Audience
            </button>
            <button
              onClick={() => setActiveTab('research')}
              className={`whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm ${
                activeTab === 'research'
                  ? 'border-primary-500 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-200'
              }`}
            >
              Research
            </button>
          </nav>
        </div>
        
        {/* Main Content */}
        {activeTab === 'content' ? (
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
            <div className="p-6">
              <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-6">Top Performing Content</h2>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                  <thead className="bg-gray-50 dark:bg-gray-700">
                    <tr>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Content
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Platform
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        <div className="flex items-center">
                          <ThumbsUp className="h-4 w-4 mr-1" />
                          <span>Likes</span>
                        </div>
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        <div className="flex items-center">
                          <MessageCircle className="h-4 w-4 mr-1" />
                          <span>Comments</span>
                        </div>
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        <div className="flex items-center">
                          <Share2 className="h-4 w-4 mr-1" />
                          <span>Shares</span>
                        </div>
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        <div className="flex items-center">
                          <Eye className="h-4 w-4 mr-1" />
                          <span>Views</span>
                        </div>
                      </th>
                      <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-300 uppercase tracking-wider">
                        Eng. Rate
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white dark:bg-gray-800 divide-y divide-gray-200 dark:divide-gray-700">
                    {topPerformingContent.map((content) => (
                      <tr key={content.id} className="hover:bg-gray-50 dark:hover:bg-gray-700/50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center">
                            <div className="flex-shrink-0 h-10 w-10 rounded-md bg-blue-100 dark:bg-blue-900/50 flex items-center justify-center text-blue-600 dark:text-blue-300 font-medium">
                              {content.type.charAt(0)}
                            </div>
                            <div className="ml-4">
                              <div className="text-sm font-medium text-gray-900 dark:text-white">{content.title}</div>
                              <div className="text-xs text-gray-500 dark:text-gray-400">{content.type}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-900 dark:text-white">{content.platform}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-900 dark:text-white">{content.likes.toLocaleString()}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-900 dark:text-white">{content.comments.toLocaleString()}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-900 dark:text-white">{content.shares.toLocaleString()}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm text-gray-900 dark:text-white">{content.views.toLocaleString()}</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300">
                            {content.engagementRate}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column */}
            <div className="lg:col-span-2 space-y-6">
            {/* Performance Card */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
              <div className="p-6">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-lg font-medium text-gray-900 dark:text-white">Performance</h3>
                  <div className="flex items-center space-x-2">
                    <select 
                      value={forecastDays}
                      onChange={(e) => setForecastDays(Number(e.target.value))}
                      className="bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-md px-3 py-1.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                    >
                      <option value={7}>Last 7 days</option>
                      <option value={14}>Last 14 days</option>
                      <option value={30}>Last 30 days</option>
                      <option value={60}>Last 60 days</option>
                    </select>
                    <button
                      onClick={handleGenerateForecast}
                      disabled={isForecasting}
                      className={`px-4 py-1.5 rounded-md text-sm font-medium ${
                        isForecasting
                          ? 'bg-gray-200 dark:bg-gray-700 text-gray-500 cursor-not-allowed'
                          : 'bg-primary-600 hover:bg-primary-700 text-white'
                      }`}
                    >
                      {isForecasting ? 'Generating...' : 'Generate Forecast'}
                    </button>
                  </div>
                </div>
                
                <div className="h-80 w-full">
                  {historicalData.length > 0 ? (
                    <Line data={getChartData()} options={chartOptions} ref={chartRef} />
                  ) : (
                    <div className="h-full flex items-center justify-center bg-gray-50 dark:bg-gray-700/30 rounded-lg">
                      <p className="text-gray-500 dark:text-gray-400">No data available</p>
                    </div>
                  )}
                </div>
                
                {forecastData.length > 0 && (
                  <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-100 dark:border-blue-800">
                    <h4 className="text-sm font-medium text-blue-800 dark:text-blue-200 mb-2">Forecast Insights</h4>
                    <p className="text-sm text-blue-700 dark:text-blue-300">
                      Based on the current trend, this video is projected to reach{' '}
                      <span className="font-semibold">
                        {new Intl.NumberFormat('en-US').format(forecastData[forecastData.length - 1].cumulativeViews)} views
                      </span>{' '}
                      in the next {forecastDays} days. That's an estimated increase of{' '}
                      <span className="font-semibold">
                        {Math.round(
                          ((forecastData[forecastData.length - 1].cumulativeViews - historicalData[historicalData.length - 1].cumulativeViews) / 
                          historicalData[historicalData.length - 1].cumulativeViews) * 100
                        )}%
                      </span>{' '}
                      from the current view count.
                    </p>
                  </div>
                )}
              </div>
            </div>
            
            {/* Engagement Card */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
              <div className="p-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-6">Engagement</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">Engagement Rate</h4>
                    <div className="h-48 bg-gray-50 dark:bg-gray-700/30 rounded-lg flex flex-col items-center justify-center p-4">
                      <p className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                        {videoData?.statistics?.engagementRate !== undefined ? videoData.statistics.engagementRate : '0.0'}
                        <span className="text-lg text-gray-500 dark:text-gray-400">/1000</span>
                      </p>
                      <p className="text-sm text-center text-gray-500 dark:text-gray-400">
                        Likes + Comments per 1,000 views
                      </p>
                      <div className="w-full bg-gray-200 rounded-full h-2.5 mt-4 dark:bg-gray-700">
                        <div 
                          className="bg-blue-600 h-2.5 rounded-full" 
                          style={{ width: `${Math.min(100, (videoData?.statistics?.engagementRate || 0) * 10)}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">Audience Retention</h4>
                    <div className="h-48 bg-gray-50 dark:bg-gray-700/30 rounded-lg flex flex-col items-center justify-center p-4">
                      <p className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                        {videoData?.statistics?.audienceRetention || '0%'}
                      </p>
                      <p className="text-sm text-center text-gray-500 dark:text-gray-400">
                        Average view duration: {videoData?.statistics?.averageViewDuration || 'N/A'}
                      </p>
                      <p className="text-sm text-center text-gray-500 dark:text-gray-400">
                        ({videoData?.statistics?.averageViewPercentage || '0%'} of video)
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          {/* Right Column */}
          <div className="space-y-6">
            {/* Quick Stats Card */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
              <div className="p-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Quick Stats</h3>
                <div className="space-y-4">
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Average View Duration</p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">2:45</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Watch Time (hours)</p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">1,245</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Impressions</p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">
                      {new Intl.NumberFormat('en-US').format(Math.round(parseInt(videoData?.statistics?.viewCount || '0') * 1.5))}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 dark:text-gray-400">Click-Through Rate</p>
                    <p className="text-lg font-semibold text-gray-900 dark:text-white">
                      {Math.round((Math.random() * 5 + 3) * 10) / 10}%
                    </p>
                  </div>
                </div>
              </div>
            </div>
            
            {/* Top Videos Card */}
            <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
              <div className="p-6">
                <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">Top Videos from {videoData?.snippet?.channelTitle || 'This Channel'}</h3>
                <div className="space-y-4">
                  {/* Current Video */}
                  <div className="flex items-center space-x-3 p-2 hover:bg-gray-50 dark:hover:bg-gray-700/50 rounded-lg transition-colors">
                    <div className="flex-shrink-0 w-16 h-9 bg-gray-200 dark:bg-gray-700 rounded overflow-hidden">
                      {videoData?.snippet?.thumbnails?.high?.url ? (
                        <img 
                          src={videoData.snippet.thumbnails.high.url} 
                          alt="Thumbnail"
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <Youtube className="h-5 w-5 text-red-600" />
                        </div>
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                        {videoData?.snippet?.title || 'Current Video'}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400">
                        {new Intl.NumberFormat('en-US').format(videoData?.statistics?.viewCount || 0)} views
                      </p>
                    </div>
                  </div>
                  
                  {/* Related Videos */}
                  {relatedVideos.map((video) => (
                    <div key={video.id} className="flex items-center space-x-3 p-2 hover:bg-gray-50 dark:hover:bg-gray-700/50 rounded-lg transition-colors">
                      <div className="flex-shrink-0 w-16 h-9 bg-gray-200 dark:bg-gray-700 rounded overflow-hidden">
                        {video.thumbnail ? (
                          <img 
                            src={video.thumbnail} 
                            alt="Thumbnail"
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full flex items-center justify-center">
                            <Youtube className="h-5 w-5 text-red-600" />
                          </div>
                        )}
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                          {video.title}
                        </p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {new Intl.NumberFormat('en-US').format(video.viewCount)} views
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            
            {/* Ask AI Section */}
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/30 dark:to-blue-800/20 rounded-xl shadow-sm border border-blue-100 dark:border-blue-800/50 overflow-hidden">
              <div className="p-6">
                <h3 className="text-lg font-medium text-blue-900 dark:text-blue-100 mb-3">Ask AI</h3>
                <p className="text-sm text-blue-800 dark:text-blue-200 mb-4">
                  Get insights and recommendations for your video's performance.
                </p>
                <div className="space-y-3">
                  <button className="w-full text-left px-4 py-2 bg-white dark:bg-blue-800/50 text-sm font-medium text-blue-700 dark:text-blue-200 rounded-lg border border-blue-200 dark:border-blue-700 hover:bg-blue-50 dark:hover:bg-blue-800/70 transition-colors">
                    How can I improve my video's performance?
                  </button>
                  <button className="w-full text-left px-4 py-2 bg-white dark:bg-blue-800/50 text-sm font-medium text-blue-700 dark:text-blue-200 rounded-lg border border-blue-200 dark:border-blue-700 hover:bg-blue-50 dark:hover:bg-blue-800/70 transition-colors">
                    What's the best time to post my next video?
                  </button>
                  <button className="w-full text-left px-4 py-2 bg-white dark:bg-blue-800/50 text-sm font-medium text-blue-700 dark:text-blue-200 rounded-lg border border-blue-200 dark:border-blue-700 hover:bg-blue-50 dark:hover:bg-blue-800/70 transition-colors">
                    Analyze my audience demographics
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
        
        {/* Forecast Insights - Moved inside main content */}
        {forecastData.length > 0 && (
          <div className="mt-6">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-100 dark:border-blue-800">
              <h4 className="text-sm font-medium text-blue-800 dark:text-blue-200 mb-2">Forecast Insights</h4>
              <p className="text-sm text-blue-700 dark:text-blue-300">
                Based on the current trend, this video is projected to reach{' '}
                <span className="font-semibold">
                  {new Intl.NumberFormat('en-US').format(forecastData[forecastData.length - 1].cumulativeViews)} views
                </span>{' '}
                in the next {forecastDays} days. That's an estimated increase of{' '}
                <span className="font-semibold">
                  {Math.round(
                    ((forecastData[forecastData.length - 1].cumulativeViews - historicalData[historicalData.length - 1].cumulativeViews) / 
                    historicalData[historicalData.length - 1].cumulativeViews) * 100
                  )}%
                </span>{' '}
                from the current view count.
              </p>
            </div>
          </div>
        )}
      </main>
    </main>
  </div>
  );
};

export default YouTubeDashboardPage;
