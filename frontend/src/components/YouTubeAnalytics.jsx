import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Loader2, Youtube, BarChart2, ThumbsUp, Eye, Calendar, TrendingUp, CalendarDays, Clock } from 'lucide-react';
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
  const [error, setError] = useState('');
  const [videoData, setVideoData] = useState(null);
  const [relatedVideos, setRelatedVideos] = useState([]);
  const [isLoadingRelated, setIsLoadingRelated] = useState(false);
  const navigate = useNavigate();
  
  const API_KEY = 'AIzaSyBp6Wj5YPdGe-GY9gVmO1BRzCQqDsKVp9A';
  
  const extractVideoId = (url) => {
    const regExp = /^.*(youtu.be\/|v\/|u\/\w\/|embed\/|watch\?v=|&v=)([^#&?]*).*/;
    const match = url.match(regExp);
    return (match && match[2].length === 11) ? match[2] : null;
  };

  const fetchRelatedVideos = async (channelId, videoId, videoTitle = '', videoTags = []) => {
    try {
      setIsLoadingRelated(true);
      
      // First, get the video details to extract tags and title for better search
      const searchQuery = videoTitle.split(' ').slice(0, 3).join(' '); // Use first 3 words of title for search
      
      // Search for videos in the same channel with similar topic
      const response = await fetch(
        `https://www.googleapis.com/youtube/v3/search?` + 
        `part=snippet&channelId=${channelId}&maxResults=5&q=${encodeURIComponent(searchQuery)}&type=video&key=${API_KEY}`
      );
      
      if (!response.ok) {
        throw new Error('Failed to fetch related videos');
      }
      
      const data = await response.json();
      
      // If no results from title search, fall back to general channel videos
      if (!data.items || data.items.length === 0) {
        const fallbackResponse = await fetch(
          `https://www.googleapis.com/youtube/v3/search?part=snippet&channelId=${channelId}&maxResults=5&type=video&order=viewCount&key=${API_KEY}`
        );
        
        if (fallbackResponse.ok) {
          const fallbackData = await fallbackResponse.json();
          if (fallbackData.items) {
            const videos = fallbackData.items
              .filter(item => item.id.videoId !== videoId)
              .slice(0, 5) // Limit to 5 videos
              .map(item => ({
                id: item.id.videoId,
                title: item.snippet.title,
                thumbnail: item.snippet.thumbnails.medium.url,
                channelTitle: item.snippet.channelTitle,
                publishedAt: item.snippet.publishedAt,
                viewCount: '0', // Not available in search results
                isTopVideo: true
              }));
            setRelatedVideos(videos);
            return;
          }
        }
      }
      
      // Process the search results
      if (data.items) {
        const videos = data.items
          .filter(item => item.id.videoId !== videoId)
          .slice(0, 5) // Limit to 5 videos
          .map(item => ({
            id: item.id.videoId,
            title: item.snippet.title,
            thumbnail: item.snippet.thumbnails.medium.url,
            channelTitle: item.snippet.channelTitle,
            publishedAt: item.snippet.publishedAt,
            viewCount: '0', // Not available in search results
            isTopVideo: false
          }));
        
        setRelatedVideos(videos);
      }
    } catch (err) {
      console.error('Error fetching related videos:', err);
    } finally {
      setIsLoadingRelated(false);
    }
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
      // Get video statistics, snippet, and contentDetails
      const response = await fetch(
        `https://www.googleapis.com/youtube/v3/videos?part=statistics,snippet,contentDetails&id=${videoId}&key=${API_KEY}`
      );
      
      if (!response.ok) {
        throw new Error('Failed to fetch video data');
      }
      
      const data = await response.json();
      
      if (data.items && data.items.length > 0) {
        const video = data.items[0];
        const viewCount = parseInt(video.statistics.viewCount || 0);
        const likeCount = parseInt(video.statistics.likeCount || 0);
        const commentCount = parseInt(video.statistics.commentCount || 0);
        
        // Calculate engagement rate (likes + comments per 1000 views)
        const engagementRate = viewCount > 0 
          ? ((likeCount + commentCount) / viewCount * 1000).toFixed(1) 
          : 0;
        
        // Parse duration from ISO 8601 format (e.g., PT1H2M3S -> 1:02:03)
        const parseDuration = (duration) => {
          if (!duration) return 'N/A';
          const match = duration.match(/PT(\d+H)?(\d+M)?(\d+S)?/);
          if (!match) return 'N/A';
          
          const hours = (parseInt(match[1]) || 0);
          const minutes = (parseInt(match[2]) || 0);
          const seconds = (parseInt(match[3]) || 0);
          
          const parts = [];
          if (hours > 0) parts.push(hours);
          parts.push(minutes.toString().padStart(hours > 0 ? 2 : 1, '0'));
          parts.push(seconds.toString().padStart(2, '0'));
          
          return parts.join(':');
        };
        
        // Calculate duration in seconds for retention calculations
        const durationInSeconds = (() => {
          if (!video.contentDetails?.duration) return 0;
          const match = video.contentDetails.duration.match(/PT(\d+H)?(\d+M)?(\d+S)?/);
          if (!match) return 0;
          return (parseInt(match[1] || 0) * 3600) + (parseInt(match[2] || 0) * 60) + parseInt(match[3] || 0);
        })();
        
        // Simulate audience retention (40-70% of video length on average)
        const averageViewPercentage = 0.4 + (Math.random() * 0.3);
        const audienceRetention = (averageViewPercentage * 100).toFixed(1) + '%';
        
        const videoData = {
          snippet: {
            title: video.snippet?.title || 'Untitled',
            thumbnails: {
              high: { url: video.snippet?.thumbnails?.high?.url || '' }
            },
            channelTitle: video.snippet?.channelTitle || 'Unknown Channel',
            publishedAt: video.snippet?.publishedAt || new Date().toISOString(),
            description: video.snippet?.description || ''
          },
          contentDetails: {
            duration: video.contentDetails?.duration || 'PT0M0S',
            durationFormatted: parseDuration(video.contentDetails?.duration || ''),
            dimension: video.contentDetails?.dimension || '2d',
            definition: video.contentDetails?.definition || 'hd',
            caption: video.contentDetails?.caption || 'false'
          },
          statistics: {
            viewCount: viewCount,
            likeCount: likeCount,
            commentCount: commentCount,
            favoriteCount: parseInt(video.statistics?.favoriteCount || '0'),
            engagementRate: parseFloat(engagementRate),
            audienceRetention: audienceRetention,
            averageViewDuration: Math.round(durationInSeconds * averageViewPercentage) + 's',
            averageViewPercentage: parseFloat((averageViewPercentage * 100).toFixed(1))
          }
        };
        
        setVideoData(videoData);
        // Fetch related videos after setting video data
        if (video.snippet?.channelId) {
          const videoTitle = video.snippet?.title || '';
          const videoTags = video.snippet?.tags || [];
          fetchRelatedVideos(video.snippet.channelId, videoId, videoTitle, videoTags);
        }
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
  
  // Remove the handleGenerateForecast function as it's now handled in the dashboard
  
  // Remove the chart data generation as it's now handled in the dashboard
  
  // Remove chart options as they're now in the dashboard
  
  // Navigate to dashboard when video data is loaded
  useEffect(() => {
    if (videoData?.statistics?.viewCount !== undefined) {
      // Safely handle viewCount whether it's a string or number
      const views = typeof videoData.statistics.viewCount === 'string' 
        ? parseInt(videoData.statistics.viewCount.replace(/[^0-9]/g, ''), 10) || 0
        : Math.floor(Number(videoData.statistics.viewCount)) || 0;
      
      const historical = generateHistoricalData(views);
      
      // Navigate to dashboard with the video data
      navigate('/youtube-dashboard', { 
        state: { 
          videoData: {
            ...videoData,
            statistics: {
              ...videoData.statistics,
              viewCount: views
            }
          },
          historicalData: historical
        } 
      });
    }
  }, [videoData, navigate]);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl overflow-hidden border border-gray-200 dark:border-gray-700 mt-10">
      <div className="p-6 md:p-8">
        <div className="flex items-center mb-6">
          <Youtube className="w-8 h-8 text-red-600 mr-3" />
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">YouTube Video Analytics</h2>
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
          <>
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-gray-50 dark:bg-gray-700 rounded-xl p-6 mt-6"
            >
              <div className="text-center py-8">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500 mx-auto mb-4"></div>
                <p className="text-gray-600 dark:text-gray-300">Loading dashboard...</p>
              </div>
            </motion.div>

            {relatedVideos.length > 0 && (
              <div className="mt-8">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">More from this channel</h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {relatedVideos.map((video) => (
                    <a
                      key={video.id}
                      href={`https://www.youtube.com/watch?v=${video.id}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="group block bg-white dark:bg-gray-800 rounded-lg overflow-hidden shadow-sm hover:shadow-md transition-shadow duration-200 border border-gray-200 dark:border-gray-700"
                    >
                      <div className="relative pt-[56.25%] bg-gray-100 dark:bg-gray-700">
                        <img
                          src={video.thumbnail}
                          alt={video.title}
                          className="absolute inset-0 w-full h-full object-cover"
                        />
                      </div>
                      <div className="p-4">
                        <div className="flex justify-between items-start mb-1">
                          <h4 className="font-medium text-gray-900 dark:text-white line-clamp-2 group-hover:text-red-600 dark:group-hover:text-red-500 transition-colors pr-2">
                            {video.title}
                          </h4>
                          {video.isTopVideo && (
                            <span className="bg-yellow-100 text-yellow-800 text-xs font-medium px-2 py-0.5 rounded dark:bg-yellow-900 dark:text-yellow-300">
                              Top Video
                            </span>
                          )}
                        </div>
                        <p className="text-sm text-gray-500 dark:text-gray-400">{video.channelTitle}</p>
                        <div className="flex items-center justify-between mt-2 text-xs text-gray-400">
                          <div className="flex items-center">
                            <Clock className="w-3.5 h-3.5 mr-1" />
                            {new Date(video.publishedAt).toLocaleDateString()}
                          </div>
                          {video.viewCount !== '0' && (
                            <div className="flex items-center">
                              <Eye className="w-3.5 h-3.5 mr-1" />
                              {parseInt(video.viewCount).toLocaleString()}
                            </div>
                          )}
                        </div>
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            )}
            {isLoadingRelated && (
              <div className="mt-8 flex justify-center">
                <Loader2 className="animate-spin h-6 w-6 text-gray-400" />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default YouTubeAnalytics;
