"""
Real-World Data Fetchers for ChronoCast
Fetch time series data from various sources
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import json
from typing import Optional, Dict, List
import os


class YouTubeDataFetcher:
    """
    Fetch YouTube video statistics over time
    Requires: YouTube Data API v3 key
    """
    
    def __init__(self, api_key: str):
        """
        Initialize with YouTube API key
        Get your key: https://console.cloud.google.com/
        """
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL"""
        if 'youtu.be/' in url:
            return url.split('youtu.be/')[1].split('?')[0]
        elif 'youtube.com/watch?v=' in url:
            return url.split('v=')[1].split('&')[0]
        else:
            return url
    
    def get_video_stats(self, video_url: str) -> pd.DataFrame:
        """
        Get current statistics for a video
        
        Args:
            video_url: YouTube video URL or ID
        
        Returns:
            DataFrame with video statistics
        """
        video_id = self.extract_video_id(video_url)
        
        url = f"{self.base_url}/videos"
        params = {
            'part': 'statistics,snippet',
            'id': video_id,
            'key': self.api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'items' not in data or len(data['items']) == 0:
            raise ValueError("Video not found or API key invalid")
        
        item = data['items'][0]
        stats = item['statistics']
        snippet = item['snippet']
        
        return pd.DataFrame([{
            'date': datetime.now(),
            'video_id': video_id,
            'title': snippet['title'],
            'views': int(stats.get('viewCount', 0)),
            'likes': int(stats.get('likeCount', 0)),
            'comments': int(stats.get('commentCount', 0)),
            'published_at': snippet['publishedAt']
        }])
    
    def get_channel_stats(self, channel_id: str, max_videos: int = 50) -> pd.DataFrame:
        """
        Get statistics for all videos in a channel
        
        Args:
            channel_id: YouTube channel ID
            max_videos: Maximum number of videos to fetch
        
        Returns:
            DataFrame with all videos' statistics
        """
        # Get channel's videos
        url = f"{self.base_url}/search"
        params = {
            'part': 'id',
            'channelId': channel_id,
            'maxResults': max_videos,
            'order': 'date',
            'type': 'video',
            'key': self.api_key
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        video_ids = [item['id']['videoId'] for item in data.get('items', [])]
        
        # Get statistics for all videos
        all_stats = []
        for video_id in video_ids:
            try:
                stats = self.get_video_stats(video_id)
                all_stats.append(stats)
            except:
                continue
        
        return pd.concat(all_stats, ignore_index=True) if all_stats else pd.DataFrame()


class StockDataFetcher:
    """
    Fetch stock market data
    Uses Alpha Vantage API (free tier available)
    """
    
    def __init__(self, api_key: str):
        """
        Initialize with Alpha Vantage API key
        Get free key: https://www.alphavantage.co/support/#api-key
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_daily_data(self, symbol: str, outputsize: str = 'compact') -> pd.DataFrame:
        """
        Get daily stock data
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'GOOGL')
            outputsize: 'compact' (100 days) or 'full' (20+ years)
        
        Returns:
            DataFrame with date and price data
        """
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        data = response.json()
        
        if 'Time Series (Daily)' not in data:
            raise ValueError(f"Error fetching data: {data.get('Note', 'API limit or invalid symbol')}")
        
        time_series = data['Time Series (Daily)']
        
        df = pd.DataFrame([
            {
                'date': pd.to_datetime(date),
                'open': float(values['1. open']),
                'high': float(values['2. high']),
                'low': float(values['3. low']),
                'close': float(values['4. close']),
                'volume': int(values['5. volume'])
            }
            for date, values in time_series.items()
        ])
        
        return df.sort_values('date').reset_index(drop=True)


class TwitterDataFetcher:
    """
    Fetch Twitter/X data
    Requires: Twitter API v2 Bearer Token
    """
    
    def __init__(self, bearer_token: str):
        """
        Initialize with Twitter Bearer Token
        Get token: https://developer.twitter.com/
        """
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2"
    
    def get_user_tweets_count(self, username: str, days: int = 30) -> pd.DataFrame:
        """
        Get tweet count per day for a user
        
        Args:
            username: Twitter username (without @)
            days: Number of days to fetch
        
        Returns:
            DataFrame with daily tweet counts
        """
        # Note: This is a simplified example
        # Full implementation requires Twitter API v2 access
        headers = {'Authorization': f'Bearer {self.bearer_token}'}
        
        # Get user ID
        user_url = f"{self.base_url}/users/by/username/{username}"
        user_response = requests.get(user_url, headers=headers)
        
        if user_response.status_code != 200:
            raise ValueError("User not found or API access denied")
        
        user_id = user_response.json()['data']['id']
        
        # Fetch tweets (simplified - actual implementation needs pagination)
        tweets_url = f"{self.base_url}/users/{user_id}/tweets"
        params = {
            'max_results': 100,
            'tweet.fields': 'created_at,public_metrics'
        }
        
        response = requests.get(tweets_url, headers=headers, params=params)
        
        if response.status_code != 200:
            raise ValueError("Error fetching tweets")
        
        tweets = response.json().get('data', [])
        
        # Convert to daily counts
        df = pd.DataFrame([
            {
                'date': pd.to_datetime(tweet['created_at']).date(),
                'likes': tweet['public_metrics']['like_count'],
                'retweets': tweet['public_metrics']['retweet_count'],
                'replies': tweet['public_metrics']['reply_count']
            }
            for tweet in tweets
        ])
        
        return df.groupby('date').sum().reset_index()


class CryptoDataFetcher:
    """
    Fetch cryptocurrency data
    Uses CoinGecko API (no key required for basic use)
    """
    
    def __init__(self):
        """Initialize CoinGecko fetcher"""
        self.base_url = "https://api.coingecko.com/api/v3"
    
    def get_historical_data(self, coin_id: str, days: int = 365) -> pd.DataFrame:
        """
        Get historical price data for a cryptocurrency
        
        Args:
            coin_id: Coin ID (e.g., 'bitcoin', 'ethereum')
            days: Number of days (max 365 for free tier)
        
        Returns:
            DataFrame with price data
        """
        url = f"{self.base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'prices' not in data:
            raise ValueError(f"Error fetching data for {coin_id}")
        
        df = pd.DataFrame([
            {
                'date': pd.to_datetime(item[0], unit='ms'),
                'price': item[1]
            }
            for item in data['prices']
        ])
        
        return df


class WeatherDataFetcher:
    """
    Fetch weather data
    Uses OpenWeatherMap API
    """
    
    def __init__(self, api_key: str):
        """
        Initialize with OpenWeatherMap API key
        Get key: https://openweathermap.org/api
        """
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
    
    def get_historical_weather(self, city: str, days: int = 30) -> pd.DataFrame:
        """
        Get historical weather data
        Note: Historical data requires paid plan
        This is a simplified example
        
        Args:
            city: City name
            days: Number of days
        
        Returns:
            DataFrame with weather data
        """
        # For demonstration - actual historical data requires paid API
        # This fetches current weather
        url = f"{self.base_url}/weather"
        params = {
            'q': city,
            'appid': self.api_key,
            'units': 'metric'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if response.status_code != 200:
            raise ValueError(f"Error fetching weather: {data.get('message', 'Unknown error')}")
        
        return pd.DataFrame([{
            'date': datetime.now(),
            'city': city,
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],
            'pressure': data['main']['pressure'],
            'weather': data['weather'][0]['description']
        }])


class GoogleTrendsDataFetcher:
    """
    Fetch Google Trends data
    Uses pytrends library
    """
    
    def __init__(self):
        """Initialize Google Trends fetcher"""
        try:
            from pytrends.request import TrendReq
            self.pytrends = TrendReq(hl='en-US', tz=360)
        except ImportError:
            raise ImportError("pytrends not installed. Run: pip install pytrends")
    
    def get_interest_over_time(self, keyword: str, timeframe: str = 'today 3-m') -> pd.DataFrame:
        """
        Get search interest over time
        
        Args:
            keyword: Search keyword
            timeframe: Time range (e.g., 'today 3-m', 'today 12-m', 'all')
        
        Returns:
            DataFrame with search interest over time
        """
        self.pytrends.build_payload([keyword], timeframe=timeframe)
        df = self.pytrends.interest_over_time()
        
        if df.empty:
            raise ValueError(f"No data found for keyword: {keyword}")
        
        df = df.reset_index()
        df = df.rename(columns={keyword: 'interest', 'date': 'date'})
        df = df[['date', 'interest']]
        
        return df


# Example usage function
def fetch_youtube_example():
    """
    Example: Fetch YouTube data and prepare for forecasting
    """
    # Get API key from environment
    api_key = os.getenv('YOUTUBE_API_KEY')
    
    if not api_key:
        print("Set YOUTUBE_API_KEY environment variable")
        return None
    
    fetcher = YouTubeDataFetcher(api_key)
    
    # Example video URL
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    # Get stats
    stats = fetcher.get_video_stats(video_url)
    
    print("YouTube Video Stats:")
    print(stats)
    
    return stats


def fetch_stock_example():
    """
    Example: Fetch stock data and prepare for forecasting
    """
    api_key = os.getenv('ALPHA_VANTAGE_KEY')
    
    if not api_key:
        print("Set ALPHA_VANTAGE_KEY environment variable")
        print("Get free key: https://www.alphavantage.co/support/#api-key")
        return None
    
    fetcher = StockDataFetcher(api_key)
    
    # Fetch Apple stock data
    df = fetcher.get_daily_data('AAPL', outputsize='compact')
    
    print(f"\nFetched {len(df)} days of AAPL stock data")
    print(df.head())
    
    return df


def fetch_crypto_example():
    """
    Example: Fetch cryptocurrency data (no API key needed)
    """
    fetcher = CryptoDataFetcher()
    
    # Fetch Bitcoin data
    df = fetcher.get_historical_data('bitcoin', days=90)
    
    print(f"\nFetched {len(df)} days of Bitcoin data")
    print(df.head())
    
    return df


if __name__ == "__main__":
    print("="*80)
    print("ChronoCast Data Fetchers - Examples")
    print("="*80)
    
    # Crypto example (no key needed)
    print("\n1. Fetching Crypto Data (No API key required):")
    print("-"*80)
    try:
        crypto_data = fetch_crypto_example()
    except Exception as e:
        print(f"Error: {e}")
    
    # Stock example (requires API key)
    print("\n2. Fetching Stock Data (Requires ALPHA_VANTAGE_KEY):")
    print("-"*80)
    try:
        stock_data = fetch_stock_example()
    except Exception as e:
        print(f"Error: {e}")
    
    # YouTube example (requires API key)
    print("\n3. Fetching YouTube Data (Requires YOUTUBE_API_KEY):")
    print("-"*80)
    try:
        youtube_data = fetch_youtube_example()
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*80)
    print("To use these fetchers, set up API keys:")
    print("  export YOUTUBE_API_KEY='your_key'")
    print("  export ALPHA_VANTAGE_KEY='your_key'")
    print("  export TWITTER_BEARER_TOKEN='your_token'")
    print("="*80)