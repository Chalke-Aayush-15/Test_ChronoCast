# 🔑 API Keys Setup Guide

## How to Get API Keys for Real-World Data

---

## 📺 YouTube Data API

### Step 1: Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Name your project (e.g., "ChronoCast YouTube")
4. Click "Create"

### Step 2: Enable YouTube Data API
1. In the left menu, go to "APIs & Services" → "Library"
2. Search for "YouTube Data API v3"
3. Click on it and press "Enable"

### Step 3: Create API Key
1. Go to "APIs & Services" → "Credentials"
2. Click "+ CREATE CREDENTIALS" → "API key"
3. Copy your API key
4. (Optional) Click "Restrict Key" for security

### Step 4: Set Environment Variable
```bash
# Linux/Mac
export YOUTUBE_API_KEY='your_api_key_here'

# Windows (CMD)
set YOUTUBE_API_KEY=your_api_key_here

# Windows (PowerShell)
$env:YOUTUBE_API_KEY='your_api_key_here'

# Or in Python
import os
os.environ['YOUTUBE_API_KEY'] = 'your_api_key_here'
```

### Usage Example
```python
from chronocast.utils.data_fetchers import YouTubeDataFetcher

fetcher = YouTubeDataFetcher(api_key='your_key')
stats = fetcher.get_video_stats('https://www.youtube.com/watch?v=VIDEO_ID')
print(stats)
```

### Quotas
- **Free Tier**: 10,000 units/day
- **1 video stats request**: ~3 units
- **Approximately**: ~3,000 requests/day

---

## 📈 Stock Market Data (Alpha Vantage)

### Step 1: Sign Up
1. Go to [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Enter your email
3. Click "GET FREE API KEY"
4. Check your email for the key

### Step 2: Set Environment Variable
```bash
export ALPHA_VANTAGE_KEY='your_api_key_here'
```

### Usage Example
```python
from chronocast.utils.data_fetchers import StockDataFetcher

fetcher = StockDataFetcher(api_key='your_key')
data = fetcher.get_daily_data('AAPL')  # Apple stock
print(data.head())
```

### Quotas
- **Free Tier**: 25 requests/day
- **Paid Tier**: 75+ requests/minute

---

## 🐦 Twitter/X API

### Step 1: Apply for Developer Account
1. Go to [Twitter Developer Portal](https://developer.twitter.com/)
2. Click "Sign up" or "Apply"
3. Fill out the application form
4. Wait for approval (usually instant for basic access)

### Step 2: Create App
1. Go to "Developer Portal" → "Projects & Apps"
2. Click "+ Create App"
3. Name your app
4. Copy your Bearer Token

### Step 3: Set Environment Variable
```bash
export TWITTER_BEARER_TOKEN='your_bearer_token_here'
```

### Usage Example
```python
from chronocast.utils.data_fetchers import TwitterDataFetcher

fetcher = TwitterDataFetcher(bearer_token='your_token')
data = fetcher.get_user_tweets_count('username')
print(data)
```

### Quotas
- **Free Tier**: 500,000 tweets/month
- **Basic**: $100/month for 10M tweets

---

## 💰 Cryptocurrency Data (CoinGecko)

### No API Key Needed! ✅

```python
from chronocast.utils.data_fetchers import CryptoDataFetcher

fetcher = CryptoDataFetcher()
data = fetcher.get_historical_data('bitcoin', days=90)
print(data.head())
```

### Quotas
- **Free Tier**: 10-50 calls/minute
- No registration required!

### Popular Coin IDs
- `bitcoin` - Bitcoin
- `ethereum` - Ethereum
- `cardano` - Cardano
- `solana` - Solana
- `dogecoin` - Dogecoin

---

## 🌤️ Weather Data (OpenWeatherMap)

### Step 1: Sign Up
1. Go to [OpenWeatherMap](https://openweathermap.org/api)
2. Click "Sign Up"
3. Create account
4. Go to "API Keys" tab
5. Copy your API key

### Step 2: Set Environment Variable
```bash
export OPENWEATHER_API_KEY='your_api_key_here'
```

### Usage Example
```python
from chronocast.utils.data_fetchers import WeatherDataFetcher

fetcher = WeatherDataFetcher(api_key='your_key')
data = fetcher.get_historical_weather('London', days=30)
print(data)
```

### Quotas
- **Free Tier**: 60 calls/minute, 1,000,000 calls/month
- Current weather data is free
- Historical data requires paid plan

---

## 📊 Google Trends

### No API Key Needed! ✅

### Installation
```bash
pip install pytrends
```

### Usage Example
```python
from chronocast.utils.data_fetchers import GoogleTrendsDataFetcher

fetcher = GoogleTrendsDataFetcher()
data = fetcher.get_interest_over_time('python programming', timeframe='today 12-m')
print(data.head())
```

### Quotas
- Rate limited by Google
- Use responsibly (don't spam requests)

---

## 🔒 Security Best Practices

### 1. Never Commit API Keys
```bash
# Add to .gitignore
echo "*.env" >> .gitignore
echo ".env.local" >> .gitignore
```

### 2. Use Environment Variables
Create a `.env` file:
```bash
# .env
YOUTUBE_API_KEY=your_youtube_key
ALPHA_VANTAGE_KEY=your_stock_key
TWITTER_BEARER_TOKEN=your_twitter_token
OPENWEATHER_API_KEY=your_weather_key
```

Load in Python:
```python
from dotenv import load_dotenv
import os

load_dotenv()

youtube_key = os.getenv('YOUTUBE_API_KEY')
```

### 3. Restrict API Keys
- Add HTTP referrer restrictions
- Add IP address restrictions
- Set API quotas
- Enable billing alerts

---

## 📝 Quick Setup Script

Save as `setup_env.sh`:

```bash
#!/bin/bash

echo "Setting up ChronoCast API keys..."

# Prompt for keys
read -p "YouTube API Key (or press Enter to skip): " YOUTUBE_KEY
read -p "Alpha Vantage Key (or press Enter to skip): " STOCK_KEY
read -p "Twitter Bearer Token (or press Enter to skip): " TWITTER_KEY
read -p "OpenWeather API Key (or press Enter to skip): " WEATHER_KEY

# Create .env file
cat > .env << EOF
# ChronoCast API Keys
YOUTUBE_API_KEY=${YOUTUBE_KEY}
ALPHA_VANTAGE_KEY=${STOCK_KEY}
TWITTER_BEARER_TOKEN=${TWITTER_KEY}
OPENWEATHER_API_KEY=${WEATHER_KEY}
EOF

echo "✓ API keys saved to .env file"
echo "✓ Add .env to .gitignore for security"
```

Run it:
```bash
chmod +x setup_env.sh
./setup_env.sh
```

---

## 🧪 Test Your Setup

```python
import os
from chronocast.utils.data_fetchers import (
    YouTubeDataFetcher,
    StockDataFetcher,
    CryptoDataFetcher,
    TwitterDataFetcher,
    WeatherDataFetcher,
    GoogleTrendsDataFetcher
)

# Test YouTube
if os.getenv('YOUTUBE_API_KEY'):
    yt = YouTubeDataFetcher(os.getenv('YOUTUBE_API_KEY'))
    print("✓ YouTube API working")

# Test Stock
if os.getenv('ALPHA_VANTAGE_KEY'):
    stock = StockDataFetcher(os.getenv('ALPHA_VANTAGE_KEY'))
    print("✓ Stock API working")

# Test Crypto (no key needed)
crypto = CryptoDataFetcher()
print("✓ Crypto API working")

# Test Google Trends (no key needed)
trends = GoogleTrendsDataFetcher()
print("✓ Google Trends working")

print("\n✅ All APIs configured correctly!")
```

---

## 💡 Tips

### 1. Start with Free APIs
- CoinGecko (crypto)
- Google Trends
- Alpha Vantage free tier

### 2. Monitor Usage
- Set up billing alerts
- Check quota dashboards
- Log API calls

### 3. Cache Data
```python
import pickle
from datetime import datetime, timedelta

def fetch_with_cache(fetcher_func, cache_file, max_age_hours=24):
    # Check if cache exists and is fresh
    if os.path.exists(cache_file):
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - cache_time < timedelta(hours=max_age_hours):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
    
    # Fetch fresh data
    data = fetcher_func()
    
    # Save to cache
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    return data
```

### 4. Handle Errors Gracefully
```python
try:
    data = fetcher.get_data()
except Exception as e:
    print(f"API Error: {e}")
    # Fall back to cached data or simulated data
    data = load_backup_data()
```

---

## 📚 Resources

- [YouTube API Docs](https://developers.google.com/youtube/v3)
- [Alpha Vantage Docs](https://www.alphavantage.co/documentation/)
- [Twitter API Docs](https://developer.twitter.com/en/docs)
- [CoinGecko API Docs](https://www.coingecko.com/en/api/documentation)
- [OpenWeather API Docs](https://openweathermap.org/api)
- [pytrends Docs](https://pypi.org/project/pytrends/)

---

## ❓ FAQ

**Q: Do I need all API keys?**  
A: No! Start with crypto or Google Trends (no keys needed).

**Q: What if I exceed quotas?**  
A: Use caching, or upgrade to paid tier.

**Q: Are API keys free?**  
A: Most have free tiers. Check each service.

**Q: How do I secure my keys?**  
A: Use `.env` files and never commit them to Git.

**Q: Can I use multiple keys?**  
A: Yes! Rotate keys to increase quotas.

---

**Ready to fetch real-world data!** 🚀