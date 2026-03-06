# YouTube API Setup Guide

## Overview
The ChronoCast frontend is already configured to fetch real YouTube data, but it requires a YouTube Data API v3 key to work properly.

## Current Status
✅ **Frontend**: Fully implemented with real-time data fetching
✅ **Backend**: Real YouTube API integration implemented  
❌ **API Key**: Missing configuration

## Setup Steps

### 1. Get YouTube Data API v3 Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **YouTube Data API v3**:
   - Go to "APIs & Services" → "Library"
   - Search for "YouTube Data API v3"
   - Click "Enable"
4. Create API credentials:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "API Key"
   - Copy the generated API key

### 2. Configure the Backend

**Option A: Environment Variable (Recommended)**
```bash
# In your backend directory
export YOUTUBE_API_KEY=your_actual_api_key_here
```

**Option B: .env file**
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
YOUTUBE_API_KEY=your_actual_api_key_here
```

**Option C: Temporary (for testing)**
Add directly to `settings.py` (not recommended for production):
```python
YOUTUBE_API_KEY = 'your_actual_api_key_here'
```

### 3. Restart the Backend Server

```bash
# Navigate to backend directory
cd backend/chronocast_api

# Restart Django server
python manage.py runserver
```

### 4. Test the Integration

1. Navigate to any YouTube video in your application
2. Click on the "Content" tab
3. The "Top Performing Content" table should now show real data from the channel

## What the API Returns

The integration fetches:
- **Top 3 most popular videos** from the channel (by view count)
- **Real statistics**: views, likes, comments, engagement rate
- **Video metadata**: title, thumbnail, publish date
- **Content type classification**: Shorts vs Long videos

## Features Already Implemented

### Frontend (`YouTubeDashboardPage.jsx`)
- ✅ Real-time data fetching
- ✅ Loading states and error handling  
- ✅ Responsive table display
- ✅ Engagement rate calculations
- ✅ Thumbnail support
- ✅ Content type badges

### Backend (`youtube/views.py`)
- ✅ YouTube Data API v3 integration
- ✅ Channel video fetching
- ✅ Statistics aggregation
- ✅ Content type detection (Shorts vs Videos)
- ✅ Engagement rate calculation
- ✅ Error handling and fallbacks

## Troubleshooting

### "No top videos found" Error
- **Cause**: YouTube API key not configured or invalid
- **Solution**: Follow the setup steps above

### API Quota Issues
- YouTube Data API has daily quotas
- Monitor usage in Google Cloud Console
- Consider implementing caching for production

### CORS Errors
- Ensure backend CORS settings include your frontend URL
- Check that both frontend and backend are running

## API Endpoints Used

- `GET /api/youtube/channel/{channel_id}/top-content/`
- Fetches channel's most popular videos with full statistics

## Security Notes

- Never commit API keys to version control
- Use environment variables in production
- Consider implementing API key rotation for security
- Monitor API usage and costs
