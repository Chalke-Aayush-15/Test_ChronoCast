from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from django.conf import settings
import requests
import json
from datetime import datetime, timedelta

class YouTubeAnalyticsViewSet(viewsets.ViewSet):
    """
    YouTube Analytics API ViewSet
    Provides endpoints for fetching YouTube channel analytics
    """
    permission_classes = []  # Remove authentication for testing
    
    def get_youtube_api_key(self):
        """Get YouTube API key from settings"""
        return getattr(settings, 'YOUTUBE_API_KEY', None)
    
    def make_youtube_request(self, url):
        """Make request to YouTube API with proper error handling"""
        api_key = self.get_youtube_api_key()
        if not api_key:
            return None, "YouTube API key not configured"
        
        try:
            params = {'key': api_key}
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json(), None
        except requests.exceptions.RequestException as e:
            return None, str(e)
    
    @action(detail=False, methods=['get'], url_path='channel/(?P<channel_id>[^/]+)/analytics')
    def channel_analytics(self, request, channel_id=None):
        """Get channel analytics"""
        # Mock data for now - replace with real YouTube API calls
        mock_data = {
            'channel_id': channel_id,
            'total_views': 1500000,
            'total_subscribers': 50000,
            'total_videos': 150,
            'engagement_rate': 8.5
        }
        return Response(mock_data)
    
    @action(detail=False, methods=['get'], url_path='channel/(?P<channel_id>[^/]+)/shorts')
    def shorts(self, request, channel_id=None):
        """Get channel shorts"""
        # Mock shorts data - replace with real YouTube API calls
        mock_shorts = [
            {
                'id': 'short1',
                'title': 'Quick Study Tips',
                'type': 'Shorts',
                'likes': 2450,
                'comments': 320,
                'shares': 78,
                'views': 18700,
                'engagementRate': 15.2
            },
            {
                'id': 'short2',
                'title': '5 Minute Math Tricks',
                'type': 'Shorts',
                'likes': 1890,
                'comments': 245,
                'shares': 56,
                'views': 12300,
                'engagementRate': 18.1
            }
        ]
        return Response(mock_shorts)
    
    @action(detail=False, methods=['get'], url_path='channel/(?P<channel_id>[^/]+)/live-streams')
    def live_streams(self, request, channel_id=None):
        """Get channel live streams"""
        # Mock live streams data
        mock_live_streams = [
            {
                'id': 'live1',
                'title': 'Study Session Live',
                'type': 'Live Stream',
                'likes': 980,
                'comments': 145,
                'shares': 32,
                'views': 8900,
                'engagementRate': 11.3
            }
        ]
        return Response(mock_live_streams)
    
    @action(detail=False, methods=['get'], url_path='channel/(?P<channel_id>[^/]+)/long-videos')
    def long_videos(self, request, channel_id=None):
        """Get channel long videos"""
        # Mock long videos data
        mock_long_videos = [
            {
                'id': 'long1',
                'title': 'Complete Course Tutorial',
                'type': 'Long Video',
                'likes': 3450,
                'comments': 420,
                'shares': 95,
                'views': 28700,
                'engagementRate': 13.8
            }
        ]
        return Response(mock_long_videos)
    
    @action(detail=False, methods=['get'], url_path='channel/(?P<channel_id>[^/]+)/community-posts')
    def community_posts(self, request, channel_id=None):
        """Get channel community posts"""
        # Mock community posts data
        mock_posts = [
            {
                'id': 'post1',
                'title': 'Behind the Scenes',
                'type': 'Posts',
                'likes': 780,
                'comments': 98,
                'shares': 23,
                'views': 6500,
                'engagementRate': 13.8
            }
        ]
        return Response(mock_posts)
    
    @action(detail=False, methods=['get'], url_path='channel/(?P<channel_id>[^/]+)/top-content')
    def top_content(self, request, channel_id=None):
        """Get top performing content across all types"""
        content_type = request.query_params.get('type', 'all')
        
        # Combine all content types
        all_content = [
            {
                'id': 'short1',
                'title': 'Quick Study Tips',
                'type': 'Shorts',
                'likes': 2450,
                'comments': 320,
                'shares': 78,
                'views': 18700,
                'engagementRate': 15.2
            },
            {
                'id': 'live1',
                'title': 'Study Session Live',
                'type': 'Live Stream',
                'likes': 980,
                'comments': 145,
                'shares': 32,
                'views': 8900,
                'engagementRate': 11.3
            },
            {
                'id': 'long1',
                'title': 'Complete Course Tutorial',
                'type': 'Long Video',
                'likes': 3450,
                'comments': 420,
                'shares': 95,
                'views': 28700,
                'engagementRate': 13.8
            },
            {
                'id': 'post1',
                'title': 'Behind the Scenes',
                'type': 'Posts',
                'likes': 780,
                'comments': 98,
                'shares': 23,
                'views': 6500,
                'engagementRate': 13.8
            }
        ]
        
        # Filter by content type if specified
        if content_type != 'all':
            all_content = [item for item in all_content if item['type'].lower() == content_type.lower()]
        
        # Sort by engagement rate
        sorted_content = sorted(all_content, key=lambda x: x['engagementRate'], reverse=True)
        
        return Response(sorted_content)
    
    @action(detail=False, methods=['get'], url_path='video/(?P<video_id>[^/]+)/details')
    def video_details(self, request, video_id=None):
        """Get detailed video analytics"""
        # Mock video details
        mock_details = {
            'video_id': video_id,
            'title': 'Sample Video Title',
            'views': 15000,
            'likes': 1250,
            'comments': 234,
            'shares': 45,
            'engagement_rate': 10.2,
            'average_watch_time': '3:45',
            'audience_retention': '65%'
        }
        return Response(mock_details)
