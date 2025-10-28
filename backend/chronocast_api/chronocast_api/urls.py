"""
URL Configuration for ChronoCast API
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from rest_framework.routers import DefaultRouter
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from rest_framework import permissions

from forecast.views import (
    DatasetViewSet,
    ForecastRunViewSet,
    ModelComparisonViewSet,
    ExplainabilityViewSet
)

# API Router
router = DefaultRouter()
router.register(r'datasets', DatasetViewSet, basename='dataset')
router.register(r'forecast-runs', ForecastRunViewSet, basename='forecastrun')
router.register(r'comparisons', ModelComparisonViewSet, basename='comparison')
router.register(r'explainability', ExplainabilityViewSet, basename='explainability')

# Swagger/OpenAPI schema
schema_view = get_schema_view(
    openapi.Info(
        title="ChronoCast API",
        default_version='v1',
        description="API for time series forecasting with ChronoCast",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="contact@chronocast.local"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)

urlpatterns = [
    # Admin
    path('admin/', admin.site.urls),
    
    # API
    path('api/', include(router.urls)),
    
    # API Documentation
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
    path('swagger.json', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    
    # Health check
    path('health/', lambda request: JsonResponse({'status': 'ok'})),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

from django.http import JsonResponse