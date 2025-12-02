/**
 * API Service for ChronoCast
 * Handles all backend communication
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ============================================================
// DATASETS
// ============================================================

export const datasetAPI = {
  // List all datasets
  list: () => api.get('/datasets/'),
  
  // Get dataset by ID
  get: (id) => api.get(`/datasets/${id}/`),
  
  // Upload dataset
  upload: (formData) => {
    return api.post('/datasets/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  
  // Validate dataset
  validate: (id, data) => api.post(`/datasets/${id}/validate/`, data),
  
  // Preview dataset
  preview: (id, nRows = 10) => api.get(`/datasets/${id}/preview/?n_rows=${nRows}`),
  
  // Delete dataset
  delete: (id) => api.delete(`/datasets/${id}/`),
};

// ============================================================
// FORECAST RUNS
// ============================================================

export const forecastAPI = {
  // List all forecast runs
  list: (params = {}) => api.get('/forecast-runs/', { params }),
  
  // Get forecast run by ID
  get: (id) => api.get(`/forecast-runs/${id}/`),
  
  // Create forecast run
  create: (data) => api.post('/forecast-runs/', data),
  
  // Get metrics
  getMetrics: (id) => api.get(`/forecast-runs/${id}/metrics/`),
  
  // Get predictions
  getPredictions: (id, page = 1, pageSize = 100) => 
    api.get(`/forecast-runs/${id}/predictions/?page=${page}&page_size=${pageSize}`),
  
  // Generate explainability
  generateExplainability: (id, maxSamples = 50) => 
    api.post(`/forecast-runs/${id}/generate_explainability/`, { max_samples: maxSamples }),
  
  // Delete forecast run
  delete: (id) => api.delete(`/forecast-runs/${id}/`),
};

// ============================================================
// MODEL COMPARISONS
// ============================================================

export const comparisonAPI = {
  // List all comparisons
  list: () => api.get('/comparisons/'),
  
  // Get comparison by ID
  get: (id) => api.get(`/comparisons/${id}/`),
  
  // Create comparison
  create: (data) => api.post('/comparisons/create_comparison/', data),
  
  // Get chart data
  getChartData: (id) => api.get(`/comparisons/${id}/chart_data/`),
  
  // Delete comparison
  delete: (id) => api.delete(`/comparisons/${id}/`),
};

// ============================================================
// EXPLAINABILITY
// ============================================================

export const explainabilityAPI = {
  // List all explainability results
  list: () => api.get('/explainability/'),
  
  // Get explainability by ID
  get: (id) => api.get(`/explainability/${id}/`),
  
  // Get feature contributions
  getContributions: (id, instanceIdx = 0) => 
    api.get(`/explainability/${id}/feature_contributions/?instance_idx=${instanceIdx}`),
};

// ============================================================
// POLLING UTILITY
// ============================================================

/**
 * Poll forecast run status until completion
 * @param {string} runId - Forecast run ID
 * @param {function} onProgress - Callback for progress updates
 * @param {number} interval - Polling interval in ms
 * @returns {Promise} - Resolves when complete or fails
 */
export const pollForecastStatus = async (runId, onProgress, interval = 2000) => {
  return new Promise((resolve, reject) => {
    const poll = async () => {
      try {
        const response = await forecastAPI.get(runId);
        const { status, progress, error_message } = response.data;
        
        // Call progress callback
        if (onProgress) {
          onProgress({ status, progress });
        }
        
        // Check if complete
        if (status === 'completed') {
          resolve(response.data);
        } else if (status === 'failed') {
          reject(new Error(error_message || 'Forecast failed'));
        } else {
          // Continue polling
          setTimeout(poll, interval);
        }
      } catch (error) {
        reject(error);
      }
    };
    
    poll();
  });
};

// ============================================================
// ERROR HANDLER
// ============================================================

export const handleAPIError = (error) => {
  if (error.response) {
    // Server responded with error
    const message = error.response.data?.error || 
                   error.response.data?.message || 
                   error.response.statusText;
    return { message, status: error.response.status };
  } else if (error.request) {
    // Request made but no response
    return { message: 'No response from server', status: 0 };
  } else {
    // Something else happened
    return { message: error.message, status: -1 };
  }
};

export default api;