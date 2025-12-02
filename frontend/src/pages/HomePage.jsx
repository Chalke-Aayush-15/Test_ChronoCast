import { useNavigate } from 'react-router-dom';
import { Upload, BarChart3, GitCompare, Brain, TrendingUp, Zap } from 'lucide-react';

export default function HomePage() {
  const navigate = useNavigate();
  
  const features = [
    {
      icon: Upload,
      title: 'Upload Data',
      description: 'Upload your time series data in CSV or Excel format',
      action: () => navigate('/upload'),
      color: 'blue',
    },
    {
      icon: BarChart3,
      title: 'Generate Forecast',
      description: 'Train models and generate predictions with multiple algorithms',
      action: () => navigate('/forecast'),
      color: 'green',
    },
    {
      icon: GitCompare,
      title: 'Compare Models',
      description: 'Compare multiple models side-by-side to find the best fit',
      action: () => navigate('/compare'),
      color: 'purple',
    },
  ];
  
  const highlights = [
    {
      icon: Brain,
      title: 'AI-Powered',
      description: '7 different ML algorithms including XGBoost, Random Forest, and more',
    },
    {
      icon: TrendingUp,
      title: 'Accurate Predictions',
      description: 'State-of-the-art forecasting with comprehensive evaluation metrics',
    },
    {
      icon: Zap,
      title: 'Fast & Easy',
      description: 'Get results in minutes with our intuitive interface',
    },
  ];
  
  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <div className="text-center">
        <h1 className="text-5xl font-bold text-gray-900 mb-4">
          Welcome to ChronoCast
        </h1>
        <p className="text-xl text-gray-600 max-w-3xl mx-auto">
          A transparent and powerful time series forecasting platform powered by machine learning.
          Upload your data, train models, and get insights in minutes.
        </p>
      </div>
      
      {/* Quick Actions */}
      <div className="grid md:grid-cols-3 gap-6">
        {features.map((feature) => {
          const Icon = feature.icon;
          return (
            <div
              key={feature.title}
              onClick={feature.action}
              className="bg-white p-6 rounded-lg shadow-sm hover:shadow-md transition-shadow cursor-pointer border border-gray-200"
            >
              <div className={`inline-flex p-3 rounded-lg bg-${feature.color}-100 text-${feature.color}-600 mb-4`}>
                <Icon className="h-6 w-6" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-600">
                {feature.description}
              </p>
              <button className="mt-4 text-primary-600 font-medium hover:text-primary-700">
                Get Started →
              </button>
            </div>
          );
        })}
      </div>
      
      {/* Highlights */}
      <div className="bg-gradient-to-r from-primary-600 to-primary-700 rounded-lg p-8 text-white">
        <h2 className="text-3xl font-bold mb-8 text-center">Why ChronoCast?</h2>
        <div className="grid md:grid-cols-3 gap-8">
          {highlights.map((highlight) => {
            const Icon = highlight.icon;
            return (
              <div key={highlight.title} className="text-center">
                <div className="inline-flex p-4 rounded-full bg-white/10 mb-4">
                  <Icon className="h-8 w-8" />
                </div>
                <h3 className="text-xl font-semibold mb-2">{highlight.title}</h3>
                <p className="text-primary-100">{highlight.description}</p>
              </div>
            );
          })}
        </div>
      </div>
      
      {/* Stats */}
      <div className="bg-white rounded-lg shadow-sm p-8">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {[
            { label: 'ML Models', value: '7+' },
            { label: 'Accuracy', value: '95%+' },
            { label: 'Training Time', value: '< 1 min' },
            { label: 'Easy to Use', value: '100%' },
          ].map((stat) => (
            <div key={stat.label} className="text-center">
              <div className="text-3xl font-bold text-primary-600 mb-1">
                {stat.value}
              </div>
              <div className="text-sm text-gray-600">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>
      
      {/* CTA */}
      <div className="text-center">
        <button
          onClick={() => navigate('/upload')}
          className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-md text-white bg-primary-600 hover:bg-primary-700 shadow-sm"
        >
          <Upload className="h-5 w-5 mr-2" />
          Start Forecasting Now
        </button>
      </div>
    </div>
  );
}