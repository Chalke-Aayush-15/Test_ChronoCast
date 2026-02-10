import { useNavigate } from 'react-router-dom';
import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { Upload, BarChart3, GitCompare, Brain, TrendingUp, Zap, ArrowRight, LineChart, BarChart2, Clock, CheckCircle } from 'lucide-react';
import YouTubeAnalytics from '../components/YouTubeAnalytics';

const container = {
  hidden: { opacity: 0 },
  show: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
      delayChildren: 0.3,
    },
  },
};

const item = {
  hidden: { opacity: 0, y: 20 },
  show: { opacity: 1, y: 0, transition: { duration: 0.5 } },
};

export default function HomePage() {
  const navigate = useNavigate();
  
  useEffect(() => {
    // Smooth scroll to top when component mounts
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, []);
  
  const features = [
    {
      icon: Upload,
      title: 'Upload Data',
      description: 'Easily upload your time series data in CSV or Excel format with our intuitive interface',
      action: () => navigate('/upload'),
      color: 'from-blue-500 to-blue-600',
      iconColor: 'text-blue-500',
    },
    {
      icon: BarChart3,
      title: 'Generate Forecast',
      description: 'Train multiple ML models and generate accurate predictions with just a few clicks',
      action: () => navigate('/forecast'),
      color: 'from-emerald-500 to-teal-600',
      iconColor: 'text-emerald-500',
    },
    {
      icon: GitCompare,
      title: 'Compare Models',
      description: 'Evaluate and compare different models side-by-side to find the optimal solution',
      action: () => navigate('/compare'),
      color: 'from-violet-500 to-purple-600',
      iconColor: 'text-violet-500',
    },
  ];
  
  const highlights = [
    {
      icon: Brain,
      title: 'AI-Powered Analytics',
      description: 'Leverage advanced machine learning algorithms including XGBoost, Random Forest, and more',
      color: 'text-primary-500',
    },
    {
      icon: TrendingUp,
      title: 'Accurate Predictions',
      description: 'State-of-the-art forecasting with comprehensive evaluation metrics and confidence intervals',
      color: 'text-emerald-500',
    },
    {
      icon: Zap,
      title: 'Lightning Fast',
      description: 'Process large datasets and get results in minutes with our optimized pipeline',
      color: 'text-amber-500',
    },
  ];

  const stats = [
    { label: 'Models Supported', value: '7+', icon: BarChart2 },
    { label: 'Prediction Accuracy', value: '98%', icon: CheckCircle },
    { label: 'Processing Speed', value: '10x', icon: Zap },
    { label: 'Uptime', value: '99.9%', icon: Clock },
  ];
  
  return (
    <div className="space-y-20 md:space-y-28">
      {/* Hero Section */}
      <motion.section 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        className="relative overflow-hidden"
      >
        <div className="absolute inset-0 bg-gradient-to-br from-primary-500/10 to-primary-600/20 rounded-3xl -z-10" />
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 md:py-28 text-center">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.8 }}
            className="inline-flex items-center px-4 py-2 rounded-full bg-white/10 backdrop-blur-sm border border-white/10 text-sm text-white/90 mb-6"
          >
            <span className="h-2 w-2 rounded-full bg-primary-400 mr-2 animate-pulse"></span>
            Now with enhanced forecasting models
          </motion.div>
          
          <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-700 dark:from-white dark:to-gray-200 leading-tight mb-6">
            AI-Powered Time Series
            <span className="block bg-clip-text text-transparent bg-gradient-to-r from-primary-500 to-primary-600">
              Forecasting Platform
            </span>
          </h1>
          
          <motion.p 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
            className="text-lg md:text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto mb-10"
          >
            Transform your time series data into accurate, actionable forecasts with our advanced machine learning platform. 
            No coding required.
          </motion.p>
          
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6, duration: 0.8 }}
            className="flex flex-col sm:flex-row justify-center gap-4"
          >
            <button
              onClick={() => navigate('/upload')}
              className="group relative inline-flex items-center justify-center px-8 py-4 overflow-hidden font-semibold text-white bg-gradient-to-r from-primary-500 to-primary-600 rounded-xl hover:shadow-lg hover:-translate-y-0.5 transition-all duration-300"
            >
              Get Started Free
              <ArrowRight className="ml-2 h-5 w-5 group-hover:translate-x-1 transition-transform" />
              <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            </button>
            
            <button
              onClick={() => document.getElementById('features')?.scrollIntoView({ behavior: 'smooth' })}
              className="px-8 py-4 font-medium text-gray-700 dark:text-gray-300 hover:text-primary-600 dark:hover:text-primary-400 transition-colors duration-300"
            >
              Learn More
            </button>
          </motion.div>
        </div>
      </motion.section>

      {/* YouTube Analytics Section */}
      <section className="py-12 sm:py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <YouTubeAnalytics />
        </div>
      </section>

      {/* Stats */}
      <motion.section 
        initial={{ opacity: 0, y: 40 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.8 }}
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8"
      >
        <div className="bg-gradient-to-r from-primary-50 to-primary-100 dark:from-gray-800 dark:to-gray-900 rounded-2xl p-8 md:p-10 grid grid-cols-2 md:grid-cols-4 gap-6 md:gap-8">
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <motion.div 
                key={index}
                variants={item}
                className="text-center"
              >
                <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-white dark:bg-gray-800 shadow-sm mb-4">
                  <Icon className="h-7 w-7 text-primary-500" />
                </div>
                <p className="text-3xl font-bold text-gray-900 dark:text-white mb-1">{stat.value}</p>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">{stat.label}</p>
              </motion.div>
            );
          })}
        </div>
      </motion.section>

      {/* Features */}
      <section id="features" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <span className="inline-block px-4 py-1.5 text-sm font-medium rounded-full bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-400 mb-4">
            Powerful Features
          </span>
          <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Everything You Need for Accurate Forecasting
          </h2>
          <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
            Our platform provides a complete suite of tools to transform your time series data into actionable insights.
          </p>
        </div>
        
        <motion.div 
          variants={container}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true }}
          className="grid md:grid-cols-3 gap-8"
        >
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <motion.div 
                key={index}
                variants={item}
                onClick={feature.action}
                className="group relative p-0.5 bg-gradient-to-br rounded-xl from-primary-500 to-secondary-500 shadow-lg cursor-pointer hover:shadow-xl transition-all duration-300"
              >
                <div className="h-full bg-white dark:bg-gray-800 rounded-xl p-6 transition-all duration-300 group-hover:bg-opacity-95 dark:group-hover:bg-gray-800/95">
                  <div className={`inline-flex items-center justify-center w-14 h-14 rounded-xl bg-gradient-to-br ${feature.color} mb-6`}>
                    <Icon className="h-6 w-6 text-white" />
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">{feature.title}</h3>
                  <p className="text-gray-600 dark:text-gray-400 mb-4">{feature.description}</p>
                  <div className="flex items-center text-primary-600 dark:text-primary-400 font-medium group-hover:translate-x-1 transition-transform duration-300">
                    Learn more
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </div>
                </div>
              </motion.div>
            );
          })}
        </motion.div>
      </section>

      {/* Highlights */}
      <section className="relative py-20 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary-50 to-white dark:from-gray-900 dark:to-gray-800 -z-10"></div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <span className="inline-block px-4 py-1.5 text-sm font-medium rounded-full bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-400 mb-4">
              Why Choose Us
            </span>
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Built for Data Scientists and Analysts
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              Our platform is designed to make time series forecasting accessible to everyone, from beginners to experts.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {highlights.map((highlight, index) => {
              const Icon = highlight.icon;
              return (
                <motion.div 
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.1, duration: 0.5 }}
                  className="group relative bg-white dark:bg-gray-800/50 backdrop-blur-sm rounded-xl p-8 shadow-sm hover:shadow-md transition-all duration-300 border border-gray-100 dark:border-gray-700/50"
                >
                  <div className={`w-14 h-14 rounded-xl ${highlight.color} bg-opacity-10 flex items-center justify-center mb-6`}>
                    <Icon className={`h-6 w-6 ${highlight.color}`} />
                  </div>
                  <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">{highlight.title}</h3>
                  <p className="text-gray-600 dark:text-gray-400">{highlight.description}</p>
                  <div className="absolute bottom-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-primary-500 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                </motion.div>
              );
            })}
          </div>
          
          <div className="mt-16 text-center">
            <motion.button
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => navigate('/upload')}
              className="relative inline-flex items-center px-8 py-4 bg-gradient-to-r from-primary-500 to-primary-600 text-white font-semibold rounded-xl overflow-hidden group"
            >
              <span className="relative z-10">Start Forecasting Now</span>
              <div className="absolute inset-0 bg-gradient-to-r from-primary-600 to-primary-700 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              <div className="absolute inset-0 bg-white/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
            </motion.button>
            
            <p className="mt-4 text-sm text-gray-500 dark:text-gray-400">
              No credit card required. Get started in seconds.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}