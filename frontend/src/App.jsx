import { unstable_HistoryRouter as Router, Routes, Route, Link, useLocation, Outlet } from 'react-router-dom';
import { createBrowserHistory } from 'history';
import { BarChart3, Upload, GitCompare, Home, Moon, Sun, Menu, X, Youtube } from 'lucide-react';
import { useState, useEffect } from 'react';

// Pages
import HomePage from './pages/HomePage';
import UploadPage from './pages/UploadPage';
import ForecastPage from './pages/ForecastPage';
import ComparePage from './pages/ComparePage';
import ResultsPage from './pages/ResultsPage';
import ForecastResultsPage from './pages/ForecastResultsPage';
import YouTubeDashboardPage from './pages/YouTubeDashboardPage';

function Navigation() {
  const location = useLocation();
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [darkMode, setDarkMode] = useState(() => {
    return localStorage.getItem('darkMode') === 'true' || false;
  });

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('darkMode', darkMode);
  }, [darkMode]);
  
  const navItems = [
    { path: '/', label: 'Home', icon: Home },
    { path: '/upload', label: 'Upload Data', icon: Upload },
    { path: '/forecast', label: 'Forecast', icon: BarChart3 },
    { path: '/compare', label: 'Compare Models', icon: GitCompare },
  ];
  
  return (
    <>
      <nav className="bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex items-center">
              <div className="flex-shrink-0 flex items-center">
                <BarChart3 className="h-8 w-8 text-primary-600 dark:text-primary-400" />
                <span className="ml-2 text-xl font-bold bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent dark:from-primary-400 dark:to-secondary-400">
                  ChronoCast
                </span>
              </div>
              <div className="hidden md:ml-10 md:flex md:space-x-1">
                {navItems.map((item) => {
                  const Icon = item.icon;
                  const isActive = location.pathname === item.path;
                  return (
                    <Link
                      key={item.path}
                      to={item.path}
                      className={`inline-flex items-center px-3 py-2 rounded-md text-sm font-medium transition-colors duration-200 ${
                        isActive
                          ? 'bg-primary-50 text-primary-700 dark:bg-gray-800 dark:text-primary-400'
                          : 'text-gray-600 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-800'
                      }`}
                    >
                      <Icon className="h-5 w-5 mr-2" />
                      {item.label}
                    </Link>
                  );
                })}
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <button
                onClick={() => setDarkMode(!darkMode)}
                className="p-2 rounded-full text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800 transition-colors duration-200"
                aria-label="Toggle dark mode"
              >
                {darkMode ? (
                  <Sun className="h-5 w-5" />
                ) : (
                  <Moon className="h-5 w-5" />
                )}
              </button>
              
              <button
                className="md:hidden p-2 rounded-md text-gray-600 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
                onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                aria-label="Toggle menu"
              >
                {isMobileMenuOpen ? (
                  <X className="h-6 w-6" />
                ) : (
                  <Menu className="h-6 w-6" />
                )}
              </button>
            </div>
          </div>
        </div>
        
        {/* Mobile menu */}
        <div className={`md:hidden ${isMobileMenuOpen ? 'block' : 'hidden'}`}>
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  onClick={() => setIsMobileMenuOpen(false)}
                  className={`flex items-center px-3 py-2 rounded-md text-base font-medium ${
                    isActive
                      ? 'bg-primary-50 text-primary-700 dark:bg-gray-800 dark:text-primary-400'
                      : 'text-gray-600 hover:bg-gray-50 dark:text-gray-300 dark:hover:bg-gray-800'
                  }`}
                >
                  <Icon className="h-5 w-5 mr-3" />
                  {item.label}
                </Link>
              );
            })}
          </div>
        </div>
      </nav>
    </>
  );
}

function Layout() {
  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
      <Navigation />
      <main className="py-6 sm:py-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm overflow-hidden transition-all duration-200">
            <div className="p-6">
              <Outlet />
            </div>
          </div>
        </div>
      </main>
      <footer className="bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-800 mt-12">
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500 dark:text-gray-400">
            &copy; {new Date().getFullYear()} ChronoCast. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}

const history = createBrowserHistory();

function App() {
  return (
    <Router 
      history={history}
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true
      }}
    >
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<HomePage />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/forecast" element={<ForecastPage />} />
          <Route path="/forecast/:forecastId" element={<ForecastResultsPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/youtube-dashboard" element={<YouTubeDashboardPage />} />
          <Route path="/youtube" element={<YouTubeDashboardPage />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;