# ChronoCast Frontend

React-based dashboard for ChronoCast time series forecasting platform.

## 🚀 Quick Start

### Prerequisites

- Node.js 16+ and npm
- ChronoCast backend running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at `http://localhost:3000`

## 📦 Project Structure

```
frontend/
├── src/
│   ├── pages/
│   │   ├── HomePage.jsx        # Landing page
│   │   ├── UploadPage.jsx      # Dataset upload
│   │   ├── ForecastPage.jsx    # Model configuration
│   │   ├── ResultsPage.jsx     # Forecast results
│   │   └── ComparePage.jsx     # Model comparison
│   ├── services/
│   │   └── api.js              # API client
│   ├── App.jsx                 # Main app component
│   ├── main.jsx                # Entry point
│   └── index.css               # Global styles
├── index.html
├── package.json
├── vite.config.js
└── tailwind.config.js
```

## 🎨 Features

### 1. Upload Dataset
- Drag & drop CSV/Excel files
- File validation
- Data preview
- Column selection

### 2. Generate Forecast
- Model selection (7 algorithms)
- Parameter configuration
- Real-time progress tracking
- Feature engineering options

### 3. View Results
- Interactive Plotly charts
- Evaluation metrics
- Feature importance
- SHAP explainability
- Training information

### 4. Compare Models
- Side-by-side comparison
- Multiple metrics visualization
- Performance ranking
- Summary table

## 🛠️ Technology Stack

- **React 18** - UI library
- **Vite** - Build tool
- **React Router** - Navigation
- **Axios** - HTTP client
- **Plotly.js** - Interactive charts
- **Tailwind CSS** - Styling
- **Lucide React** - Icons

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```env
VITE_API_URL=http://localhost:8000/api
```

### API Proxy

Vite is configured to proxy `/api` requests to the backend:

```javascript
// vite.config.js
server: {
  port: 3000,
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

## 📱 Pages

### Home Page (`/`)
- Feature overview
- Quick action cards
- Statistics
- CTA to start forecasting

### Upload Page (`/upload`)
- File upload interface
- Data validation
- Column selection
- Preview functionality

### Forecast Page (`/forecast`)
- Dataset selection
- Model configuration
- Parameter tuning
- Progress tracking

### Results Page (`/forecast/:runId`)
- Metrics cards
- Forecast visualization
- Error distribution
- Feature importance
- Explainability generation

### Compare Page (`/compare`)
- Dataset selection
- Multi-model selection
- Visual comparisons
- Summary table

## 🎨 Styling

### Tailwind CSS

Configured with custom theme:

```javascript
// tailwind.config.js
theme: {
  extend: {
    colors: {
      primary: {
        // Custom primary color palette
      },
    },
  },
}
```

### Custom Components

- Responsive cards
- Loading states
- Error messages
- Progress bars
- Interactive charts

## 📊 API Integration

### API Service (`src/services/api.js`)

Organized by resource:

```javascript
// Datasets
datasetAPI.list()
datasetAPI.upload(formData)
datasetAPI.validate(id, data)

// Forecasts
forecastAPI.create(data)
forecastAPI.get(id)
forecastAPI.getMetrics(id)

// Comparisons
comparisonAPI.create(data)
comparisonAPI.getChartData(id)
```

### Polling Utility

```javascript
await pollForecastStatus(runId, (progress) => {
  console.log(progress);
}, 2000);
```

## 🧪 Development

### Run Development Server

```bash
npm run dev
```

### Build for Production

```bash
npm run build
```

### Preview Production Build

```bash
npm run preview
```

### Lint Code

```bash
npm run lint
```

## 🐛 Troubleshooting

### API Connection Issues

1. Ensure backend is running on `http://localhost:8000`
2. Check CORS configuration in Django settings
3. Verify proxy configuration in `vite.config.js`

### Chart Not Displaying

1. Check Plotly.js is installed: `npm install plotly.js react-plotly.js`
2. Verify data format matches Plotly requirements
3. Check browser console for errors

### File Upload Fails

1. Check file size (max 10MB)
2. Verify file format (CSV, XLS, XLSX)
3. Check network tab for error details

## 🚀 Deployment

### Build

```bash
npm run build
```

Output will be in `dist/` directory.

### Deploy to Vercel

```bash
npm install -g vercel
vercel
```

### Deploy to Netlify

```bash
npm install -g netlify-cli
netlify deploy --prod --dir=dist
```

### Environment Variables

Set in deployment platform:

```env
VITE_API_URL=https://your-backend-url.com/api
```

## 📈 Performance

- Code splitting by route
- Lazy loading components
- Optimized bundle size
- Responsive images
- Efficient re-renders

## 🔒 Security

- API key handling
- Input validation
- XSS protection
- CORS configuration
- Secure HTTP only

## 📝 Best Practices

- Component composition
- Custom hooks for logic
- Error boundaries
- Loading states
- Responsive design
- Accessibility (WCAG)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

## 📄 License

MIT License

---

**Frontend Status: Complete! ✅**

Ready to forecast! 🚀