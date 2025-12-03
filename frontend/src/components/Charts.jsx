/**
 * Reusable Chart Components for ChronoCast
 */

import Plot from 'react-plotly.js';

// Forecast Line Chart
export function ForecastChart({ dates, actual, predicted, title = 'Forecast vs Actual' }) {
  return (
    <Plot
      data={[
        {
          x: dates,
          y: actual,
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Actual',
          line: { color: '#3b82f6', width: 2 },
          marker: { size: 6 },
        },
        {
          x: dates,
          y: predicted,
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Predicted',
          line: { color: '#10b981', width: 2, dash: 'dash' },
          marker: { size: 6, symbol: 'x' },
        },
      ]}
      layout={{
        title: { text: title, font: { size: 16, weight: 600 } },
        autosize: true,
        height: 400,
        xaxis: { title: 'Date', gridcolor: '#e5e7eb' },
        yaxis: { title: 'Value', gridcolor: '#e5e7eb' },
        hovermode: 'x unified',
        showlegend: true,
        legend: { x: 0, y: 1, bgcolor: 'rgba(255,255,255,0.8)' },
        plot_bgcolor: '#f9fafb',
        paper_bgcolor: '#ffffff',
      }}
      useResizeHandler
      style={{ width: '100%' }}
      config={{ responsive: true, displayModeBar: false }}
    />
  );
}

// Residual Histogram
export function ResidualHistogram({ errors, title = 'Error Distribution' }) {
  return (
    <Plot
      data={[
        {
          x: errors,
          type: 'histogram',
          marker: { color: '#8b5cf6', line: { color: '#7c3aed', width: 1 } },
          nbinsx: 30,
        },
      ]}
      layout={{
        title: { text: title, font: { size: 16, weight: 600 } },
        autosize: true,
        height: 300,
        xaxis: { title: 'Prediction Error', gridcolor: '#e5e7eb' },
        yaxis: { title: 'Frequency', gridcolor: '#e5e7eb' },
        showlegend: false,
        plot_bgcolor: '#f9fafb',
        paper_bgcolor: '#ffffff',
      }}
      useResizeHandler
      style={{ width: '100%' }}
      config={{ responsive: true, displayModeBar: false }}
    />
  );
}

// Feature Importance Bar Chart
export function FeatureImportanceChart({ features, importance, title = 'Feature Importance', topN = 10 }) {
  const topFeatures = features.slice(0, topN);
  const topImportance = importance.slice(0, topN);
  
  return (
    <Plot
      data={[
        {
          y: topFeatures.reverse(),
          x: topImportance.reverse(),
          type: 'bar',
          orientation: 'h',
          marker: {
            color: topImportance.reverse(),
            colorscale: 'Viridis',
            showscale: true,
            colorbar: { title: 'Importance' },
          },
        },
      ]}
      layout={{
        title: { text: title, font: { size: 16, weight: 600 } },
        autosize: true,
        height: Math.max(400, topN * 40),
        xaxis: { title: 'Importance Score', gridcolor: '#e5e7eb' },
        yaxis: { title: '', gridcolor: '#e5e7eb' },
        margin: { l: 150, r: 50, t: 50, b: 50 },
        plot_bgcolor: '#f9fafb',
        paper_bgcolor: '#ffffff',
      }}
      useResizeHandler
      style={{ width: '100%' }}
      config={{ responsive: true, displayModeBar: false }}
    />
  );
}

// Scatter Plot (Actual vs Predicted)
export function ScatterPlot({ actual, predicted, title = 'Actual vs Predicted' }) {
  const min = Math.min(...actual, ...predicted);
  const max = Math.max(...actual, ...predicted);
  
  return (
    <Plot
      data={[
        {
          x: predicted,
          y: actual,
          type: 'scatter',
          mode: 'markers',
          marker: { size: 8, color: '#3b82f6', opacity: 0.6 },
          name: 'Predictions',
        },
        {
          x: [min, max],
          y: [min, max],
          type: 'scatter',
          mode: 'lines',
          line: { color: '#ef4444', dash: 'dash', width: 2 },
          name: 'Perfect Prediction',
        },
      ]}
      layout={{
        title: { text: title, font: { size: 16, weight: 600 } },
        autosize: true,
        height: 400,
        xaxis: { title: 'Predicted', gridcolor: '#e5e7eb' },
        yaxis: { title: 'Actual', gridcolor: '#e5e7eb' },
        showlegend: true,
        legend: { x: 0, y: 1, bgcolor: 'rgba(255,255,255,0.8)' },
        plot_bgcolor: '#f9fafb',
        paper_bgcolor: '#ffffff',
      }}
      useResizeHandler
      style={{ width: '100%' }}
      config={{ responsive: true, displayModeBar: false }}
    />
  );
}

// Metrics Bar Chart for Comparison
export function MetricsComparisonChart({ models, metricName, values }) {
  return (
    <Plot
      data={[
        {
          x: models,
          y: values,
          type: 'bar',
          marker: {
            color: models.map((_, idx) => 
              `hsl(${idx * 360 / models.length}, 70%, 50%)`
            ),
          },
        },
      ]}
      layout={{
        title: { text: metricName, font: { size: 16, weight: 600 } },
        autosize: true,
        height: 300,
        xaxis: { title: 'Model', tickangle: -45 },
        yaxis: { title: metricName, gridcolor: '#e5e7eb' },
        showlegend: false,
        plot_bgcolor: '#f9fafb',
        paper_bgcolor: '#ffffff',
        margin: { b: 100 },
      }}
      useResizeHandler
      style={{ width: '100%' }}
      config={{ responsive: true, displayModeBar: false }}
    />
  );
}

// Time Series Decomposition
export function TimeSeriesDecomposition({ dates, original, trend, seasonal }) {
  return (
    <Plot
      data={[
        {
          x: dates,
          y: original,
          type: 'scatter',
          mode: 'lines',
          name: 'Original',
          line: { color: '#3b82f6' },
          yaxis: 'y1',
        },
        {
          x: dates,
          y: trend,
          type: 'scatter',
          mode: 'lines',
          name: 'Trend',
          line: { color: '#10b981', width: 2 },
          yaxis: 'y2',
        },
        {
          x: dates,
          y: seasonal,
          type: 'scatter',
          mode: 'lines',
          name: 'Seasonal',
          line: { color: '#8b5cf6' },
          yaxis: 'y3',
        },
      ]}
      layout={{
        title: { text: 'Time Series Decomposition', font: { size: 16, weight: 600 } },
        autosize: true,
        height: 600,
        xaxis: { domain: [0, 1] },
        yaxis: { title: 'Original', domain: [0.66, 1] },
        yaxis2: { title: 'Trend', domain: [0.33, 0.63] },
        yaxis3: { title: 'Seasonal', domain: [0, 0.3] },
        showlegend: true,
        legend: { x: 0, y: 1.05, orientation: 'h' },
        plot_bgcolor: '#f9fafb',
        paper_bgcolor: '#ffffff',
      }}
      useResizeHandler
      style={{ width: '100%' }}
      config={{ responsive: true, displayModeBar: false }}
    />
  );
}

// Loading Skeleton for Charts
export function ChartSkeleton({ height = 400 }) {
  return (
    <div 
      className="animate-pulse bg-gray-200 rounded-lg"
      style={{ height: `${height}px`, width: '100%' }}
    >
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-400">Loading chart...</div>
      </div>
    </div>
  );
}