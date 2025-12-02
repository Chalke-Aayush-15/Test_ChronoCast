import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { Upload, FileText, CheckCircle, AlertCircle, Loader } from 'lucide-react';
import { datasetAPI, handleAPIError } from '../services/api';

export default function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState(false);
  const [datasetId, setDatasetId] = useState(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [validationStep, setValidationStep] = useState(false);
  const [dateColumn, setDateColumn] = useState('');
  const [targetColumn, setTargetColumn] = useState('');
  
  const onDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setError(null);
      setUploaded(false);
    }
  }, []);
  
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.ms-excel': ['.xls'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
    },
    multiple: false,
  });
  
  const handleUpload = async () => {
    if (!file) return;
    
    setUploading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('name', file.name);
      formData.append('description', `Uploaded on ${new Date().toLocaleString()}`);
      
      const response = await datasetAPI.upload(formData);
      const dataset = response.data;
      
      setDatasetId(dataset.id);
      setUploaded(true);
      
      // Get preview
      const previewResponse = await datasetAPI.preview(dataset.id, 5);
      setPreview(previewResponse.data);
      
      // Move to validation step
      setValidationStep(true);
      
    } catch (err) {
      const { message } = handleAPIError(err);
      setError(message);
    } finally {
      setUploading(false);
    }
  };
  
  const handleValidate = async () => {
    if (!dateColumn || !targetColumn) {
      setError('Please select both date and target columns');
      return;
    }
    
    setUploading(true);
    setError(null);
    
    try {
      await datasetAPI.validate(datasetId, {
        date_column: dateColumn,
        target_column: targetColumn,
      });
      
      // Navigate to forecast page with dataset ID
      navigate(`/forecast?dataset=${datasetId}`);
      
    } catch (err) {
      const { message } = handleAPIError(err);
      setError(message);
    } finally {
      setUploading(false);
    }
  };
  
  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Upload Dataset</h1>
        <p className="text-gray-600">
          Upload your time series data in CSV or Excel format to get started
        </p>
      </div>
      
      {!validationStep ? (
        <>
          {/* File Upload */}
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-primary-400 bg-primary-50'
                : 'border-gray-300 hover:border-gray-400'
            }`}
          >
            <input {...getInputProps()} />
            <Upload className="h-12 w-12 mx-auto text-gray-400 mb-4" />
            {isDragActive ? (
              <p className="text-lg text-gray-600">Drop the file here...</p>
            ) : (
              <div>
                <p className="text-lg text-gray-600 mb-2">
                  Drag & drop your file here, or click to select
                </p>
                <p className="text-sm text-gray-500">
                  Supports CSV, XLS, XLSX (Max 10MB)
                </p>
              </div>
            )}
          </div>
          
          {/* Selected File */}
          {file && (
            <div className="mt-6 bg-white rounded-lg shadow-sm p-4 border border-gray-200">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <FileText className="h-10 w-10 text-primary-600 mr-3" />
                  <div>
                    <p className="font-medium text-gray-900">{file.name}</p>
                    <p className="text-sm text-gray-500">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                {!uploaded && (
                  <button
                    onClick={handleUpload}
                    disabled={uploading}
                    className="px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {uploading ? (
                      <span className="flex items-center">
                        <Loader className="animate-spin h-4 w-4 mr-2" />
                        Uploading...
                      </span>
                    ) : (
                      'Upload'
                    )}
                  </button>
                )}
                {uploaded && (
                  <CheckCircle className="h-6 w-6 text-green-600" />
                )}
              </div>
            </div>
          )}
        </>
      ) : (
        <>
          {/* Validation Step */}
          <div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">
              Validate Dataset
            </h2>
            
            {/* Preview */}
            {preview && (
              <div className="mb-6">
                <p className="text-sm text-gray-600 mb-2">
                  Preview (first 5 rows):
                </p>
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        {preview.columns.map((col) => (
                          <th
                            key={col}
                            className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase"
                          >
                            {col}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {preview.data.map((row, idx) => (
                        <tr key={idx}>
                          {preview.columns.map((col) => (
                            <td
                              key={col}
                              className="px-4 py-2 text-sm text-gray-900"
                            >
                              {row[col]}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
            
            {/* Column Selection */}
            <div className="grid md:grid-cols-2 gap-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Date Column *
                </label>
                <select
                  value={dateColumn}
                  onChange={(e) => setDateColumn(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
                >
                  <option value="">Select date column</option>
                  {preview?.columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Target Column *
                </label>
                <select
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
                >
                  <option value="">Select target column</option>
                  {preview?.columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            
            <button
              onClick={handleValidate}
              disabled={uploading || !dateColumn || !targetColumn}
              className="w-full px-4 py-2 bg-primary-600 text-white rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {uploading ? (
                <span className="flex items-center justify-center">
                  <Loader className="animate-spin h-4 w-4 mr-2" />
                  Validating...
                </span>
              ) : (
                'Continue to Forecast'
              )}
            </button>
          </div>
        </>
      )}
      
      {/* Error Message */}
      {error && (
        <div className="mt-6 bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-start">
            <AlertCircle className="h-5 w-5 text-red-600 mt-0.5 mr-3" />
            <div>
              <h3 className="text-sm font-medium text-red-800">Error</h3>
              <p className="text-sm text-red-700 mt-1">{error}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}