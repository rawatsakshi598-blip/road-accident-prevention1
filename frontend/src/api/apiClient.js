import axios from 'axios';

// API calls go to the same origin (through Caddy gateway or direct)
// The Next.js API proxy routes /api/* to the Python backend on port 8000
const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";
const apiClient = axios.create({
  baseURL: API_BASE,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const api = {
  health: () => apiClient.get('/api/health'),
  edaSummary: () => apiClient.get('/api/eda/summary'),
  edaChart: (chartName, datasetKey = 'primary') =>
    apiClient.get(`/api/eda/charts/${chartName}`, { params: { dataset_key: datasetKey } }),
  modelComparison: () => apiClient.get('/api/models/comparison'),
  modelConfusionMatrix: (modelName) => apiClient.get(`/api/models/${modelName}/confusion-matrix`),
  modelROC: (modelName) => apiClient.get(`/api/models/${modelName}/roc-data`),
  modelMetrics: (modelName) => apiClient.get(`/api/models/${modelName}/metrics`),
  bestModel: () => apiClient.get('/api/models/best'),
  shapFeatureImportance: (modelName = null) =>
    modelName
      ? apiClient.get(`/api/shap/feature-importance/${modelName}`)
      : apiClient.get('/api/shap/feature-importance'),
  shapSummaryPlot: (modelName = null, datasetKey = 'primary') =>
    apiClient.get('/api/shap/summary-plot', { params: { model_name: modelName, dataset_key: datasetKey } }),
  shapBarPlot: (modelName, datasetKey = 'primary') =>
    apiClient.get(`/api/shap/bar-plot/${modelName}`, { params: { dataset_key: datasetKey } }),
  shapAllModels: (datasetKey = 'primary') =>
    apiClient.get('/api/shap/all-models', { params: { dataset_key: datasetKey } }),
  predict: (data) => apiClient.post('/api/predict', data),
  predictBatch: (file, modelName = 'XGBoost') => {
    const formData = new FormData();
    formData.append('file', file);
    return apiClient.post(`/api/predict/batch?model_name=${modelName}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
  },
  availableModels: () => apiClient.get('/api/predict/models'),
  datasetsInfo: () => apiClient.get('/api/datasets/info'),
  filterOptions: () => apiClient.get('/api/filters/options'),
  dataPreview: (datasetKey = 'primary', page = 1, perPage = 25) =>
    apiClient.get(`/api/data/preview/${datasetKey}`, { params: { page, per_page: perPage } }),
  twinCities: () => apiClient.get('/api/twin/cities'),
  twinInitialize: (cityKey, forceRebuild = false) =>
    apiClient.get(`/api/twin/${cityKey}/initialize`, {
      params: { force_rebuild: forceRebuild },
      timeout: 600000,
    }),
  twinMetadata: (cityKey) => apiClient.get(`/api/twin/${cityKey}/metadata`),
  twinHeatmap: (cityKey, type = 'segments', riskThreshold = 0) =>
    apiClient.get(`/api/twin/${cityKey}/heatmap`, {
      params: { type, risk_threshold: riskThreshold },
    }),
  twinTopDangerous: (cityKey, limit = 15, minRisk = 0) =>
    apiClient.get(`/api/twin/${cityKey}/segments/top-dangerous`, {
      params: { limit, min_risk: minRisk },
    }),
  twinSegmentDetails: (cityKey, segmentId) =>
    apiClient.get(`/api/twin/${cityKey}/segment/${segmentId}`),
  twinStats: (cityKey) => apiClient.get(`/api/twin/${cityKey}/stats`),
  twinSimulate: (cityKey, segmentId, scenarioType, params = {}) =>
    apiClient.post(
      `/api/twin/${cityKey}/segment/${segmentId}/simulate`,
      null,
      { params: { scenario_type: scenarioType, ...params } }
    ),
  twinRefresh: (cityKey) => apiClient.post(`/api/twin/${cityKey}/refresh`),
};

export default apiClient;
