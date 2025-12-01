import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'multipart/form-data',
  },
});

export interface Prediction {
  disease: string;
  probability: number;
}

export interface AnalysisResult {
  predictions: { [key: string]: number };
  top_predictions?: Prediction[];
  detected_diseases?: Array<{
    disease: string;
    status: string;
    is_primary: boolean;
    is_secondary: boolean;
  }>;
  primary_disease?: string;
  secondary_disease?: string;
  total_detected?: number;
  co_occurrence: number[][];
  disease_classes: string[];
  heatmap_overlay: string;
  original_image: string;
  top_disease?: string;
  top_disease_probability?: number;
  show_probabilities?: boolean;
}

export const analyzeImage = async (file: File): Promise<AnalysisResult> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<AnalysisResult>('/api/analyze', formData);
  return response.data;
};

export const healthCheck = async () => {
  const response = await api.get('/api/health');
  return response.data;
};

export default api;
