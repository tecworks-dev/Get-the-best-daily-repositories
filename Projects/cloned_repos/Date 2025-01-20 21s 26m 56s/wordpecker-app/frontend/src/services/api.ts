import axios, { AxiosInstance } from 'axios';
import { WordList, Word, Exercise, Question } from '../types';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:3000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Add response interceptor to handle errors and unwrap data
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Helper type for API responses
type ApiResponse<T> = Promise<T>;

// Response types
interface LearnStartResponse {
  exercises: Exercise[];
}

interface LearnExercisesResponse {
  exercises: Exercise[];
}

interface QuizStartResponse {
  questions: Question[];
  total_questions: number;
}

interface QuizQuestionsResponse {
  questions: Question[];
  completed: boolean;
}

export const apiService = {
  // Lists
  getLists: (): ApiResponse<WordList[]> => api.get('/lists'),
  getList: (id: string): ApiResponse<WordList> => api.get(`/lists/${id}`),
  createList: (data: Partial<WordList>): ApiResponse<WordList> => api.post('/lists', data),
  updateList: (id: string, data: Partial<WordList>): ApiResponse<WordList> => api.put(`/lists/${id}`, data),
  deleteList: async (listId: string): Promise<void> => {
    await api.delete(`/lists/${listId}`);
  },

  // Words
  getWords: (listId: string): ApiResponse<Word[]> => api.get(`/lists/${listId}/words`),
  addWord: (listId: string, word: string): ApiResponse<Word> => api.post(`/lists/${listId}/words`, { word }),
  deleteWord: (listId: string, wordId: string): ApiResponse<void> => api.delete(`/lists/${listId}/words/${wordId}`),

  // Learning
  startLearning: (listId: string): ApiResponse<LearnStartResponse> => 
    api.post(`/learn/${listId}/start`),
  getExercises: (listId: string): ApiResponse<LearnExercisesResponse> => 
    api.post(`/learn/${listId}/more`),

  // Quiz
  startQuiz: (listId: string): ApiResponse<QuizStartResponse> => 
    api.post(`/quiz/${listId}/start`),
  getQuestions: (listId: string): ApiResponse<QuizQuestionsResponse> => 
    api.post(`/quiz/${listId}/more`)
}; 